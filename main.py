import os
import io
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# ========== Model Definition ==========
class PoseDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 128)
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 128),
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),
        )
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.block1(x)
        x = self.block2(x)
        return self.head(x)

# ========== Load Models ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler_final.pkl"))
INPUT_DIM = scaler.n_features_in_
model = PoseDNN(input_dim=INPUT_DIM)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "fold_3_best.pth"), map_location="cpu"))
model.eval()
yolo = YOLO(os.path.join(BASE_DIR, "yolo11n-pose.pt"))

REQUIRED_KPS = [0, 5, 6, 11, 12]
SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

# ========== Helpers ==========
def angle(a, b, c):
    v1, v2 = a - b, c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def compute_features(kps_xy):
    k = kps_xy.reshape(17, 2)
    torso = np.linalg.norm((k[5]+k[6])/2 - (k[11]+k[12])/2) + 1e-8
    angles = [
        angle(k[5],k[7],k[9]), angle(k[6],k[8],k[10]),
        angle(k[11],k[13],k[15]), angle(k[12],k[14],k[16]),
        angle(k[11],k[5],k[7]), angle(k[12],k[6],k[8]),
        angle(k[5],k[11],k[13]), angle(k[6],k[12],k[14]),
    ]
    ratios = [
        np.linalg.norm(k[5]-k[6])/torso, np.linalg.norm(k[11]-k[12])/torso,
        np.linalg.norm(k[9]-k[10])/torso, np.linalg.norm(k[5]-k[9])/torso,
        np.linalg.norm(k[6]-k[10])/torso, np.linalg.norm(k[15]-k[16])/torso,
    ]
    return angles + ratios

def draw_overlay(frame, kps_xyn, kps_conf, box, label, prob, color):
    h, w = frame.shape[:2]
    pts = (kps_xyn * np.array([w, h])).astype(int)
    for i, j in SKELETON:
        if kps_conf[i] > 0.3 and kps_conf[j] > 0.3:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, 2, cv2.LINE_AA)
    for pt, conf in zip(pts, kps_conf):
        if conf > 0.3:
            cv2.circle(frame, tuple(pt), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pt), 2, (255,255,255), -1, cv2.LINE_AA)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    text = f"{label} {prob:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - 6), font, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return frame

def process_frame(frame):
    res = yolo(frame, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints) == 0:
        return None
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in res.boxes.xyxy.cpu().numpy()]
    idx = int(np.argmax(areas))
    kps_xyn = res.keypoints.xyn[idx].cpu().numpy()
    kps_data = res.keypoints.data[idx].cpu().numpy()
    box = res.boxes.xyxy[idx].cpu().numpy()
    if np.any(kps_data[REQUIRED_KPS, 2] < 0.3):
        return None
    features = kps_data.flatten().tolist() + compute_features(kps_xyn.flatten())
    X = scaler.transform([features])
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        prob = torch.softmax(logits, dim=1)[0]
        pred = logits.argmax(dim=1).item()
    label = "ABNORMAL" if pred == 1 else "Normal"
    color = (0, 0, 255) if pred == 1 else (0, 200, 0)
    prob_val = prob[1].item() if pred == 1 else prob[0].item()
    return {"pred": pred, "label": label, "prob": prob_val, "color": color,
            "kps_xyn": kps_xyn, "kps_conf": kps_data[:, 2], "box": box}

# ========== FastAPI App ==========
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

@app.post("/analyze_image")
async def analyze_image(image: UploadFile = File(...)):
    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(pil_img)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    result = process_frame(frame_bgr)
    if result is None:
        return JSONResponse({"error": "No person detected"}, status_code=400)
    frame_bgr = draw_overlay(frame_bgr, result["kps_xyn"], result["kps_conf"],
                             result["box"], result["label"], result["prob"], result["color"])
    _, buffer = cv2.imencode(".jpg", frame_bgr)
    img_base64 = base64.b64encode(buffer).decode()
    return {
        "status": result["label"],
        "confidence": result["prob"],
        "image": f"data:image/jpeg;base64,{img_base64}",
        "details": []
    }

@app.post("/analyze_video")
async def analyze_video(video: UploadFile = File(...)):
    contents = await video.read()
    tmp_path = "temp_video.mp4"
    with open(tmp_path, "wb") as f:
        f.write(contents)
    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    events = []
    normal_frames = 0
    abnormal_frames = 0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % max(1, int(fps / 2)) == 0:   # analyze every ~0.5s
            result = process_frame(frame)
            if result:
                timestamp = frame_idx / fps
                if result["pred"] == 1:
                    abnormal_frames += 1
                    events.append({
                        "time": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                        "status": "cheating" if result["pred"]==1 else "safe",
                        "description": "Gian lận" if result["pred"]==1 else "Bình thường"
                    })
                else:
                    normal_frames += 1
        frame_idx += 1
    cap.release()
    os.unlink(tmp_path)
    return {
        "events": events,
        "stats": {
            "total": normal_frames + abnormal_frames,
            "safe": normal_frames,
            "suspicious": 0,
            "cheating": abnormal_frames
        }
    }