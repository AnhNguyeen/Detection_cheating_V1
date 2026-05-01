import torch
import os
import cv2
import torch.nn as nn
import numpy as np
import joblib
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

scaler    = joblib.load(os.path.join(BASE_DIR, 'scaler_final.pkl'))
INPUT_DIM = scaler.n_features_in_
model     = PoseDNN(input_dim=INPUT_DIM)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'fold_3_best.pth'), map_location='cpu'))
model.eval()

yolo = YOLO(os.path.join(BASE_DIR, 'yolo11n-pose.pt'))

REQUIRED_KPS = [0, 5, 6, 11, 12]

# Các đường nối keypoint thành skeleton
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),           # đầu
    (5,6),                              # vai
    (5,7),(7,9),                        # tay trái
    (6,8),(8,10),                       # tay phải
    (5,11),(6,12),(11,12),              # thân
    (11,13),(13,15),                    # chân trái
    (12,14),(14,16),                    # chân phải
]

def angle(a, b, c):
    v1, v2 = a - b, c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def compute_features(kps_xy):
    k = kps_xy.reshape(17, 2)
    torso = np.linalg.norm((k[5]+k[6])/2 - (k[11]+k[12])/2) + 1e-8
    angles = [
        angle(k[5],k[7],k[9]),    angle(k[6],k[8],k[10]),
        angle(k[11],k[13],k[15]), angle(k[12],k[14],k[16]),
        angle(k[11],k[5],k[7]),   angle(k[12],k[6],k[8]),
        angle(k[5],k[11],k[13]),  angle(k[6],k[12],k[14]),
    ]
    ratios = [
        np.linalg.norm(k[5]-k[6])/torso,   np.linalg.norm(k[11]-k[12])/torso,
        np.linalg.norm(k[9]-k[10])/torso,  np.linalg.norm(k[5]-k[9])/torso,
        np.linalg.norm(k[6]-k[10])/torso,  np.linalg.norm(k[15]-k[16])/torso,
    ]
    return angles + ratios

def draw_skeleton(frame, kps_xy, kps_conf, color):
    """Vẽ skeleton và keypoint lên frame."""
    h, w = frame.shape[:2]
    pts  = (kps_xy * np.array([w, h])).astype(int)

    # Vẽ đường xương
    for i, j in SKELETON:
        if kps_conf[i] > 0.3 and kps_conf[j] > 0.3:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, 2, cv2.LINE_AA)

    # Vẽ keypoint
    for idx, (pt, conf) in enumerate(zip(pts, kps_conf)):
        if conf > 0.3:
            cv2.circle(frame, tuple(pt), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pt), 4, (255,255,255), 1, cv2.LINE_AA)

def draw_bbox(frame, box, color, label, prob):
    """Vẽ bounding box + nhãn."""
    x1, y1, x2, y2 = map(int, box)

    # Box với độ dày theo kết quả
    thickness = 3
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # Background nhãn
    text      = f"{label}  {prob:.0%}"
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - 6), font, font_scale, (255,255,255), 2, cv2.LINE_AA)

def predict_video(video_path, output_path=None, skip_frames=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không mở được: {video_path}")
        return

    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (w, h))

    frame_idx      = 0
    normal_count   = 0
    abnormal_count = 0

    # Cache kết quả frame trước (để frame bị skip vẫn có label)
    last_result = {
        'pred': None, 'prob_n': 0, 'prob_a': 0,
        'box': None, 'kps_xy': None, 'kps_conf': None
    }

    print(f"Video: {total} frames | {fps:.1f} FPS | {w}x{h}")
    print("Nhấn Q để dừng\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % skip_frames == 0:
            res = yolo(frame, verbose=False)[0]

            if res.keypoints is not None and len(res.keypoints) > 0:
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in res.boxes.xyxy.cpu().numpy()]
                idx   = int(np.argmax(areas))

                kps_xyn  = res.keypoints.xyn[idx].cpu().numpy()
                kps_data = res.keypoints.data[idx].cpu().numpy()
                box      = res.boxes.xyxy[idx].cpu().numpy()

                if not np.any(kps_data[REQUIRED_KPS, 2] < 0.3):
                    features = kps_data.flatten().tolist() + compute_features(kps_xyn.flatten())
                    X        = scaler.transform([features])

                    with torch.no_grad():
                        logits = model(torch.tensor(X, dtype=torch.float32))
                        prob   = torch.softmax(logits, dim=1)[0]
                        pred   = logits.argmax(dim=1).item()

                    last_result = {
                        'pred':     pred,
                        'prob_n':   prob[0].item(),
                        'prob_a':   prob[1].item(),
                        'box':      box,
                        'kps_xy':   kps_xyn,
                        'kps_conf': kps_data[:, 2],
                    }

                    if pred == 1:
                        abnormal_count += 1
                    else:
                        normal_count   += 1

        # ── Vẽ lên frame ──
        if last_result['pred'] is not None:
            pred     = last_result['pred']
            color    = (0, 0, 255) if pred == 1 else (0, 200, 0)  # đỏ / xanh
            label    = "ABNORMAL" if pred == 1 else "Normal"
            prob_val = last_result['prob_a'] if pred == 1 else last_result['prob_n']

            draw_skeleton(frame, last_result['kps_xy'],
                          last_result['kps_conf'], color)
            draw_bbox(frame, last_result['box'], color, label, prob_val)

        # Thống kê góc trái dưới
        total_pred = normal_count + abnormal_count
        stats_text = (f"Normal: {normal_count} | "
                      f"Abnormal: {abnormal_count} | "
                      f"Frame: {frame_idx}/{total}")
        cv2.rectangle(frame, (0, h-30), (600, h), (0,0,0), -1)
        cv2.putText(frame, stats_text, (8, h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv2.LINE_AA)

        cv2.imshow('Cheat Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    total_pred = normal_count + abnormal_count
    print(f"{'='*40}")
    print(f"Tổng frames: {total_pred}")
    print(f"Normal  : {normal_count}  ({normal_count/max(total_pred,1)*100:.1f}%)")
    print(f"Abnormal: {abnormal_count} ({abnormal_count/max(total_pred,1)*100:.1f}%)")
    if output_path:
        print(f"Đã lưu: {output_path}")


# ── Chạy ──
VIDEO_INPUT  = r'D:\Project\Nguyen_Trong_Anh\IMG_0641.MOV'
VIDEO_OUTPUT = r'D:\Project\Nguyen_Trong_Anh\video_result.mp4'

predict_video(VIDEO_INPUT, output_path=VIDEO_OUTPUT, skip_frames=2)