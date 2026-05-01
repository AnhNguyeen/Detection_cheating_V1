import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import joblib
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

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

@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler_final.pkl'))
    INPUT_DIM = scaler.n_features_in_
    dnn_model = PoseDNN(input_dim=INPUT_DIM)
    dnn_model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'fold_3_best.pth'), map_location='cpu'))
    dnn_model.eval()
    yolo_model = YOLO(os.path.join(BASE_DIR, 'yolo11n-pose.pt'))
    return scaler, dnn_model, yolo_model

scaler, model, yolo = load_models()

REQUIRED_KPS = [0, 5, 6, 11, 12]
SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

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

def draw_skeleton(frame, kps_xy, kps_conf, color):
    h, w = frame.shape[:2]
    pts = (kps_xy * np.array([w, h])).astype(int)
    for i, j in SKELETON:
        if kps_conf[i] > 0.3 and kps_conf[j] > 0.3:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, 2, cv2.LINE_AA)
    for pt, conf in zip(pts, kps_conf):
        if conf > 0.3:
            cv2.circle(frame, tuple(pt), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pt), 2, (255,255,255), -1, cv2.LINE_AA)

def draw_bbox(frame, box, color, label, prob):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    text = f"{label} {prob:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - 6), font, 0.7, (255,255,255), 2, cv2.LINE_AA)

def process_frame(frame):
    res = yolo(frame, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints) == 0:
        return frame, None, 0, 0

    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in res.boxes.xyxy.cpu().numpy()]
    idx = int(np.argmax(areas))
    kps_xyn = res.keypoints.xyn[idx].cpu().numpy()
    kps_data = res.keypoints.data[idx].cpu().numpy()
    box = res.boxes.xyxy[idx].cpu().numpy()

    if np.any(kps_data[REQUIRED_KPS, 2] < 0.3):
        return frame, None, 0, 0

    features = kps_data.flatten().tolist() + compute_features(kps_xyn.flatten())
    X = scaler.transform([features])

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        prob = torch.softmax(logits, dim=1)[0]
        pred = logits.argmax(dim=1).item()

    color = (0, 0, 255) if pred == 1 else (0, 200, 0)
    label = "ABNORMAL" if pred == 1 else "Normal"
    prob_val = prob[1].item() if pred == 1 else prob[0].item()

    draw_skeleton(frame, kps_xyn, kps_data[:, 2], color)
    draw_bbox(frame, box, color, label, prob_val)

    return frame, pred, prob[0].item(), prob[1].item()

st.set_page_config(page_title="Cheat Detection", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main { background-color: #fafafa; }
    .stTabs [data-baseweb="tab"] { color: #333; }
    .stTabs [data-baseweb="tab"]:hover { color: #000; }
    .stTabs [aria-selected="true"] { color: #000; border-bottom: 2px solid #000; }
    .stButton>button { background-color: #222; color: white; border: none; }
    .stButton>button:hover { background-color: #444; }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("Cheat Detection")
st.caption("Detect abnormal behaviour in images and videos.")

tab1, tab2 = st.tabs(["Image Analysis", "Video Analysis"])

with tab1:
    uploaded_img = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'webp'])
    if uploaded_img is not None:
        image = Image.open(uploaded_img).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        with st.spinner("Analysing..."):
            processed_frame, pred, prob_n, prob_a = process_frame(frame)

        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(processed_frame_rgb, use_container_width=True)
        with col2:
            st.markdown("### Result")
            if pred is None:
                st.markdown("No person detected or keypoints occluded.")
            else:
                if pred == 1:
                    st.markdown(f"**ABNORMAL**  \nConfidence: {prob_a:.1%}")
                else:
                    st.markdown(f"**NORMAL**  \nConfidence: {prob_n:.1%}")

with tab2:
    uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        start_button = st.button("Start Analysis")
        if start_button:
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()

            normal_count = 0
            abnormal_count = 0
            frame_count = 0
            frame_skip = 2

            info_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip == 0:
                    processed_frame, pred, _, _ = process_frame(frame)

                    if pred == 1:
                        abnormal_count += 1
                    elif pred == 0:
                        normal_count += 1

                    info_placeholder.markdown(
                        f"Normal frames: **{normal_count}**  \n"
                        f"Abnormal frames: **{abnormal_count}**"
                    )

                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(processed_frame_rgb, channels="RGB", use_container_width=True)

            cap.release()
            st.markdown("Analysis completed.")