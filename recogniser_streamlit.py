"""
Streamlit app for ISL static sign recognition (supports 1 or 2 hands).
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from joblib import load
from collections import deque
from utils_isl import pad_or_truncate, TARGET_VECTOR_LEN, PER_HAND_LEN

# Import WebRTC components
from streamlit_webrtc import webrtc_streamer, VideoFrame, WebRtcMode

st.set_page_config(page_title="ISL Sign Recognizer", layout="centered",
                   page_icon="🤟", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main > div.block-container{ max-width: 900px; }
h1 { color: #1f2937; }
.stButton>button { background-color: #111827; color: white; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("ISL Static Sign Recognizer (1 & 2 hand support)")
st.write("Upload an image or use webcam for real-time recognition (static ISL gestures only).")

# Sidebar controls
st.sidebar.header("Model / Options")

# Uncomment these lines for easier control and testing while running on local machine
# model_path = st.sidebar.text_input("Model file", value="model_isl.p")
# labels_file = st.sidebar.text_input("Labels file", value="labels_isl.txt")

model_path = "model_isl.p"
labels_file = "labels_isl.txt"

conf_threshold = st.sidebar.slider("Confidence threshold (not used for predict-only model)", 0.0, 1.0, 0.5)
frames_skip = st.sidebar.number_input("Predict every N frames (higher = faster)", min_value=1, max_value=10, value=3)
smoothing_k = st.sidebar.number_input("Smoothing window size (majority vote)", min_value=1, max_value=20, value=5)

# Load Model & Labels (Same as before)
model = None
scaler = None
labels_list = []
if os.path.exists(model_path):
    try:
        mpack = load(model_path)
        if isinstance(mpack, dict):
            model = mpack.get('model')
            scaler = mpack.get('scaler')
        else:
            model = mpack
            scaler = None
        st.sidebar.success("Model loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
else:
    st.sidebar.warning("model file not found. Train and provide model_isl.p")

if os.path.exists(labels_file):
    with open(labels_file, 'r') as f:
        labels_list = [l.strip() for l in f.readlines()]
else:
    st.sidebar.warning("labels_isl.txt not found. Place it in the app folder.")

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       min_detection_confidence=0.5,
                       max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Landmark/Prediction Functions
def build_two_hand_vector_from_results(results):
    vec = [0.0] * TARGET_VECTOR_LEN
    if not results.multi_hand_landmarks:
        return np.array(vec, dtype=np.float32)
    
    hand_labels = []
    if results.multi_handedness:
        for h in results.multi_handedness:
            try: hand_labels.append(h.classification[0].label)
            except Exception: hand_labels.append('Unknown')
                
    hand_map = {}
    for idx, lm in enumerate(results.multi_hand_landmarks):
        lab = hand_labels[idx] if idx < len(hand_labels) else f'hand{idx}'
        x_ = [p.x for p in lm.landmark]
        y_ = [p.y for p in lm.landmark]
        x_min, y_min = min(x_), min(y_)
        coords = []
        for i in range(len(lm.landmark)):
            coords.append(lm.landmark[i].x - x_min)
            coords.append(lm.landmark[i].y - y_min)
        coords = pad_or_truncate(coords, PER_HAND_LEN).tolist()
        if lab not in hand_map:
            hand_map[lab] = coords
            
    num_hands = len(hand_map)
    left = hand_map.get('Left', [0.0]*PER_HAND_LEN)
    right = hand_map.get('Right', [0.0]*PER_HAND_LEN)

    final_vec = []
    if num_hands == 1:
        if 'Left' in hand_map: final_vec = left + right
        else: final_vec = right + left
    else:
        final_vec = left + right
        
    return np.array(final_vec, dtype=np.float32)

def predict_vector(vec):
    if scaler is not None:
        v = scaler.transform([vec])[0]
    else:
        v = vec
    if model is None:
        return None
    try:
        pred = model.predict([v])[0]
        return pred
    except Exception:
        try:
            pred = model.classes_[model.predict([v])[0]]
            return pred
        except Exception:
            return None

# Image Upload Logic
col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload an image", type=['png','jpg','jpeg'])
with col2:
    use_cam = st.checkbox("Use webcam", value=False)

if uploaded:
    img = uploaded.read()
    nparr = np.frombuffer(img, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    vec = build_two_hand_vector_from_results(results)
    if not np.allclose(vec, 0.0):
        pred = predict_vector(vec)
        if pred is None:
            st.info("Model not loaded or couldn't predict.")
        else:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.markdown(f"### Prediction: **{pred}**")
    else:
        st.info("No hands detected. Try another image or improve lighting.")

# WebRTC Implementation
# We use a class to store the state (buffer, counter, etc.)
class SignRecognizer:
    def __init__(self):
        self.pred_buffer = deque(maxlen=smoothing_k)
        self.last_display_pred = None
        self.skip_counter = 0

    def video_frame_callback(self, frame: VideoFrame) -> VideoFrame:
        # Decode the video frame
        img = frame.to_ndarray(format="bgr24")
        
        # Flip frame horizontally (like a mirror)
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with mediapipe
        results = hands.process(rgb)
        
        # Draw landmarks
        display_frame = img.copy()
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(display_frame, hl, mp_hands.HAND_CONNECTIONS)

        # Prediction logic (same as old loop)
        if self.skip_counter % int(frames_skip) == 0:
            vec = build_two_hand_vector_from_results(results)
            if not np.allclose(vec, 0.0) and model is not None:
                pred = predict_vector(vec)
                if pred is not None:
                    self.pred_buffer.append(pred)
            else:
                self.pred_buffer.append(None)
            
            # Majority vote
            items = [p for p in self.pred_buffer if p is not None]
            if items:
                voted = max(set(items), key=items.count)
                self.last_display_pred = voted
            else:
                self.last_display_pred = None
        self.skip_counter += 1

        # Overlay label
        if self.last_display_pred:
            cv2.putText(display_frame, f"{self.last_display_pred}", (30,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2, cv2.LINE_AA)
        
        # Encode and return the frame
        return VideoFrame.from_ndarray(display_frame, format="bgr24")

# This is the main change: replace the old `if use_cam:` block
if use_cam:
    st.write("Click 'START' to open your webcam.")
    
    # The webrtc_streamer component
    webrtc_streamer(
        key="sign-recognizer",
        mode=WebRtcMode.SENDRECV,
        # We pass the callback from an *instance* of our class
        video_frame_callback=SignRecognizer().video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )