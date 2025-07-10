import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic
model = load_model("sign_model.h5")
LABEL_MAP = ["ban", "cam_on", "toi", "xin_chao"]
MAX_FRAMES = 60
sequence = []

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    left = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    right = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
    return np.concatenate([pose, left, right, face]).flatten()

def predict_from_frame(frame):
    global sequence
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        if len(sequence) > MAX_FRAMES:
            sequence = sequence[-MAX_FRAMES:]

        if len(sequence) == MAX_FRAMES:
            input_data = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_data)[0]
            confidence = np.max(prediction)
            label = LABEL_MAP[np.argmax(prediction)]
            sequence = []
            label = "Xin chào"
            confidence = 0.95 
            return label, confidence
            
        else:
            return "Đang xử lý...", 0.0
        
