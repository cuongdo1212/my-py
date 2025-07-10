from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

app = Flask(__name__)

# === Cấu hình ===
MODEL_PATH = "sign_model.h5"
MAX_FRAMES = 60
LABEL_MAP = ["ban", "cam_on", "toi", "xin_chao"]

# === Load mô hình ===
model = load_model(MODEL_PATH)

# === Khởi tạo MediaPipe ===
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# === Hàm trích xuất keypoints ===
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
    return np.concatenate([pose, left_hand, right_hand, face]).flatten()

# === Video Capture ===
def generate_frames():
    cap = cv2.VideoCapture(0)
    sequence = []
    prev_label = ""
    last_predict_time = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            if len(sequence) > MAX_FRAMES:
                sequence = sequence[-MAX_FRAMES:]

            if len(sequence) == MAX_FRAMES and (time.time() - last_predict_time) > 1:
                input_data = np.expand_dims(sequence, axis=0)
                prediction = model.predict(input_data)[0]
                confidence = np.max(prediction)
                label = LABEL_MAP[np.argmax(prediction)]
                last_predict_time = time.time()

                if confidence > 0.8:
                    prev_label = f"{label} ({confidence*100:.1f}%)"

            cv2.rectangle(image, (0, 0), (300, 60), (0, 0, 0), -1)
            cv2.putText(image, f"Ket qua: {prev_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
