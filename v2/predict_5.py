# === Thư viện cần dùng ===
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# === Cấu hình ===
MODEL_PATH = "sign_model.h5"              # Đường dẫn tới model đã huấn luyện
MAX_FRAMES = 60                           # Số lượng khung hình để gom lại 1 lần dự đoán
LABEL_MAP = ["ban", "cam_on", "toi", "xin_chao"]  # Danh sách nhãn của các cử chỉ

# === Load model đã huấn luyện ===
model = load_model(MODEL_PATH)

# === Khởi tạo MediaPipe Holistic (bao gồm pose + face + hai tay) ===
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# === Hàm trích xuất keypoints từ kết quả MediaPipe ===
def extract_keypoints(results):
    # 33 điểm trên cơ thể (pose)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    # 21 điểm tay trái
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    # 21 điểm tay phải
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    # 468 điểm khuôn mặt
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
    # Ghép tất cả thành 1 vector phẳng
    return np.concatenate([pose, left_hand, right_hand, face]).flatten()

# === Khởi tạo các biến cần thiết ===
sequence = []        # Lưu chuỗi keypoints liên tiếp
history = []         # Lưu kết quả dự đoán gần nhất
cap = cv2.VideoCapture(0)  # Mở webcam

# === Sử dụng MediaPipe Holistic ===
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    prev_label = ""                  # Nhãn dự đoán trước đó
    last_predict_time = time.time() # Thời điểm dự đoán lần cuối

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Lật ảnh và chuyển đổi sang RGB
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Trích xuất landmark
        results = holistic.process(image)

        # Chuyển lại sang ảnh BGR để vẽ
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Vẽ landmark lên ảnh (mặt, dáng, tay)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Trích xuất keypoints và thêm vào chuỗi
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # Chỉ giữ lại MAX_FRAMES khung hình gần nhất
        if len(sequence) > MAX_FRAMES:
            sequence = sequence[-MAX_FRAMES:]

        # Nếu đủ số khung và đủ thời gian chờ → dự đoán
        if len(sequence) == MAX_FRAMES and (time.time() - last_predict_time) > 1:
            input_data = np.expand_dims(sequence, axis=0)           # Đưa vào đúng định dạng batch
            prediction = model.predict(input_data)[0]               # Dự đoán
            confidence = np.max(prediction)                         # Lấy độ tin cậy cao nhất
            label = LABEL_MAP[np.argmax(prediction)]               # Lấy nhãn tương ứng

            last_predict_time = time.time()                         # Cập nhật thời gian dự đoán

            # Nếu độ tin cậy cao hơn 80% → chấp nhận kết quả
            if confidence > 0.8:
                prev_label = f"{label} ({confidence*100:.1f}%)"     # Ghi lại nhãn và % tin cậy
                history.append(prev_label)                          # Thêm vào lịch sử
                if len(history) > 5:
                    history = history[-5:]                          # Chỉ giữ lại 5 kết quả gần nhất

        # === Hiển thị kết quả lên ảnh ===
        cv2.rectangle(image, (0, 0), (350, 120), (0, 0, 0), -1)      # Khung đen phía trên
        cv2.putText(image, f"Ket qua: {prev_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Hiển thị lịch sử kết quả
        for i, item in enumerate(reversed(history)):
            cv2.putText(image, f"{item}", (10, 60 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Hiển thị cửa sổ video
        cv2.imshow('Realtime Sign Language Recognition', image)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
