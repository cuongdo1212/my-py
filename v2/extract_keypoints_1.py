import os
import cv2
import numpy as np
import mediapipe as mp

# Đường dẫn thư mục video
VIDEO_FOLDER = "videodata"
OUTPUT_FOLDER = "keypoints"

# Tạo thư mục lưu kết quả nếu chưa có
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Khởi tạo MediaPipe
mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
    return np.concatenate([pose, left_hand, right_hand, face]).flatten()

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

        cap.release()

    sequence = np.array(sequence)
    np.save(output_path, sequence)

def main():
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")]

    for video_file in video_files:
        input_path = os.path.join(VIDEO_FOLDER, video_file)
        output_name = os.path.splitext(video_file)[0] + ".npy"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        print(f"⏳ Processing {video_file}...")
        process_video(input_path, output_path)
        print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    main()
