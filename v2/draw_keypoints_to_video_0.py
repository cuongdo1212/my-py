import os
import cv2
import mediapipe as mp

# Thư mục chứa video gốc
VIDEO_FOLDER = "videodata"
# Thư mục lưu video có vẽ keypoint
OUTPUT_FOLDER = "key_point_test"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Cấu hình MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Cấu hình màu đỏ cho các keypoint
RED_DRAW_STYLE = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

def draw_and_save_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Ghi video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Vẽ keypoint đỏ
            if results.face_landmarks:
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                          RED_DRAW_STYLE, RED_DRAW_STYLE)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          RED_DRAW_STYLE, RED_DRAW_STYLE)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          RED_DRAW_STYLE, RED_DRAW_STYLE)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          RED_DRAW_STYLE, RED_DRAW_STYLE)

            out.write(image)

        cap.release()
        out.release()
        print(f"✅ Saved keypoint video to: {output_path}")

def main():
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")]
    for video_file in video_files:
        input_path = os.path.join(VIDEO_FOLDER, video_file)
        output_path = os.path.join(OUTPUT_FOLDER, video_file)
        print(f"🎞️ Processing: {video_file}")
        draw_and_save_video(input_path, output_path)

if __name__ == "__main__":
    main()
