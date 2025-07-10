import os
import numpy as np

DATA_DIR = "keypoints"
label_file = "label_map.txt"

# Đọc ánh xạ ID → label
label_map_raw = {}
with open(label_file, "r") as f:
    for line in f:
        video_id, label = line.strip().split(",")
        label_map_raw[video_id] = label

# Đánh số label
unique_labels = sorted(set(label_map_raw.values()))
label_map = {label: idx for idx, label in enumerate(unique_labels)}

X = []
y = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".npy"):
        keypoint_path = os.path.join(DATA_DIR, file)
        data = np.load(keypoint_path)

        max_frames = 60
        if data.shape[0] > max_frames:
            data = data[:max_frames]
        else:
            pad_len = max_frames - data.shape[0]
            data = np.pad(data, ((0, pad_len), (0, 0)), mode='constant')

        X.append(data)

        # Lấy video ID (không có .npy)
        video_id = file.replace(".npy", "")
        label = label_map_raw[video_id]
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)
print("done")