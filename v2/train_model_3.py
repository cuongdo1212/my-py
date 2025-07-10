import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Kiểm tra dữ liệu đã tồn tại chưa
if not os.path.exists("X.npy") or not os.path.exists("y.npy"):
    raise FileNotFoundError("Chưa có file X.npy hoặc y.npy. Hãy chạy prepare_dataset.py trước.")

# Load dữ liệu đã gán nhãn
X = np.load("X.npy")
y = np.load("y.npy")

# One-hot encoding cho nhãn
y = to_categorical(y)

# Tách train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_test, y_test))

# Lưu mô hình
model.save("sign_model.h5")
print("✅ Đã lưu mô hình vào sign_model.h5")
