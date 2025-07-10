
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
from PIL import Image, ImageTk
import threading
import json

class SmartVideoCutter:
    def __init__(self, root):
        self.root = root
        self.root.title("✂️ Cắt Video Tự Động Theo Danh Sách")

        self.source_folder = ""
        self.save_folder = ""
        self.video_list = []
        self.current_video_index = 0
        self.current_video_path = ""
        self.cut_log = {}
        self.cut_log_file = ""
        self.start_frame = None
        self.end_frame = None

        self.cap = None
        self.fps = 30
        self.total_frames = 0
        self.current_frame = 0
        self.playing = False

        # UI
        self.canvas = tk.Canvas(root, width=640, height=360)
        self.canvas.grid(row=0, column=0, columnspan=3)

        self.scale = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, length=640, command=self.on_scale)
        self.scale.grid(row=1, column=0, columnspan=3)

        tk.Button(root, text="📁 Chọn thư mục nguồn", command=self.select_source_folder).grid(row=2, column=0)
        tk.Button(root, text="💾 Chọn thư mục lưu", command=self.select_save_folder).grid(row=2, column=2)

        tk.Button(root, text="🔹 Bắt đầu", command=self.set_start).grid(row=3, column=0)
        tk.Button(root, text="🔸 Kết thúc", command=self.set_end).grid(row=3, column=1)
        tk.Button(root, text="✂️ Cắt", command=self.cut_video, bg="green", fg="white").grid(row=3, column=2)

        tk.Button(root, text="⏭ Video tiếp theo", command=self.next_video).grid(row=4, column=1)

        self.status = tk.Label(root, text="")
        self.status.grid(row=5, column=0, columnspan=3)

        self.update_thread = threading.Thread(target=self.update_frame)
        self.update_thread.daemon = True
        self.update_thread.start()

    def select_source_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.source_folder = folder
            self.cut_log_file = os.path.join(self.source_folder, "a_cut_log.json")
            if os.path.exists(self.cut_log_file):
                with open(self.cut_log_file, "r") as f:
                    self.cut_log = json.load(f)
            else:
                self.cut_log = {}
            self.video_list = [f for f in os.listdir(self.source_folder) if f.endswith(".mp4") and f not in self.cut_log]
            if not self.video_list:
                messagebox.showinfo("Thông báo", "✅ Toàn bộ video đã được cắt.")
                self.status.config(text="✅ Danh sách trống - tất cả đã được cắt.")
                return
            self.current_video_index = 0
            self.load_video()

    def select_save_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.save_folder = folder

    def load_video(self):
        if not self.video_list:
            self.status.config(text="✅ Không còn video để cắt.")
            return

        filename = self.video_list[self.current_video_index]
        self.current_video_path = os.path.join(self.source_folder, filename)

        self.cap = cv2.VideoCapture(self.current_video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.scale.config(to=self.total_frames - 1)
        self.playing = True
        self.start_frame = None
        self.end_frame = None
        self.status.config(text=f"🎞 Đang xử lý: {filename}")

    def update_frame(self):
        while True:
            if self.cap and self.playing:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (640, 360))
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    photo = ImageTk.PhotoImage(image=image)
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                    self.canvas.image = photo
                self.scale.set(self.current_frame)
            self.root.update_idletasks()
            self.root.after(10)

    def on_scale(self, val):
        self.current_frame = int(val)

    def set_start(self):
        self.start_frame = self.current_frame
        self.status.config(text=f"✅ Bắt đầu tại frame {self.start_frame}")

    def set_end(self):
        self.end_frame = self.current_frame
        self.status.config(text=f"✅ Kết thúc tại frame {self.end_frame}")

    def cut_video(self):
        if not self.current_video_path or not self.save_folder:
            messagebox.showerror("Thiếu thông tin", "Hãy chọn thư mục cần cắt và thư mục lưu.")
            return
        if self.start_frame is None or self.end_frame is None:
            messagebox.showerror("Thiếu mốc", "Hãy chọn BẮT ĐẦU và KẾT THÚC.")
            return
        if self.end_frame <= self.start_frame:
            messagebox.showerror("Sai mốc", "KẾT THÚC phải sau BẮT ĐẦU.")
            return

        cap = cv2.VideoCapture(self.current_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        basename = os.path.basename(self.current_video_path)
        out_path = os.path.join(self.save_folder, f"cut_{basename}")
        out = cv2.VideoWriter(out_path, fourcc, self.fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self.start_frame <= frame_idx <= self.end_frame:
                out.write(frame)
            frame_idx += 1
            if frame_idx > self.end_frame:
                break

        cap.release()
        out.release()

        self.cut_log[basename] = True
        with open(self.cut_log_file, "w") as f:
            json.dump(self.cut_log, f, indent=2)

        self.status.config(text=f"✅ Đã cắt và lưu: {out_path}")

    def next_video(self):
        if self.current_video_index + 1 < len(self.video_list):
            self.current_video_index += 1
            self.load_video()
        else:
            messagebox.showinfo("Hoàn tất", "✅ Đã xử lý toàn bộ video.")
            self.status.config(text="✅ Không còn video để cắt.")

# Chạy giao diện
root = tk.Tk()
app = SmartVideoCutter(root)
root.mainloop()
