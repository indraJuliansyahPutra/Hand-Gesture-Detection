import os
import tkinter as tk
from tkinter import filedialog

from ultralytics import YOLO
import cv2

model_path = os.path.join('.', 'models', 'best.pt')
model = YOLO(model_path)

# Skor Akurasi
threshold = 0.5
class_name_dict = {0: "Hello", 1: "I Love You", 2: "No", 3: "Please", 4: "Thanks", 5: "Yes"}

class VideoStreamWidget:
    def __init__(self, cap, window):
        self.window = window
        self.cap = cap

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.btn_start = tk.Button(window, text="Start Detection", command=self.start, bg="green", width=15, height=3)
        self.btn_start.pack(side="left", padx=60, pady=10)

        self.btn_stop = tk.Button(window, text="Stop Detection", command=self.stop, bg="red", width=15, height=3)
        self.btn_stop.pack(side="right", padx=60, pady=10)

        self.btn_select = tk.Button(window, text="Select Video", command=self.select_video, bg="blue", fg="white", width=15, height=3)
        self.btn_select.pack(side="bottom", pady=20)
        
        self.btn_detect_image = tk.Button(window, text="Select Image", command=self.detect_image, bg="purple", fg="white", width=15, height=3)
        self.btn_detect_image.pack(side="bottom", pady=20)

    def select_video(self):
        filename = tk.filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if filename:
            self.cap = cv2.VideoCapture(filename)
            self.start()

    
    def start(self):
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            cv2.destroyAllWindows()
            return

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score < threshold:
                color = (0, 0, 255)  # warna merah
            elif score < 0.75:
                color = (255, 0, 0)  # warna biru
            else:
                color = (0, 255, 0)  # warna hijau

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            cv2.putText(frame, f"{class_name_dict[int(class_id)].upper()} {score:.2f}", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)



        img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        photo = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())
        self.canvas.create_image(0, 0, image=photo, anchor="nw")
        self.canvas.image = photo

        self.window.after(1, self.update)
  

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.window.destroy()

    def detect_image(self):
        filename = tk.filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if filename:
            img = cv2.imread(filename)
            results = model(img)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                if score < threshold:
                    color = (0, 0, 255)  # warna merah
                elif score < 0.75:
                    color = (255, 0, 0)  # warna biru
                else:
                    color = (0, 255, 0)  # warna hijau

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                cv2.putText(img, f"{class_name_dict[int(class_id)].upper()} {score:.2f}", (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

            img = cv2.cvtColor(img, cv2.IMREAD_COLOR)
            photo = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())
            self.canvas.create_image(0, 0, image=photo, anchor="nw")
            self.canvas.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    VideoStreamWidget(cv2.VideoCapture(0), root)
    root.mainloop()
