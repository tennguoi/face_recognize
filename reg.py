import torch
from PIL import Image
import cv2
import numpy as np
import json
import os
from tkinter import Tk, Button, Label, StringVar
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceRecognitionSystem:
    def __init__(self):
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.yolo_model.classes = [0]
        self.mtcnn = MTCNN(margin=20, keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.known_embeddings = {}
        self.names = {}
        self.load_known_faces()

    def load_known_faces(self):
        if os.path.exists('face_embeddings.npz'):
            data = np.load('face_embeddings.npz', allow_pickle=True)
            self.known_embeddings = dict(data['embeddings'][()])
        if os.path.exists('names.json'):
            with open('names.json', 'r') as f:
                self.names = json.load(f)

    def get_face_embedding(self, face_img):
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        face_tensor = transform(Image.fromarray(face_img)).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            embedding = self.facenet(face_tensor)
        return embedding.cpu().numpy()

    def recognize_face(self, face_embedding, threshold=0.7):
        if not self.known_embeddings:
            return "Unknown", 0
        min_dist = float('inf')
        identity = "Unknown"
        for name, embedding in self.known_embeddings.items():
            dist = np.linalg.norm(face_embedding - embedding)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                identity = self.names.get(str(name), "Unknown")
        confidence = 1 - min_dist if min_dist < threshold else 0
        return identity, confidence * 100

    def detect_and_recognize(self, frame):
        results = self.yolo_model(frame)
        detected_name = "Unknown"
        confidence = 0
        for det in results.xyxy[0]:
            if det[-1] == 0:
                x1, y1, x2, y2 = map(int, det[:4])
                person_img = frame[y1:y2, x1:x2]
                faces = self.mtcnn(person_img)
                if faces is None:
                    continue
                boxes, _ = self.mtcnn.detect(person_img)
                if boxes is None:
                    continue
                for box in boxes:
                    bbox = list(map(int, box.tolist()))
                    face_img = person_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    embedding = self.get_face_embedding(face_img)
                    detected_name, confidence = self.recognize_face(embedding[0])
                    cv2.rectangle(frame, (x1 + bbox[0], y1 + bbox[1]), 
                                  (x1 + bbox[2], y1 + bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{detected_name} ({confidence:.1f}%)", 
                                (x1 + bbox[0], y1 + bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame, detected_name, confidence

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.face_system = FaceRecognitionSystem()
        
        # Biến hiển thị tên người nhận diện
        self.detected_name_var = StringVar()
        self.detected_name_var.set("Detected: Unknown")
        self.label_name = Label(window, textvariable=self.detected_name_var, font=("Arial", 16))
        self.label_name.pack()

        # Nút thoát
        self.btn_quit = Button(window, text="Thoát", width=15, command=self.window.quit)
        self.btn_quit.pack()
        
        self.update()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame, detected_name, confidence = self.face_system.detect_and_recognize(frame)
            # Cập nhật tên người nhận diện lên GUI
            self.detected_name_var.set(f"Detected: {detected_name} ({confidence:.1f}%)")
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                self.window.quit()
        self.window.after(10, self.update)

if __name__ == "__main__":
    root = Tk()
    App(root, "Hệ thống nhận diện khuôn mặt CMC")
    root.mainloop()
