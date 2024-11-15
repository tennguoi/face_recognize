import cv2
import os
import json
import torch
import numpy as np
from retinaface import RetinaFace

def create_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_face_id(directory: str) -> int:
    user_ids = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            number = int(os.path.split(filename)[-1].split("-")[1])
            user_ids.append(number)
    user_ids = sorted(list(set(user_ids)))
    max_user_ids = 1 if len(user_ids) == 0 else max(user_ids) + 1
    for i in sorted(range(0, max_user_ids)):
        try:
            if user_ids.index(i):
                face_id = i
        except ValueError:
            return i
    return max_user_ids

def save_name(face_id: int, face_name: str, filename: str) -> None:
    names_json = {}
    if os.path.exists(filename):
        with open(filename, 'r') as fs:
            names_json = json.load(fs)
    names_json[str(face_id)] = face_name
    with open(filename, 'w') as fs:
        json.dump(names_json, ensure_ascii=False, indent=4, fp=fs)

def main():
    directory = 'images'
    names_json_filename = 'names.json'

    create_directory(directory)
    
    # Initialize RetinaFace
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize camera
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    
    face_name = input('\nEnter user name and press <return> -->  ')
    face_id = get_face_id(directory)
    save_name(face_id, face_name, names_json_filename)
    print('\n[INFO] Initializing face capture. Look at the camera and wait...')
    
    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        
        # Detect faces using RetinaFace
        faces = RetinaFace.detect_faces(frame)
        if len(faces) == 0:
            continue
        
        # Lọc khuôn mặt lớn nhất (gần nhất)
        largest_face = None
        largest_area = 0
        for key in faces.keys():
            face = faces[key]
            facial_area = face['facial_area']
            x1, y1, x2, y2 = map(int, facial_area)
            area = (x2 - x1) * (y2 - y1)
            
            # Kiểm tra nếu đây là khuôn mặt lớn nhất
            if area > largest_area:
                largest_area = area
                largest_face = (x1, y1, x2, y2)

        # Nếu tìm thấy khuôn mặt lớn nhất, chụp ảnh
        if largest_face:
            x1, y1, x2, y2 = largest_face
            face_img = frame[y1:y2, x1:x2]

            # Chỉ chụp nếu khuôn mặt lớn hơn 20% kích thước khung hình
            frame_area = frame.shape[0] * frame.shape[1]
            if largest_area / frame_area > 0.2:
                save_path = f'{directory}/Users-{face_id}-{count}.jpg'
                cv2.imwrite(save_path, cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY))
                count += 1
                
                # Vẽ khung xanh quanh khuôn mặt
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Face Capture', frame)
        
        # Break conditions
        k = cv2.waitKey(100) & 0xff
        if k == 27:  # ESC key
            break
        elif count >= 30:  # Collected enough samples
            break

    print('\n[INFO] Face capture completed successfully!')
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
