import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
file_path = r"C:\adu7\data\user\0.jpg"  # Avoids escaping backslashes
results = model(file_path)
results.show()
