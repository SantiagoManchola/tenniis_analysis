from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.predict('input_videos/input_video.mp4', save=True)