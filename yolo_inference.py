from ultralytics import YOLO

model = YOLO("yolo11x.pt")

model.predict('input_videos/input_video.mp4', save=True)