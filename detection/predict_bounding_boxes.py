from ultralytics import YOLO

# Load your trained model
model = YOLO("./runs/detect/train3/weights/best.pt")

# Run detection
results = model.predict(
    source="handwritten-yolo/test",  # folder, image, video, or webcam index
    conf=0.8,                        # confidence threshold
    save=True,                       # save output images with boxes
    save_crop=True                   # save cropped detections
)

# Loop through results if you want to inspect
for r in results:
    print(f"Detected {len(r.boxes)} objects in {r.path}")
    for box in results[0].boxes:
        print(box.xyxy, box.conf, box.cls)

