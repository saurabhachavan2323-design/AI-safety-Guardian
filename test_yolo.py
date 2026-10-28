
from ultralytics import YOLO

def main():
    print("Loading YOLO model...")

    # Load a small pretrained object-detection model.
    model = YOLO("yolo11n.pt")

    print("Model loaded successfully.")
    print("Testing object detection...")

    # Run detection on a sample image from the internet.
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        save=True,
        conf=0.4
    )

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]

            print(
                f"Detected: {class_name} "
                f"| Confidence: {confidence:.2f}"
            )

    print("Detection test finished.")
    print("Check the runs/detect folder for the output image.")

if __name__ == "__main__":
    main()
