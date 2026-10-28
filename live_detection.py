
import cv2
from ultralytics import YOLO


def main():
    print("Loading YOLO model...")
    model = YOLO("yolo11n.pt")
    print("Model loaded. Opening webcam...")

    # Open the default webcam on Windows.
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Try the default OpenCV backend if DirectShow fails.
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Cannot open webcam.")
        print("Check Windows camera permissions and close other camera apps.")
        return

    print("Live detection started. Press Q to stop.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read a camera frame.")
                break

            # Run YOLO detection on the current webcam frame.
            results = model.predict(
                source=frame,
                conf=0.4,
                imgsz=640,
                verbose=False
            )

            # Draw detection boxes and labels on the frame.
            annotated_frame = results[0].plot()

            cv2.imshow(
                "AI Safety Guardian - Live Detection",
                annotated_frame
            )

            # Press Q in the video window to exit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released. Live detection stopped.")


if __name__ == "__main__":
    main()
