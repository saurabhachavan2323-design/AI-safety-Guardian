
import cv2
import time

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Webcam could not be opened.")
else:
    print("Webcam opened. Press Q in the camera window to exit.")

    for attempt in range(20):
        success, frame = camera.read()

        if success and frame is not None:
            print(
                f"Frame {attempt + 1}: "
                f"shape={frame.shape}, "
                f"brightness={frame.mean():.2f}"
            )
            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
        else:
            print(f"Frame {attempt + 1}: FAILED")

        time.sleep(0.1)

camera.release()
cv2.destroyAllWindows()

