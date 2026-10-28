
from flask import Flask, render_template, Response, jsonify
from pathlib import Path
from ultralytics import YOLO
import cv2
import threading
import time

app = Flask(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"
RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

# Load YOLO model once
model = YOLO(str(MODEL_PATH))

# Shared camera and recording resources
camera = None
camera_lock = threading.Lock()
recording_lock = threading.Lock()

recording = False
video_writer = None
camera_error = None


def get_camera():
    """Open and return the webcam."""
    global camera, camera_error

    with camera_lock:
        if camera is None or not camera.isOpened():
            if camera is not None:
                camera.release()

            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(0)

        if camera.isOpened():
            camera_error = None
            return camera

        camera.release()
        camera = None
        camera_error = "Webcam unavailable"
        return None


def release_camera():
    """Release the webcam on application shutdown."""
    global camera

    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None


def generate_frames():
    """Capture webcam frames, run YOLO and stream video."""
    while True:
        cam = get_camera()

        if cam is None:
            time.sleep(1)
            continue

        with camera_lock:
            success, frame = cam.read()

        if not success or frame is None or frame.size == 0:
            print("WARNING: Camera frame could not be read")
            time.sleep(0.1)
            continue

        try:
            results = model.predict(
                source=frame,
                conf=0.4,
                imgsz=640,
                verbose=False
            )
            annotated_frame = results[0].plot()

        except Exception as error:
            print("Detection error:", error)
            annotated_frame = frame

        # Record annotated frames
        with recording_lock:
            if recording and video_writer is not None:
                video_writer.write(annotated_frame)

        success, buffer = cv2.imencode(".jpg", annotated_frame)

        if not success:
            print("WARNING: Could not encode camera frame")
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


# IMPORTANT: Dashboard home route
@app.route("/")
def dashboard():
    return render_template("index.html")


# System health endpoint
@app.route("/api/health")
def health():
    cam = get_camera()

    with recording_lock:
        is_recording = recording

    return jsonify({
        "status": "online",
        "project": "AI Safety Guardian",
        "version": "0.3.1",
        "camera": "available" if cam is not None else "unavailable",
        "model": "YOLO loaded",
        "recording": is_recording
    })


# Live video endpoint
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# Start recording
@app.route("/api/record/start", methods=["POST"])
def start_recording():
    global recording, video_writer

    cam = get_camera()

    if cam is None:
        return jsonify({"error": "Webcam unavailable"}), 503

    with camera_lock:
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with recording_lock:
        if recording:
            return jsonify({"status": "already_recording"})

        if width <= 0 or height <= 0:
            return jsonify({
                "error": "Could not read webcam dimensions"
            }), 500

        filename = time.strftime(
            "safety_monitoring_%Y%m%d_%H%M%S.avi"
        )
        output_path = RECORDINGS_DIR / filename

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"XVID"),
            20.0,
            (width, height)
        )

        if not writer.isOpened():
            writer.release()
            return jsonify({
                "error": "Could not create recording file"
            }), 500

        video_writer = writer
        recording = True

    print("Recording started:", output_path)

    return jsonify({
        "status": "recording",
        "filename": filename
    })


# Stop recording and finalize the AVI
@app.route("/api/record/stop", methods=["POST"])
def stop_recording():
    global recording, video_writer

    with recording_lock:
        if video_writer is not None:
            video_writer.release()
            video_writer = None

        recording = False

    print("Recording stopped")

    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    try:
        app.run(
            host="127.0.0.1",
            port=5000,
            debug=True,
            threaded=True,
            use_reloader=False
        )
    finally:
        with recording_lock:
            if video_writer is not None:
                video_writer.release()
                video_writer = None

            recording = False

        release_camera()
