import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def main(video_path=None):
    project_root = Path(__file__).resolve().parents[2]
    if video_path is None:
        video_path = project_root / "samples" / "squats.mp4"
    else:
        video_path = Path(video_path)
        
    output_dir = project_root / "runs" / "pose"
    csv_path = output_dir / "squats_keypoints.csv"
    overlay_path = output_dir / "squats_pose.mp4"
    model_path = project_root / "yolo11n-pose.pt"

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise ValueError("Video has an invalid frame rate.")

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video has invalid frame dimensions.")

    columns = [
        "frame_index",
        "timestamp_ms",
        "image_width",
        "image_height",
        "person_detected",
    ]
    for name in KEYPOINT_NAMES:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_confidence"])

    video_writer = None
    processed_frames = 0
    detected_frames = 0

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()

            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_index = processed_frames
                timestamp_ms = round(frame_index * 1000.0 / fps, 2)

                results = model.predict(
                    source=frame,
                    imgsz=640,
                    conf=0.25,
                    verbose=False,
                )
                result = results[0]

                row = {
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "image_width": width,
                    "image_height": height,
                    "person_detected": False,
                }

                annotated_frame = frame

                if result.keypoints is not None and len(result.keypoints) > 0:
                    # Select the detected person with the highest box confidence.
                    if result.boxes is not None and result.boxes.conf is not None:
                        best_index = int(result.boxes.conf.argmax().item())
                    else:
                        best_index = 0

                    xy = result.keypoints.xy[best_index].cpu().numpy()
                    confidence_values = None

                    if result.keypoints.conf is not None:
                        confidence_values = (
                            result.keypoints.conf[best_index].cpu().numpy()
                        )

                    row["person_detected"] = True
                    detected_frames += 1

                    for index, name in enumerate(KEYPOINT_NAMES):
                        x, y = xy[index]
                        confidence = (
                            confidence_values[index]
                            if confidence_values is not None
                            else None
                        )

                        row[f"{name}_x"] = round(float(x), 2)
                        row[f"{name}_y"] = round(float(y), 2)
                        row[f"{name}_confidence"] = (
                            round(float(confidence), 4)
                            if confidence is not None
                            else ""
                        )

                    annotated_frame = result.plot()

                writer.writerow(row)

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(overlay_path), fourcc, fps, (width, height)
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(
                            f"Could not create overlay video: {overlay_path}"
                        )

                video_writer.write(annotated_frame)
                processed_frames += 1

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()


    print("Keypoint extraction completed.")
    print(f"Frames processed: {processed_frames}")
    print(f"Frames with a detected person: {detected_frames}")
    print(f"Expected frames from video metadata: {expected_frames}")
    print(f"CSV output: {csv_path}")
    print(f"Overlay video: {overlay_path}")


if __name__ == "__main__":
    main()
