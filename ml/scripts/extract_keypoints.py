import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# Paths
RAW_VIDEOS_DIR = Path("ml/data/raw_videos")
KEYPOINTS_DIR = Path("ml/data/keypoints")
MODEL_PATH = Path("ml/models/holistic_landmarker.task")

# MediaPipe Tasks setup
BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def extract_keypoints_from_video(video_path: Path) -> np.ndarray:
    """HolisticLandmarker."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    timestamp_ms = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:  # handle NaN
        fps = 30.0
    frame_interval_ms = int(1000 / fps)

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    with HolisticLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += frame_interval_ms

            # Left hand: 21 x 3 = 63
            if result.left_hand_landmarks:
                lh = np.array([[p.x, p.y, p.z] for p in result.left_hand_landmarks]).flatten()
            else:
                lh = np.zeros(63)

            # Right hand: 21 x 3 = 63
            if result.right_hand_landmarks:
                rh = np.array([[p.x, p.y, p.z] for p in result.right_hand_landmarks]).flatten()
            else:
                rh = np.zeros(63)

            # Pose: 33 x 3 = 99
            if result.pose_landmarks:
                pose = np.array([[p.x, p.y, p.z] for p in result.pose_landmarks]).flatten()
            else:
                pose = np.zeros(99)

            # Face: 15 selected landmarks x 3 = 45
            if result.face_landmarks:
                face_indices = [61, 291, 13, 14, 17, 0, 33, 133, 362, 263, 70, 300, 105, 334, 1]
                face = np.array([[result.face_landmarks[i].x,
                                  result.face_landmarks[i].y,
                                  result.face_landmarks[i].z] for i in face_indices]).flatten()
            else:
                face = np.zeros(45)

            frame_keypoints = np.concatenate([lh, rh, pose, face])
            frames.append(frame_keypoints)

    cap.release()
    return np.array(frames)


def process_all_videos():
    """raw_videos"""
    KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Download it first.")
        return

    video_files = list(RAW_VIDEOS_DIR.glob("*.mp4")) + list(RAW_VIDEOS_DIR.glob("*.mov"))
    if not video_files:
        print(f"No videos found in {RAW_VIDEOS_DIR}. Add some and try again.")
        return

    print(f"Found {len(video_files)} videos.")

    for i, video_path in enumerate(video_files, 1):
        keypoints = extract_keypoints_from_video(video_path)
        if len(keypoints) == 0:
            print(f"[{i}/{len(video_files)}] SKIPPED {video_path.name} (no frames read)")
            continue

        output_path = KEYPOINTS_DIR / f"{video_path.stem}.npy"
        np.save(output_path, keypoints)
        print(f"[{i}/{len(video_files)}] {video_path.name} -> {output_path.name} shape={keypoints.shape}")


if __name__ == "__main__":
    process_all_videos()