import os
import shutil
from contextlib import contextmanager

# Reduce noisy native logs from MediaPipe / TFLite before importing them.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
import numpy as np

# -------- SETTINGS --------
DATASET_ROOT = "dataset"
OUTPUT_ROOT = "processed_dataset"
FRAME_SKIP = 2
OCCLUSION_THRESHOLD = 0.1
MISSING_COPY_LOG = "missing_copies.log"
DELETED_LOG = "deleted_files.log"

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands


@contextmanager
def suppress_stderr():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def is_occluded(video_path, frame_skip=2,
                face_threshold=0.1,
                hand_threshold=0.1):

    cap = cv2.VideoCapture(video_path)

    total = 0
    face_bad = 0
    hand_frames = 0

    with suppress_stderr():
        with mp_face.FaceMesh(static_image_mode=False) as face_mesh, \
             mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands:

            frame_id = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % frame_skip == 0:
                    total += 1

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # ---- FACE CHECK ----
                    face_result = face_mesh.process(rgb)

                    if not face_result.multi_face_landmarks:
                        face_bad += 1
                    else:
                        lm = face_result.multi_face_landmarks[0].landmark

                        left = np.array([lm[61].x, lm[61].y])
                        right = np.array([lm[291].x, lm[291].y])
                        mouth_width = np.linalg.norm(left - right)

                        if mouth_width < 0.02:
                            face_bad += 1

                    # ---- HAND CHECK ----
                    hand_result = hands.process(rgb)

                    if hand_result.multi_hand_landmarks:
                        hand_frames += 1

                frame_id += 1

    cap.release()

    if total == 0:
        return True

    face_ratio = face_bad / total
    hand_ratio = hand_frames / total

    # ---- FINAL DECISION ----
    if face_ratio > face_threshold:
        return True

    if hand_ratio > hand_threshold:
        return True

    return False


# -------- MAIN PROCESS --------
def process_dataset():
    video_root = os.path.join(DATASET_ROOT, "videos")
    mask_root = os.path.join(DATASET_ROOT, "masks")
    masked_root = os.path.join(DATASET_ROOT, "masked_videos")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    processed = 0
    deleted = 0
    kept = 0
    missing_copies = 0

    missing_log_path = os.path.join(OUTPUT_ROOT, MISSING_COPY_LOG)
    deleted_log_path = os.path.join(OUTPUT_ROOT, DELETED_LOG)
    with open(missing_log_path, "w", encoding="utf-8") as missing_log, \
         open(deleted_log_path, "w", encoding="utf-8") as deleted_log:
        missing_log.write("Files that could not be copied\n")
        missing_log.write("source_type\tsource_path\treason\n")
        deleted_log.write("Files deleted by the occlusion filter\n")
        deleted_log.write("video_path\treason\n")

        for vid_id in os.listdir(video_root):
            video_dir = os.path.join(video_root, vid_id)
            mask_dir = os.path.join(mask_root, vid_id)
            masked_dir = os.path.join(masked_root, vid_id)

            if not os.path.isdir(video_dir):
                continue

            for file in os.listdir(video_dir):
                video_path = os.path.join(video_dir, file)

                processed += 1

                # ---- OCCLUSION FILTER ----
                if is_occluded(video_path):
                    deleted += 1
                    deleted_log.write(f"{video_path}\tocclusion_detected\n")
                else:
                    kept += 1

                    out_dir = os.path.join(OUTPUT_ROOT, vid_id)
                    os.makedirs(out_dir, exist_ok=True)

                    name, ext = os.path.splitext(file)

                    mask_path = os.path.join(mask_dir, file)
                    masked_path = os.path.join(masked_dir, file)

                    # Copy original
                    shutil.copy(video_path,
                                os.path.join(out_dir, f"{name}{ext}"))

                    # Copy mask
                    if os.path.exists(mask_path):
                        shutil.copy(mask_path,
                                    os.path.join(out_dir, f"{name}_mask{ext}"))
                    else:
                        missing_copies += 1
                        missing_log.write(f"mask\t{mask_path}\tmissing source file\n")

                    # Copy masked video
                    if os.path.exists(masked_path):
                        shutil.copy(masked_path,
                                    os.path.join(out_dir, f"{name}_masked_video{ext}"))
                    else:
                        missing_copies += 1
                        missing_log.write(f"masked_video\t{masked_path}\tmissing source file\n")

                # ---- PROGRESS DISPLAY ----
                print(
                    f"\rProcessed: {processed} | Deleted: {deleted} | Kept: {kept} | Missing copies: {missing_copies}",
                    end=""
                )

    print()  # newline after loop
    print(f"Final → Processed: {processed}, Deleted: {deleted}, Kept: {kept}, Missing copies: {missing_copies}")
    print(f"Missing copy log: {missing_log_path}")
    print(f"Deleted file log: {deleted_log_path}")


if __name__ == "__main__":
    process_dataset()
