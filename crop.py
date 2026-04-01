import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager

# Reduce noisy native logs from MediaPipe / TFLite before importing them.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp

# -------- PATH SETUP (ROBUST) --------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
INPUT_ROOT = os.path.join(PROJECT_ROOT, "mp4")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "cropped_faces")

# -------- SETTINGS --------
OUTPUT_SIZE = 256
SMOOTHING = 0.8   # higher = smoother box
FFMPEG_LOGLEVEL = "error"

mp_face = mp.solutions.face_mesh


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


def get_face_bbox(landmarks, h, w):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    return x_min, y_min, x_max, y_max


def make_square_crop(x_min, y_min, x_max, y_max, h, w, scale=1.4):
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    size = int(max(x_max - x_min, y_max - y_min) * scale)

    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w, cx + size // 2)
    y2 = min(h, cy + size // 2)

    return x1, y1, x2, y2


def iter_mp4_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in sorted(files):
            if file.lower().endswith(".mp4"):
                yield os.path.join(root, file)


def mux_audio(video_only_path, source_path, output_path):
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        FFMPEG_LOGLEVEL,
        "-i",
        video_only_path,
        "-i",
        source_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_video(video_path, out_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = os.path.dirname(out_path)
    fd, temp_path = tempfile.mkstemp(suffix=".mp4", dir=out_dir)
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (OUTPUT_SIZE, OUTPUT_SIZE))

    prev_box = None

    try:
        with suppress_stderr():
            with mp_face.FaceMesh(static_image_mode=False) as face_mesh:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = face_mesh.process(rgb)

                    if result.multi_face_landmarks:
                        lm = result.multi_face_landmarks[0].landmark
                        x_min, y_min, x_max, y_max = get_face_bbox(lm, h, w)
                        box = make_square_crop(x_min, y_min, x_max, y_max, h, w)

                        if prev_box is not None:
                            box = [
                                int(SMOOTHING * p + (1 - SMOOTHING) * c)
                                for p, c in zip(prev_box, box)
                            ]

                        prev_box = box

                    if prev_box is None:
                        continue

                    x1, y1, x2, y2 = prev_box
                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    crop = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE))
                    out.write(crop)
    finally:
        cap.release()
        out.release()

    try:
        mux_audio(temp_path, video_path, out_path)
    except subprocess.CalledProcessError:
        shutil.move(temp_path, out_path)
    else:
        os.remove(temp_path)


def process_dataset():
    video_files = list(iter_mp4_files(INPUT_ROOT))
    total_files = len(video_files)

    if total_files == 0:
        print("No MP4 files found to crop.")
        return

    for index, in_path in enumerate(video_files, start=1):
        rel = os.path.relpath(os.path.dirname(in_path), INPUT_ROOT)
        out_dir = os.path.join(OUTPUT_ROOT, rel)
        os.makedirs(out_dir, exist_ok=True)

        file = os.path.basename(in_path)
        out_path = os.path.join(out_dir, file)

        process_video(in_path, out_path)

        percent = (index / total_files) * 100
        print(
            f"\r[{index}/{total_files}] {percent:6.2f}% - {file}",
            end="",
            flush=True,
        )

    print()
    print(f"Completed cropping {total_files} files.")


if __name__ == "__main__":
    process_dataset()
