import os
import cv2
import numpy as np
import soundfile as sf
import subprocess
from multiprocessing import Pool, cpu_count
from ultralytics import YOLO

# -------- PATH SETUP --------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
INPUT_ROOT = os.path.join(PROJECT_ROOT, "cropped_faces")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "final_dataset")

# -------- SETTINGS --------
FPS = 25
SEQ_LEN = 5
SETS_PER_SECOND = 1
AUDIO_WINDOW_SEC = 0.2
BATCH_SIZE = 32

# -------- LOAD MODEL (GPU) --------
face_model = YOLO("yolov8n-face.pt")  # auto GPU


# -------- FAST AUDIO --------
def load_audio_fast(video_path):
    cmd = [
        "ffmpeg", "-i", video_path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", "16000",
        "-"
    ]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio = np.frombuffer(out.stdout, np.float32)
    return audio, 16000


# -------- AUDIO WINDOW --------
def get_audio_window(wav, sr, frame_idx):
    samples_per_frame = sr // FPS
    center = frame_idx * samples_per_frame

    window_size = int(AUDIO_WINDOW_SEC * sr)

    start = max(0, center - window_size // 2)
    end = min(len(wav), start + window_size)

    return wav[start:end]


# -------- FIND CONTINUOUS SEQUENCES --------
def find_sequences(flags, seq_len=5):
    sequences = []
    i = 0

    while i <= len(flags) - seq_len:
        if all(flags[i:i + seq_len]):
            sequences.append((i, i + seq_len))
            i += seq_len
        else:
            i += 1

    return sequences


# -------- PICK REFERENCE FRAME --------
def pick_reference_frame(good_flags, seq_start, seq_end):
    sequence_center = (seq_start + seq_end - 1) / 2.0

    outside = [
        i for i, g in enumerate(good_flags)
        if g and not (seq_start <= i < seq_end)
    ]

    if outside:
        return max(outside, key=lambda i: abs(i - sequence_center))

    inside = [i for i in range(seq_start, seq_end) if good_flags[i]]
    if inside:
        return max(inside, key=lambda i: abs(i - sequence_center))

    return None


# -------- SAVE SET --------
def save_sequence_set(frames, wav, sr, seq_start, seq_end, set_index, good_flags):
    set_dir = os.path.join(OUTPUT_ROOT, f"set{set_index}")
    os.makedirs(set_dir, exist_ok=True)

    for frame_offset, global_idx in enumerate(range(seq_start, seq_end), start=1):
        frame = frames[global_idx]

        cv2.imwrite(os.path.join(set_dir, f"frame_{frame_offset}.png"), frame)
        sf.write(
            os.path.join(set_dir, f"audio_{frame_offset}.wav"),
            get_audio_window(wav, sr, global_idx),
            sr
        )

    ref_idx = pick_reference_frame(good_flags, seq_start, seq_end)
    if ref_idx is not None:
        cv2.imwrite(os.path.join(set_dir, "reference_frame.png"), frames[ref_idx])


# -------- GPU FACE FILTER --------
def compute_good_flags(frames):
    good_flags = []

    for i in range(0, len(frames), BATCH_SIZE):
        batch = frames[i:i+BATCH_SIZE]
        results = face_model(batch, verbose=False)

        for frame, r in zip(batch, results):

            # reject multiple / no faces
            if len(r.boxes) != 1:
                good_flags.append(False)
                continue

            box = r.boxes.xyxy[0]
            conf = r.boxes.conf[0]

            face_w = box[2] - box[0]
            face_h = box[3] - box[1]

            h_frame, w_frame, _ = frame.shape

            good = (
                conf > 0.6 and
                face_w / w_frame > 0.1 and
                face_h / h_frame > 0.1 and
                0.75 < (face_w / face_h) < 1.5
            )

            good_flags.append(bool(good))

    return good_flags


# -------- PROCESS VIDEO --------
def process_video(video_path, start_set_index):
    video_set_count = 0

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps != FPS:
        cap.release()
        return start_set_index

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        return start_set_index

    wav, sr = load_audio_fast(video_path)

    good_flags = compute_good_flags(frames)

    total_frames = len(frames)
    seconds = total_frames // FPS
    next_set_index = start_set_index

    for sec in range(seconds):
        if video_set_count >= 5:
            break

        start = sec * FPS
        end = start + FPS

        sequences = find_sequences(good_flags[start:end], SEQ_LEN)
        if len(sequences) < SETS_PER_SECOND:
            continue

        for local_start, local_end in sequences[:SETS_PER_SECOND]:
            if video_set_count >= 5:
                break

            seq_start = start + local_start
            seq_end = start + local_end

            save_sequence_set(
                frames, wav, sr,
                seq_start, seq_end,
                next_set_index, good_flags
            )

            next_set_index += 1
            video_set_count += 1

    return next_set_index


# -------- ITER VIDEOS --------
def iter_video_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                yield os.path.join(root, f)


# -------- MULTIPROCESS DATASET --------
def process_dataset():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    video_files = list(iter_video_files(INPUT_ROOT))

    num_workers = min(16, cpu_count())
    print(f"Using {num_workers} workers")

    args = [(vp, i * 10000) for i, vp in enumerate(video_files)]

    with Pool(num_workers) as pool:
        pool.starmap(process_video, args)


# -------- MAIN --------
if __name__ == "__main__":
    process_dataset()