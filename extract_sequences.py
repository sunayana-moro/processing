import os

# Reduce noisy native logs from MediaPipe / TFLite before importing them.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import librosa
import mediapipe as mp
import numpy as np
import soundfile as sf

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

mp_face = mp.solutions.face_mesh


# -------- FRAME QUALITY CHECK --------
def is_good_frame(frame, face_mesh):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return False

    lm = result.multi_face_landmarks[0].landmark

    left = np.array([lm[61].x, lm[61].y])
    right = np.array([lm[291].x, lm[291].y])
    mouth_width = np.linalg.norm(left - right)

    return mouth_width > 0.02


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


# -------- AUDIO WINDOW --------
def get_audio_window(wav, sr, frame_idx):
    samples_per_frame = sr // FPS
    center = frame_idx * samples_per_frame

    window_size = int(AUDIO_WINDOW_SEC * sr)

    start = max(0, center - window_size // 2)
    end = min(len(wav), start + window_size)

    return wav[start:end]


def pick_reference_frame(good_flags, seq_start, seq_end):
    sequence_center = (seq_start + seq_end - 1) / 2.0
    outside_good_indices = [
        idx for idx, is_good in enumerate(good_flags)
        if is_good and not (seq_start <= idx < seq_end)
    ]

    if outside_good_indices:
        return max(outside_good_indices, key=lambda idx: abs(idx - sequence_center))

    sequence_good_indices = [
        idx for idx in range(seq_start, seq_end)
        if good_flags[idx]
    ]

    if sequence_good_indices:
        return max(sequence_good_indices, key=lambda idx: abs(idx - sequence_center))

    return None


def save_sequence_set(frames, wav, sr, seq_start, seq_end, set_index, good_flags):
    set_dir = os.path.join(OUTPUT_ROOT, f"set{set_index}")
    os.makedirs(set_dir, exist_ok=True)

    for frame_offset, global_idx in enumerate(range(seq_start, seq_end), start=1):
        frame = frames[global_idx]
        frame_path = os.path.join(set_dir, f"frame_{frame_offset}.png")
        audio_path = os.path.join(set_dir, f"audio_{frame_offset}.wav")

        cv2.imwrite(frame_path, frame)
        sf.write(audio_path, get_audio_window(wav, sr, global_idx), sr)

    reference_idx = pick_reference_frame(good_flags, seq_start, seq_end)
    if reference_idx is not None:
        reference_path = os.path.join(set_dir, "reference_frame.png")
        cv2.imwrite(reference_path, frames[reference_idx])


# -------- PROCESS VIDEO --------
def process_video(video_path, start_set_index):
    video_set_count = 0
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps != FPS:
        cap.release()
        print(f"Skipping (fps mismatch): {video_path}")
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

    wav, sr = librosa.load(video_path, sr=None)
    total_frames = len(frames)
    seconds = total_frames // FPS
    next_set_index = start_set_index

    with mp_face.FaceMesh(static_image_mode=False) as face_mesh:
        good_flags = [is_good_frame(frame, face_mesh) for frame in frames]

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
                frames,
                wav,
                sr,
                seq_start,
                seq_end,
                next_set_index,
                good_flags,
            )
            next_set_index += 1
            video_set_count += 1

    return next_set_index


def iter_video_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in sorted(files):
            if file.lower().endswith(".mp4"):
                yield os.path.join(root, file)


# -------- DATASET LOOP --------
def process_dataset():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    next_set_index = 1
    for video_path in iter_video_files(INPUT_ROOT):
        print(f"Processing: {video_path}")
        next_set_index = process_video(video_path, next_set_index)

    print(f"Finished. Created {next_set_index - 1} sets in {OUTPUT_ROOT}")


if __name__ == "__main__":
    process_dataset()
