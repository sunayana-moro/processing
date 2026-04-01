import os
import tarfile
from pathlib import Path

# -------- SETTINGS --------
INPUT_ROOT = "/workspace/Headshot_5frames"
OUTPUT_ROOT = "/workspace/shards"
SAMPLES_PER_SHARD = 1000   # adjust (500–5000 is good)
INCLUDE_REFERENCE = True   # include reference_frame.png or not

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# -------- HELPERS --------
def add_file(tar, file_path, arcname):
    tar.add(file_path, arcname=arcname)

# -------- MAIN --------
sample_idx = 0
shard_idx = 0

def get_tar_path(idx):
    return os.path.join(OUTPUT_ROOT, f"shard_{idx:05d}.tar")

tar = tarfile.open(get_tar_path(shard_idx), "w")

for set_name in sorted(os.listdir(INPUT_ROOT)):
    set_path = os.path.join(INPUT_ROOT, set_name)
    if not os.path.isdir(set_path):
        continue

    # collect frames + audio
    for i in range(1, 6):  # assuming 5 frames/audio
        frame_path = os.path.join(set_path, f"frame_{i}.png")
        audio_path = os.path.join(set_path, f"audio_{i}.wav")

        if not (os.path.exists(frame_path) and os.path.exists(audio_path)):
            continue

        key = f"{sample_idx:08d}"

        # add image + audio
        add_file(tar, frame_path, f"{key}.png")
        add_file(tar, audio_path, f"{key}.wav")

        # optionally add reference frame
        if INCLUDE_REFERENCE:
            ref_path = os.path.join(set_path, "reference_frame.png")
            if os.path.exists(ref_path):
                add_file(tar, ref_path, f"{key}.ref.png")

        sample_idx += 1

        # shard rollover
        if sample_idx % SAMPLES_PER_SHARD == 0:
            tar.close()
            shard_idx += 1
            tar = tarfile.open(get_tar_path(shard_idx), "w")

# close last shard
tar.close()

print(f"✅ Done! Created {shard_idx + 1} shards with {sample_idx} samples.")