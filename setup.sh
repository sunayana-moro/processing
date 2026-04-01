#!/bin/bash

set -e

echo "🚀 Setting up Conda environment..."

ENV_NAME="lipsync"
PYTHON_VERSION="3.10"

# ------------------------------
# Check conda
# ------------------------------
if ! command -v conda &> /dev/null
then
    echo "❌ Conda not found. Install Miniconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# ------------------------------
# Init conda (important for scripts)
# ------------------------------
eval "$(conda shell.bash hook)"

# ------------------------------
# Create environment (if not exists)
# ------------------------------
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "📦 Environment '$ENV_NAME' already exists"
else
    echo "📦 Creating environment..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi

# ------------------------------
# Activate environment
# ------------------------------
echo "🔄 Activating environment..."
conda activate $ENV_NAME

# ------------------------------
# Install system deps (ffmpeg)
# ------------------------------
echo "🎬 Installing ffmpeg..."
sudo apt update
sudo apt install -y ffmpeg

# ------------------------------
# Upgrade pip
# ------------------------------
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ------------------------------
# Install Python dependencies
# ------------------------------
echo "📚 Installing Python packages..."

pip install \
    opencv-python==4.9.0.80 \
    mediapipe==0.10.14 \
    numpy==1.26.4 \
    librosa==0.10.2.post1 \
    soundfile==0.12.1 \
    tqdm==4.66.4

# ------------------------------
# Verify installation
# ------------------------------
echo "🔍 Verifying setup..."

python - <<EOF
import cv2, mediapipe, librosa, numpy
print("✅ All core libraries imported successfully!")
EOF

# ------------------------------
# Done
# ------------------------------
echo ""
echo "🎉 Setup complete!"
echo ""
echo "👉 Activate with:"
echo "conda activate $ENV_NAME"