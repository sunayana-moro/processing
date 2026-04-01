# 📦 Dataset Setup Guide

This guide walks you through downloading datasets from Google Cloud Storage (GCS) and Hugging Face.

---

## 🚀 1. Install Google Cloud SDK

```
apt update && apt install -y curl apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
apt update && apt install -y google-cloud-sdk
gcloud auth login

```

---

## 🔐 2. Authenticate with Google Cloud

```
gcloud config set project prj-d-fi-base-3cyk
```

---

## 📥 3. Download Dataset from GCS (Parallelized)

```
gsutil -m \
  -o "GSUtil:parallel_process_count=8" \
  -o "GSUtil:parallel_thread_count=32" \
  cp -r gs://prj-d-fi-bkt-flam-ai-team-0/flam-avs/mp4 .
```

Notes:

* `-m` enables parallel transfers
* `parallel_process_count=8` → number of processes
* `parallel_thread_count=32` → threads per process
* Adjust based on CPU cores and bandwidth

---

## 🤗 4. Install Hugging Face CLI

```
pip install -U huggingface_hub
```

---

## 🔐 5. Authenticate with Hugging Face

```
huggingface-cli login
```

Paste your access token when prompted.

---

## 🧠 Tips

* For large datasets:

  * Run inside `tmux` or `screen`
  * Ensure sufficient disk space
  * Prefer SSD storage for faster I/O

* If using remote GPUs (e.g., Vast.ai):

  * Increase thread count if CPU allows
  * Monitor with `htop`

---

## ✅ Done

You should now have:

* GCS dataset in `./mp4`
* Hugging Face dataset in your specified directory
