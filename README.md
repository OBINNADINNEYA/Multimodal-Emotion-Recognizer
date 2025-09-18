# Multimodal-Emotion-Recognizer
End-to-end reimplementation of the **MER2023** multimodal emotion recognition baseline with custom preprocessing, feature extraction, and model training.

---

## MER2023 Dataset Setup (Azure Blob + VM)

This project runs on large multimodal data (video, audio, text). Below is the exact setup I used on Azure.

---

## 0. Prerequisites
- Azure Subscription  
- Hugging Face account with **read token** + accepted MER2023 terms  
- Local Python ≥3.10  
- **Do not commit `.env`** (add to `.gitignore`)  

---

## 1. Azure Blob Storage
- **Storage account**: `mymlprojectsstorage`  
- **Container**: `merdata-23`  

```bash
az login
az storage container create \
  --account-name mymlprojectsstorage \
  --name merdata-23

2. Upload Dataset Parts
Use the custom uploader (upload_mer_parallel.py) to stream .z01–.z06 + .zip from Hugging Face into Blob.
Parallelized (3 at once)
Chunked (8 MB)
Retry-safe
python upload_mer_parallel.py


Azure VM Setup
Created Ubuntu VM
Attached a 512 GB data disk (/dev/sdb) for MER data

# format + mount
sudo mkfs.ext4 /dev/sdb
sudo mkdir -p /mnt/merbig
sudo mount /dev/sdb /mnt/merbig
sudo chown -R azureuser:azureuser /mnt/merbig

# make persistent
sudo blkid /dev/sdb   # get UUID
sudo nano /etc/fstab  # add UUID line
sudo mount -a


Download & Extract

Download from Blob into /mnt/merbig and unzip with 7z.
Passwords provided by MER2023 team.

cd /mnt/merbig

# train
7z x "mer2023train.zip" -omer2023train -p234151723 -y -aos -bsp1 -bso0

# test
7z x "mer2023test1&2.zip" -omer2023test -p092363023 -y
7z x "test-labels.zip"    -otest_labels -pcfn3oi4rjhonvco -y



Normalize Dataset
Run main_baseline.py (copied into project root) to unify dataset structure.

python3 main_baseline.py normalize_dataset_format \
  --data_root="/mnt/merbig/mer2023train" \
  --save_root="/mnt/merbig/dataset-process"

  Pipeline (In-Repo Components)

We use the modular scripts in src/components.

(a) Data ingestion

Build JSONL manifests from videos + labels.

python3 -m src.components.data_ingestion \
  --video_dir /mnt/merbig/dataset-process/video \
  --npz /mnt/merbig/dataset-process/label-6way.npz \
  --split train \
  --out data/manifests/train.jsonl

(b) Data transformation

Extract features (audio embeddings, text MiniLM, video stubs).

python3 -m src.components.data_transformation \
  --manifest data/manifests/train.jsonl \
  --outdir data/features/train \
  --device cpu

(c) Model training

Train baseline (LogReg/MLP with balanced classes).

python3 -m src.components.model_trainer \
  --features_dir data/features/train \
  --model_out models/mer_baseline.pkl

(d) Evaluation

Generate report on held-out or subsampled set.

python3 -m src.components.model_evaluator \
  --features_dir data/features/train \
  --model models/mer_baseline.pkl \
  --report_out reports/train_report.txt

7. Troubleshooting Notes
Timeouts during unzip → use tmux + 7z -aos (skip existing files).
“No such file” for dataset-process → disk not mounted, remount /mnt/merbig.
Empty text embeddings → ensure transcription step runs before data_transformation.
Small training set (~675) → caused by filtered/limited manifests; use full manifest to train on 3,373+ samples.