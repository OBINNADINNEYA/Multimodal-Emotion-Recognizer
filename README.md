# Multimodal-Emotion-Recognizer
MER End2End



# MER2023 Dataset Setup (Azure Blob + VM)

This document explains step-by-step how I prepared the **MER2023 dataset** for use in my project, using Azure Blob Storage and an Azure VM. It covers:

1. Creating the storage container  
2. Setting up environment variables  
3. Streaming dataset parts from Hugging Face into Blob  
4. Creating and connecting to an Azure VM  
5. Installing prerequisites on the VM  
6. Downloading and extracting the dataset with the correct passwords  

---

## 0. Prerequisites

- Azure Subscription
- Hugging Face account with **read token** and MER2023 terms accepted
- Local machine with Python 3.10+ and `pip`
- **Do not commit `.env`** — add it to `.gitignore`

---

## 1. Azure Storage + Container

- **Storage account**: `mymlprojectsstorage`  
- **Container**: `merdata-23`  

### Option A — Portal  
Go to your storage account → **Data storage → Containers → + Container** → name it `merdata-23`.

### Option B — CLI
```bash
az login
az storage container create \
  --account-name mymlprojectsstorage \
  --name merdata-23

#.env file
# Azure
AZURE_ACCOUNT_NAME=mymlprojectsstorage
AZURE_ACCOUNT_KEY=<YOUR_ACCOUNT_KEY>
AZURE_CONN_STR=DefaultEndpointsProtocol=https;AccountName=mymlprojectsstorage;AccountKey=<YOUR_ACCOUNT_KEY>;EndpointSuffix=core.windows.net
AZURE_CONTAINER=merdata-23

# Hugging Face
HF_TOKEN=hf_xxx...   # from Hugging Face Settings → Access Tokens


#python env local 
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests tqdm python-dotenv azure-storage-blob adlfs


4. Parallel Uploader (Hugging Face → Azure Blob
File: upload_mer_parallel.py
Purpose: stream MER2023 dataset parts (.z01–.z06 + .zip) directly into Azure Blob
Features:
Parallel uploads (3 files at a time)
Chunked streaming (8 MB)
Automatic retries
Idempotent (overwrite=True)
Run:
python upload_mer_parallel.py