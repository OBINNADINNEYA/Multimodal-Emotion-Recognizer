# scripts/download_from_blob_streaming.py
import os, pathlib, math
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm

CONN = os.environ["AZURE_CONN_STR"]
CONTAINER = "merdata-23"
DEST = pathlib.Path(".").resolve()

svc  = BlobServiceClient.from_connection_string(CONN)
cont = svc.get_container_client(CONTAINER)

DEST.mkdir(parents=True, exist_ok=True)

# optionally limit to specific files while testing:
# TARGETS = {"mer2023train.z01","mer2023train.z02","mer2023train.z03","mer2023train.z04","mer2023train.z05","mer2023train.z06","mer2023train.zip","mer2023test1&2.zip","test-labels.zip"}
TARGETS = None  # download all if None

for b in cont.list_blobs():
    name = b.name.split("/")[-1]
    if TARGETS and name not in TARGETS:
        continue
    path = DEST / name
    if path.exists() and path.stat().st_size == b.size:
        print("exists (skipping):", name)
        continue

    bc = cont.get_blob_client(b)
    total = b.size or 0
    chunk = 8 * 1024 * 1024   # 8 MB

    print("downloading:", name, "->", path)
    with tqdm(total=total, unit="B", unit_scale=True, desc=name) as pbar:
        stream = bc.download_blob(max_concurrency=8)
        with open(path, "wb") as f:
            for chunk_bytes in stream.chunks():  # streams sequentially; no big RAM
                f.write(chunk_bytes)
                pbar.update(len(chunk_bytes))

print("✅ done")
