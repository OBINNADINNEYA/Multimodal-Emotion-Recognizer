# upload_mer_parallel.py
import os, re, time, requests
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

# --- Load .env from CWD, then fall back to HOME ---
loaded = load_dotenv() or load_dotenv(os.path.expanduser("~/.env"))

ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
ACCOUNT_KEY  = os.getenv("AZURE_ACCOUNT_KEY")
CONN_STR     = os.getenv("AZURE_CONN_STR")  # use if present
CONTAINER    = (os.getenv("AZURE_CONTAINER") or "").strip().lower()
HF_TOKEN     = os.getenv("HF_TOKEN")

# Validate container name (3–63 chars, lowercase letters/digits/hyphens)
if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{1,61})[a-z0-9]", CONTAINER):
    raise ValueError(f"AZURE_CONTAINER invalid: {CONTAINER!r}")

if not HF_TOKEN:
    raise SystemExit("HF_TOKEN missing. Create one on HF (read scope) and add to .env")

# Build client: prefer full connection string if provided
if CONN_STR:
    svc = BlobServiceClient.from_connection_string(CONN_STR)
else:
    if not (ACCOUNT_NAME and ACCOUNT_KEY):
        raise SystemExit("Provide AZURE_CONN_STR or AZURE_ACCOUNT_NAME and AZURE_ACCOUNT_KEY in .env")
    conn_str = (
        f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};"
        f"AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
    )
    svc = BlobServiceClient.from_connection_string(conn_str)

container_client = svc.get_container_client(CONTAINER)
# Create if missing
if not container_client.exists():
    try:
        container_client.create_container()
        print(f"Created container: {CONTAINER}")
    except ResourceExistsError:
        print(f"Container already exists: {CONTAINER}")
else:
    print(f"Container exists: {CONTAINER}")

HF_BASE = "https://huggingface.co/datasets/MERChallenge/MER2023/resolve/main/"
# FILES = [
#     "mer2023train.z01","mer2023train.z02","mer2023train.z03",
#     "mer2023train.z04","mer2023train.z05","mer2023train.z06",
#     "mer2023train.zip",
# ]

FILES = [
    "mer2023train.z05",           # the failed one
    "mer2023test1&2.zip",         # add test archive
    "test-labels.zip",            # add labels
]

def upload_one(fname, chunk_mb=8, max_retries=4):
    url = HF_BASE + fname
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    for attempt in range(max_retries):
        try:
            with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                blob = container_client.get_blob_client(fname)

                with tqdm(total=total, unit="B", unit_scale=True, desc=fname) as pbar:
                    def gen():
                        for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                            if chunk:
                                pbar.update(len(chunk))
                                yield chunk
                    # Stream to Azure in blocks (concurrency speeds this up)
                    blob.upload_blob(gen(), overwrite=True, max_concurrency=8)
            return f"ok:{fname}"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"fail:{fname}:{e}"

if __name__ == "__main__":
    print("Using container:", repr(CONTAINER))
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(upload_one, f, 8) for f in FILES]  # bump to 16 MB if you like
        for fut in as_completed(futs):
            print(fut.result())
    print("Done.")
