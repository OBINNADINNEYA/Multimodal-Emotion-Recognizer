# src/components/transcription.py
import os, glob, argparse, pandas as pd
import whisper
import numpy as np

def load_split_names(npz_path: str, split: str):
    d = np.load(npz_path, allow_pickle=True)
    key = f"{split}_corpus"
    if key not in d:
        raise ValueError(f"Split '{split}' not in {npz_path}. Keys: {list(d.keys())}")
    corpus = d[key].item()
    return set(corpus.keys())

def collect_files(video_dir: str, exts=(".mp4", ".avi")):
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
    return sorted(files)

def transcribe_videos(video_dir: str, out_csv: str, model_size: str = "small",
                      limit: int = None, npz: str = None, split: str = None):
    # gather files
    files = collect_files(video_dir, exts=(".mp4", ".avi"))
    if npz and split:
        keep = load_split_names(npz, split)
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in keep]

    if limit:
        files = files[:limit]

    print(f"Found {len(files)} files to transcribe in: {video_dir}")
    if not files:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        pd.DataFrame(columns=["name","sentence"]).to_csv(out_csv, index=False)
        print(f"✅ wrote 0 transcripts to {out_csv}")
        return

    model = whisper.load_model(model_size)
    rows = []
    for i, f in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(f))[0]
        print(f"[{i}/{len(files)}] {name}")
        result = model.transcribe(f, fp16=False)
        rows.append({"name": name, "sentence": result.get("text","").strip()})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ wrote {len(df)} transcripts to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True, help="Dir containing .mp4/.avi files")
    ap.add_argument("--out_csv", required=True, help="Output CSV path (name,sentence)")
    ap.add_argument("--model_size", default="small")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--npz", help="Path to label-6way.npz to filter by split")
    ap.add_argument("--split", choices=["train","test1","test2","test3"], help="Which split to filter")
    args = ap.parse_args()
    transcribe_videos(args.video_dir, args.out_csv, args.model_size, args.limit, args.npz, args.split)
