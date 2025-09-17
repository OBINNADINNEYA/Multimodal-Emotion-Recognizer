import os, os.path as op, argparse, json, glob, numpy as np
from typing import Dict

def load_npz_labels(npz_path: str, split: str) -> Dict[str, dict]:
    d = np.load(npz_path, allow_pickle=True)
    key = f"{split}_corpus"
    if key not in d:
        raise ValueError(f"Split '{split}' not found in {npz_path}. Keys: {list(d.keys())}")
    return d[key].item()  # {name: {'emo':..., 'val':...}}

def index_videos(video_dir: str) -> Dict[str,str]:
    idx = {}
    for p in glob.glob(op.join(video_dir, "*")):
        if op.isfile(p):
            idx[op.splitext(op.basename(p))[0]] = p
    return idx

def write_manifest(video_dir: str, npz_path: str, split: str, out_path: str) -> int:
    os.makedirs(op.dirname(out_path), exist_ok=True)
    labels = load_npz_labels(npz_path, split)
    vids = index_videos(video_dir)
    n = 0
    with open(out_path, "w") as f:
        for name, info in labels.items():
            vp = vids.get(name)
            if not vp:
                continue
            rec = {"name": name, "split": split, "video": vp, "label": info.get("emo","neutral"), "text": ""}
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--npz", required=True)  # /mnt/merbig/dataset-process/label-6way.npz
    ap.add_argument("--split", choices=["train","test1","test2","test3"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = write_manifest(args.video_dir, args.npz, args.split, args.out)
    print(f" wrote {n} rows → {args.out}")

if __name__ == "__main__":
    main()
