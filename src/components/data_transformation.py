import os, os.path as op, argparse, json, numpy as np, tempfile, subprocess
from tqdm import tqdm

from src.components.features.audio_wav2vec2 import AudioWav2Vec2Utt
from src.components.features.video_resnet   import VideoResNet18Utt
from src.components.features.text_minilm    import TextMiniLMUtt


def ensure_wav16k(video_path:str) -> str:
    tmp_wav = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg","-loglevel","error","-y","-i",video_path,"-ar","16000","-ac","1",tmp_wav]
    subprocess.run(cmd, check=True)
    return tmp_wav

def run(manifest:str, outdir:str, device:str):
    os.makedirs(outdir, exist_ok=True)
    a_enc = AudioWav2Vec2Utt(device=device)
    v_enc = VideoResNet18Utt(device=device)
    t_enc = TextMiniLMUtt()

    with open(manifest) as f:
        rows = [json.loads(x) for x in f]

    for ex in tqdm(rows, desc="features"):
        name, vpath, label = ex["name"], ex["video"], ex.get("label","neutral")
        # video
        x_v = v_enc(vpath)
        # audio from video
        wav = ensure_wav16k(vpath)
        x_a = a_enc(wav)
        try: os.remove(wav)
        except: pass
        # text
        x_t = t_enc(ex.get("text",""))
        # save
        np.savez_compressed(op.join(outdir, f"{name}.npz"),
                            x_audio=x_a, x_video=x_v, x_text=x_t, y=label, name=name)
    print(f" features saved to {outdir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cpu")  # "cuda:0" if you have GPU
    args = ap.parse_args()
    run(args.manifest, args.outdir, args.device)

if __name__ == "__main__":
    main()
