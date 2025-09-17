import os, os.path as op, argparse, numpy as np, joblib, tempfile, subprocess
from features.audio_wav2vec2 import AudioWav2Vec2Utt
from features.video_resnet   import VideoResNet18Utt
from features.text_minilm    import TextMiniLMUtt

def ensure_wav16k(video_path:str) -> str:
    tmp_wav = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg","-loglevel","error","-y","-i",video_path,"-ar","16000","-ac","1",tmp_wav]
    subprocess.run(cmd, check=True)
    return tmp_wav

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    a_enc = AudioWav2Vec2Utt(device=args.device)
    v_enc = VideoResNet18Utt(device=args.device)
    t_enc = TextMiniLMUtt()

    x_v = v_enc(args.video)
    wav = ensure_wav16k(args.video); x_a = a_enc(wav); os.remove(wav)
    x_t = t_enc(args.text)
    x = np.concatenate([x_a, x_v, x_t]).astype(np.float32).reshape(1,-1)

    clf = joblib.load(args.model)
    pred = clf.predict(x)[0]
    print("🎯 Prediction:", pred)

if __name__ == "__main__":
    main()
