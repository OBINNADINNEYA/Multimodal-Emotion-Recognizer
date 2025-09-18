# src/components/data_transformation.py
import os
import sys
import json
import subprocess
import tempfile
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from src.exceptions import CustomException
from src.logger import logging

# import feature extractors
from src.components.features.audio_wav2vec2 import AudioWav2Vec2Utt
from src.components.features.text_minilm import TextMiniLMUtt
from src.components.features.video_resnet import VideoResNet18Utt


@dataclass
class DataTransformationConfig:
    features_dir: str = os.path.join("data", "features")
    device: str = "cpu"


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        Path(self.config.features_dir).mkdir(parents=True, exist_ok=True)

        # initialize extractors once
        self.audio_extractor = AudioWav2Vec2Utt(device=self.config.device)
        self.text_extractor = TextMiniLMUtt()
        self.video_extractor = VideoResNet18Utt(device=self.config.device)

    def ensure_wav16k(self, video_path: str) -> str:
        """Extract mono 16kHz wav from video using ffmpeg"""
        try:
            tmp_wav = tempfile.mktemp(suffix=".wav")
            cmd = [
                "ffmpeg", "-loglevel", "error", "-y",
                "-i", video_path,
                "-ar", "16000", "-ac", "1", tmp_wav
            ]
            subprocess.run(cmd, check=True)
            return tmp_wav
        except Exception as e:
            raise CustomException(f"ffmpeg failed on {video_path}: {e}", sys)

    def process_entry(self, entry: dict, outdir: str):
        """Extract A/V/T features for one entry and save to npz"""
        try:
            video_path, label = entry["video"], entry.get("label")

            # audio
            wav = self.ensure_wav16k(video_path)
            a_feat = self.audio_extractor(wav)
            os.remove(wav)

            # video
            v_feat = self.video_extractor(video_path)

            # text (optional, may be missing)
            t_feat = self.text_extractor(entry.get("transcript", ""))

            out_path = os.path.join(outdir, os.path.basename(video_path) + ".npz")
            np.savez_compressed(out_path, x_audio=a_feat, x_video=v_feat, x_text=t_feat, y=label)
            return out_path

        except Exception as e:
            logging.warning(f"Failed to process {entry.get('video')}: {e}")
            return None

    def initiate_data_transformation(self, manifest_path: str, outdir: str):
        """Main entry point"""
        Path(outdir).mkdir(parents=True, exist_ok=True)
        written = 0
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    out_path = self.process_entry(entry, outdir)
                    if out_path:
                        written += 1
            logging.info(f"Wrote {written} feature files → {outdir}")
            return outdir
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    transformer = DataTransformation()
    transformer.config.device = args.device
    transformer.run(args.manifest, args.outdir)
