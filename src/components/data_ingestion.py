# src/components/data_ingestion.py
import os
import sys
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from src.exceptions import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    artifacts_dir: str = "artifacts"
    manifests_dir: str = os.path.join("artifacts", "manifests")
    video_dir: str = "/mnt/merbig/dataset-process/video"  # adjust if needed
    label_npz: str = "/mnt/merbig/dataset-process/label-6way.npz"


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        Path(self.config.manifests_dir).mkdir(parents=True, exist_ok=True)

    def build_manifest(self, split: str):
        """
        Build manifest JSONL for a given split.
        Each line = {"video": path, "label": emotion or None}
        """
        try:
            logging.info(f"Building manifest for split={split}")

            # Load label npz
            data = np.load(self.config.label_npz, allow_pickle=True)
            key = f"{split}_corpus"
            if key not in data:
                raise CustomException(
                    f"Split '{split}' not found in {self.config.label_npz}. Keys={list(data.keys())}"
                )

            corpus = data[key].item()
            out_path = os.path.join(self.config.manifests_dir, f"{split}.jsonl")

            written = 0
            with open(out_path, "w", encoding="utf-8") as fout:
                for vid, meta in corpus.items():
                    # videos are usually .avi, fallback to .mp4
                    video_path = os.path.join(self.config.video_dir, f"{vid}.avi")
                    if not os.path.exists(video_path):
                        video_path = os.path.join(self.config.video_dir, f"{vid}.mp4")
                    if not os.path.exists(video_path):
                        logging.warning(f"Missing video for {vid}, skipping")
                        continue

                    label = meta.get("emo") if "emo" in meta else None
                    entry = {"video": video_path, "label": label}
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    written += 1

            logging.info(f"Wrote {written} rows → {out_path}")
            return out_path

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self, splits=None):
        """
        Orchestrates manifest building for train/test splits.
        """
        try:
            if splits is None:
                splits = ["train", "test3"]

            outputs = {}
            for split in splits:
                outputs[split] = self.build_manifest(split)

            logging.info(" Data ingestion completed")
            return outputs

        except Exception as e:
            raise CustomException(e, sys)


