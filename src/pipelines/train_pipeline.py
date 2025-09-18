import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object, load_object


@dataclass
class TrainPipelineConfig:
    artifacts_dir: str = "artifacts"
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class TrainPipeline:
    def __init__(self):
        self.config = TrainPipelineConfig()

    def initiate_training(self, X: np.ndarray, y: np.ndarray):
        """
        Full training pipeline:
          - split train/test
          - fit preprocessor + model
          - save both artifacts
        """
        try:
            logging.info("===== Starting training pipeline =====")

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logging.info(f"Train={len(X_train)} | Test={len(X_test)}")

            # Preprocessor
            preprocessor = StandardScaler()
            preprocessor.fit(X_train)

            X_train_scaled = preprocessor.transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Simple baseline model (can swap for MLP/Transformer later)
            model = LogisticRegression(
                max_iter=3000,
                n_jobs=4,
                class_weight="balanced"
            )
            model.fit(X_train_scaled, y_train)

            acc = model.score(X_test_scaled, y_test)
            logging.info(f"Validation accuracy={acc:.4f}")

            # Save artifacts
            os.makedirs(self.config.artifacts_dir, exist_ok=True)
            save_object(self.config.model_path, model)
            save_object(self.config.preprocessor_path, preprocessor)

            logging.info("Artifacts saved successfully")
            return {
                "accuracy": acc,
                "model_path": self.config.model_path,
                "preprocessor_path": self.config.preprocessor_path,
            }

        except Exception as e:
            raise CustomException(e, sys)
