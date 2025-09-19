# src/pipelines/predict_pipeline.py
import os, sys
import numpy as np
from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_object, extract_features_from_file


class PredictPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, file_path: str):
        """
        Predict emotion from a raw video/audio file.
        """
        try:
            logging.info(f"Running prediction for {file_path}")

            # Step 1: Extract features
            feats = extract_features_from_file(file_path).reshape(1, -1)

            # Step 2: Load model
            model = load_object(self.model_path)

            # Step 3: Optional preprocessor
            if os.path.exists(self.preprocessor_path):
                logging.info("Applying preprocessor")
                preprocessor = load_object(self.preprocessor_path)
                feats = preprocessor.transform(feats)

            # Step 4: Predict
            preds = model.predict(feats)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Represents an upload from user (video/audio file).
    For prediction, only the file path is required.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_data_as_features(self) -> np.ndarray:
        try:
            return extract_features_from_file(self.file_path).reshape(1, -1)
        except Exception as e:
            raise CustomException(e, sys)
