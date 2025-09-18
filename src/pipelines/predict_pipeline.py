import sys
import numpy as np
import pandas as pd

from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_object, extract_features_from_file


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, file_path: str):
        """
        Predict emotion from a raw video/audio file.
        """
        try:
            logging.info(f"Running prediction for {file_path}")

            # Load preprocessor + trained model
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # Extract features (audio + video baseline)
            feats = extract_features_from_file(file_path).reshape(1, -1)

            # Preprocess features (scaling etc.)
            feats_transformed = preprocessor.transform(feats)

            # Predict
            preds = model.predict(feats_transformed)
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
        """
        Convert the uploaded file into model-ready feature vector.
        """
        try:
            return extract_features_from_file(self.file_path).reshape(1, -1)
        except Exception as e:
            raise CustomException(e, sys)
