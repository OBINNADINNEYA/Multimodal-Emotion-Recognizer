# src/components/model_trainer.py
import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("data", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        """
        Trains multiple candidate classifiers, evaluates, and saves the best one.
        """
        try:
            logging.info("Splitting training and test input data")

            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=3000, class_weight="balanced"
                ),
                "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=50),
                "Random Forest": RandomForestClassifier(n_estimators=200),
                "Gradient Boosting": GradientBoostingClassifier(),
                "SVM": SVC(probability=True),
            }

            params = {
                "Logistic Regression": {"C": [0.1, 1, 10]},
                "MLP": {"learning_rate_init": [1e-3, 1e-4]},
                "Random Forest": {"n_estimators": [100, 200]},
                "Gradient Boosting": {"learning_rate": [0.1, 0.05]},
                "SVM": {"C": [0.1, 1]},
            }

            logging.info("Evaluating candidate models")
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
                scoring="f1_macro",
            )

            # best score & name
            best_model_score = max(model_report.values())
            best_model_name = [
                name for name, score in model_report.items() if score == best_model_score
            ][0]
            best_model = models[best_model_name]

            if best_model_score < 0.4:
                raise CustomException("No good model found (f1_macro < 0.4)")
            logging.info(f"Best model found: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            preds = best_model.predict(X_test)
            report = classification_report(y_test, preds)
            logging.info("\n" + report)

            return {
                "best_model": best_model_name,
                "f1_macro": best_model_score,
                "report": report,
            }

        except Exception as e:
            raise CustomException(e, sys)
