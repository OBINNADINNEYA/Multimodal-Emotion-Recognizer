# src/components/run_test.py
import argparse
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--split", type=str, default="train", help="Dataset split: train or test3")
    # args = ap.parse_args()

    # # Step 1: Ingest manifests
    # ingestion = DataIngestion()
    # manifests = ingestion.initiate_data_ingestion([args.split])
    # print("Manifests:", manifests)

    # # Step 2: Transform features
    # transformer = DataTransformation()
    # outdir = f"artifacts/features/{args.split}"
    # features_path = transformer.initiate_data_transformation(manifests[args.split], outdir)
    # print(f"Features for split '{args.split}' saved to {features_path}")

    # Step 3: Train models
    features_path = 'artifacts/features/train'
    trainer = ModelTrainer()
    results = trainer.initiate_model_training(features_path)
    print("Training results:", results)


if __name__ == "__main__":
    main()
