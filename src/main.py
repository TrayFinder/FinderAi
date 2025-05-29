import os
import sys
import argparse

from roboflow import Roboflow
import utils.config as constants
from utils.dataset_preparator import DatasetPreparator
from FinderAi.src.training.detector_trainer import DetectorTrainer
from training.embedding_trainer import EmbeddingTrainer
from utils.logger_class import LoggerClass

LoggerClass.configure(f'detection', debug=True)

def download_dataset():
    LoggerClass.info("📥 Downloading dataset from Roboflow...")
    rf = Roboflow(api_key="le2CyyH9KBXSfalWzMP7")
    project = rf.workspace("cfemodel").project("cverde")
    version = project.version(1)
    dataset = version.download("yolov11", location=constants.DATASETS_DIR + 'cverde-1', overwrite=True)
    return os.path.join(constants.DATASETS_DIR, 'cverde-1')

def prepare_dataset(dataset_dir: str):
    LoggerClass.info("🛠️ Preparing dataset...")
    prep = DatasetPreparator(dataset_dir)

    prep.flatten_all_files()
    prep.remove_unwanted_files()
    prep.remove_empty_dirs()
    prep.update_all_labels()
    prep.split_dataset(train_ratio=0.8)
    prep.create_yaml(class_names=["product"])

    LoggerClass.info("✅ Dataset preparation complete.")

def train_detector(dataset_dir: str):
    LoggerClass.info("🏋️ Training Detection model...")
    trainer = DetectorTrainer(
        project_dir=dataset_dir,
        image_size=640,
        batch_size=16,
        epochs=40,
    )
    trainer.train()

def train_embedding(dataset_dir: str):
    LoggerClass.info("🏋️ Training Embeddings model...")
    trainer = EmbeddingTrainer(
        project_dir=dataset_dir,
        batch_size=32,
        epochs=20
    )

def main(model_type: str):
    dataset_dir = download_dataset()
    prepare_dataset(dataset_dir)

    if model_type == "detector":
        train_detector(dataset_dir)
    elif model_type == "embedding":
        train_embedding(dataset_dir)
    else:
        LoggerClass.info("❌ Invalid model type. Use 'detector' or 'embedding'.")
        sys.exit(1)

    LoggerClass.info("✅ All steps completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model (Detector or Embedding)")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model to train: 'detector' or 'embedding'"
    )
    args = parser.parse_args()

    try:
        main(args.model)
    except Exception as e:
        LoggerClass.info(f"❌ Pipeline failed: {e}")
        sys.exit(1)
