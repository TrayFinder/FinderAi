import os
import sys

from roboflow import Roboflow
import utils.config as constants
from utils.dataset_preparator import DatasetPreparator
from training.yolo_model_trainer import YoloTrainer
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


def train_model(dataset_dir: str):
    LoggerClass.info("🏋️ Starting training process...")
    trainer = YoloTrainer(
        project_dir=dataset_dir,
        image_size=640,
        batch_size=16,
        epochs=40,
        model_size='s',
    )
    trainer.train()


def main():
    dataset_dir = download_dataset()
    prepare_dataset(dataset_dir)
    train_model(dataset_dir)
    LoggerClass.info("✅ All steps completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        LoggerClass.info(f"❌ Pipeline failed: {e}")
        sys.exit(1)
