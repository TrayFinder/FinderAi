import os
import sys
import torch
import shutil
from datetime import date
from ultralytics import YOLO
from multiprocessing import cpu_count
import utils.config as constants
from utils.logger_class import LoggerClass

class DetectorTrainer:
    """
    Handles the training and export of a Yolo Detector model using a specified dataset and configuration.
    """

    def __init__(
        self,
        project_dir: str,
        image_size: int,
        batch_size: int,
        epochs: int,
        lr: float = 1e-3,
        model_size: str = 's'
    ):
        self.project_dir = project_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.model_size = model_size
        self.cpu_cores, self.gpu_devices = self._validate_hardware()

    def _validate_hardware(self) -> tuple[int, list[int]]:
        """Check CUDA and return available hardware specs."""
        if not torch.cuda.is_available():
            LoggerClass.debug("CUDA is not available. Training cannot proceed.")
            sys.exit(1)

        cpu_cores = cpu_count()
        gpu_count = torch.cuda.device_count()

        LoggerClass.debug(f"🧠 CPU Cores Available: {cpu_cores}")
        LoggerClass.debug(f"⚡ CUDA GPUs Available: {gpu_count}")

        return cpu_cores, list(range(gpu_count)) if gpu_count > 1 else [0]

    def train(self):
        """
        Train a YOLO model using the dataset.yaml inside project_dir.
        """
        yaml_path = os.path.join(self.project_dir, 'dataset.yaml')
        model_path = f'yolo11{self.model_size}.pt'
        arch_path = f'yolo11{self.model_size}.yaml'

        # LoggerClass.debug(f"📂 Loading model: {model_path}")
        # model = YOLO(arch_path).load(model_path)
        LoggerClass.debug("📂 Loading model..")
        local_model_path = os.path.join(constants.MODELS_DIR, '10_epochs_small.pt')
        model = YOLO(local_model_path)

        LoggerClass.info(f"🚀 Starting training for {self.epochs} epochs")
        model.train(
            data=yaml_path,
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            device=self.gpu_devices,
            lr0=0.001,
            project=constants.MODELS_DIR,
            name="results",
            exist_ok=True,
            patience=4,
            single_cls=True,
            amp=True
        )

        LoggerClass.info("✅ Training completed successfully")

        self._export_model(model)

    def _export_model(self, model):
        """
        Export the trained model to ONNX format.
        """
        results_dir = os.path.join(constants.MODELS_DIR, "results")
        best_model_path = os.path.join(results_dir, "weights", "best.pt")

        if os.path.exists(best_model_path):
            day_dir = os.path.join(constants.MODELS_DIR, str(date.today()))
            os.makedirs(day_dir, exist_ok=True)
            dest_path = os.path.join(day_dir, "best.pt")
            shutil.copy2(best_model_path, dest_path)
            LoggerClass.debug(f"📁 best.pt copied to: {dest_path}")
        else:
            LoggerClass.debug("⚠️ best.pt not found, skipping copy.")
        LoggerClass.info("✅ Model exported successfully")
