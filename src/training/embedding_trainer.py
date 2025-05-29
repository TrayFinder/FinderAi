import os
import yaml
import torch
import timm
from datetime import date
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from utils.logger_class import LoggerClass
import utils.config as constants

class EmbeddingTrainer:
    """
    Handles the training and export of a EfficientNet Embedding model using a specified dataset and configuration.
    """

    def __init__(
        self,
        project_dir: str,
        batch_size: int,
        epochs: int,
        lr: float = 1e-3,
        model_name: str = 'efficientnetv2_s',
        embedding_dim: int = 256
    ):
        self.project_dir = project_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # read dataset.yaml
        yaml_path = os.path.join(self.project_dir, 'dataset.yaml')
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        self.train_dir = os.path.join(self.project_dir, cfg['train'])
        self.val_dir   = os.path.join(self.project_dir, cfg['val'])
        self.num_classes = len(cfg['names'])
        self.output_dir = os.path.join(constants.SRC_MODELS_DIR, f'embedding_{date.today()}')
        os.makedirs(self.output_dir, exist_ok=True)

        LoggerClass.configure('embedding_trainer', debug=False)
        LoggerClass.info(f"Device: {self.device}")

    def _build_dataloaders(self):
        # common transforms
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(288),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406),
                                 std =(0.229,0.224,0.225))
        ])
        val_tf = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(288),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406),
                                 std =(0.229,0.224,0.225))
        ])

        train_ds = datasets.ImageFolder(self.train_dir, transform=train_tf)
        val_ds   = datasets.ImageFolder(self.val_dir,   transform=val_tf)

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                       shuffle=True,  num_workers=4, pin_memory=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=self.batch_size,
                                       shuffle=False, num_workers=4, pin_memory=True)

        LoggerClass.info(f"🔀 Loaded {len(train_ds)} training and {len(val_ds)} validation samples")

    def _build_model(self):
        # load backbone from timm
        backbone = timm.create_model(self.model_name, pretrained=True, features_only=True)
        feat_dim = backbone.feature_info[-1]['num_chs']  # last feature map channels

        # define head
        self.model = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, self.embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.num_classes)
        ).to(self.device)

        LoggerClass.info(f"🧩 Built model {self.model_name} with embedding dim {self.embedding_dim}")

    def train(self):
        self._build_dataloaders()
        self._build_model()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        best_val_acc = 0.0

        for epoch in range(1, self.epochs+1):
            # training
            self.model.train()
            total, correct = 0, 0
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, preds = torch.max(outputs, 1)
                correct += (preds==labels).sum().item()
                total += labels.size(0)

            train_acc = correct/total
            LoggerClass.info(f"Epoch {epoch}/{self.epochs} — Train Acc: {train_acc:.4f}")

            # validation
            self.model.eval()
            total, correct = 0, 0
            with torch.no_grad():
                for imgs, labels in self.val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    outputs = self.model(imgs)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds==labels).sum().item()
                    total += labels.size(0)
            val_acc = correct/total
            LoggerClass.info(f"Epoch {epoch}/{self.epochs} — Val   Acc: {val_acc:.4f}")

            # save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(self.output_dir, 'best.pth')
                torch.save(self.model.state_dict(), best_path)
                LoggerClass.info(f"💾 Saved best model with Val Acc {best_val_acc:.4f}")

        LoggerClass.info("✅ Training complete")
        self._export_onnx()

    def _export_onnx(self):
        best_path = os.path.join(self.output_dir, 'best.pth')
        if not os.path.exists(best_path):
            LoggerClass.debug("⚠️ No best.pth found, skipping ONNX export")
            return

        # rebuild model and load weights
        self._build_model()
        self.model.load_state_dict(torch.load(best_path, map_location='cpu'))
        self.model.eval()

        # dummy input
        dummy = torch.randn(1, 3, 288, 288, device='cpu')
        onnx_path = os.path.join(self.output_dir, 'model.onnx')
        torch.onnx.export(
            self.model, dummy, onnx_path,
            input_names=['input'], output_names=['output'],
            opset_version=13
        )
        LoggerClass.info(f"🚀 Exported ONNX model to {onnx_path}")
