import os
import yaml
import shutil
import itertools
from pathlib import Path
from random import random
from multiprocessing import Pool


class DatasetPreparator:
    """
    A utility class for preparing local datasets:
    - Moves files from subfolders to main directory
    - Splits files into train/val
    - Cleans unwanted files and empty folders
    - Updates label files
    - Generates a dataset.yaml
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────
    # FILE ORGANIZATION
    # ─────────────────────────────────────────────

    def flatten_all_files(self) -> None:
        """Move all files from subdirectories into the base directory."""
        print("Flattening all files...")
        files = [f for f in self.base_dir.rglob("*") if f.is_file()]
        args = [(str(f), str(self.base_dir)) for f in files]

        with Pool() as pool:
            pool.starmap(self._move_file_safely, args)

    @staticmethod
    def _move_file_safely(src: str, dst_dir: str) -> None:
        try:
            src_path = Path(src)
            dst_path = Path(dst_dir) / src_path.name

            if dst_path.exists() and dst_path.suffix == ".jpg":
                dst_path.unlink()
            shutil.move(str(src_path), str(dst_path))
        except Exception as e:
            print(f"Error moving {src} -> {dst_dir}: {e}")

    def split_dataset(self, train_ratio=0.8) -> None:
        """
        Split images and labels into train/val folders under images/ and labels/.
        """
        print("Splitting dataset into train/val...")
        self._create_split_dirs()

        files = [f for f in self.base_dir.iterdir() if f.suffix in {".jpg", ".png", ".txt"}]
        args = [(str(f), train_ratio) for f in files]

        with Pool() as pool:
            pool.starmap(self._assign_to_split_folder, args)

    def _create_split_dirs(self):
        for category, split in itertools.product(["images", "labels"], ["train", "val"]):
            (self.base_dir / category / split).mkdir(parents=True, exist_ok=True)

    def _assign_to_split_folder(self, file_path: str, train_ratio: float):
        fpath = Path(file_path)
        category = "images" if fpath.suffix in {".jpg", ".png"} else "labels"
        split = "train" if random() < train_ratio else "val"

        dest = self.base_dir / category / split / fpath.name
        try:
            shutil.move(str(fpath), str(dest))
        except Exception as e:
            print(f"Failed to move {fpath} to {dest}: {e}")

    # ─────────────────────────────────────────────
    # LABEL CLEANING
    # ─────────────────────────────────────────────

    def update_all_labels(self) -> None:
        """Reset class index (first token) to 0 for all .txt label files."""
        print("Updating label files...")
        txt_files = [f for f in self.base_dir.rglob("*.txt")]
        with Pool() as pool:
            pool.map(self._reset_label_classes, txt_files)

    @staticmethod
    def _reset_label_classes(file_path: Path):
        try:
            with file_path.open("r+", encoding="utf-8") as file:
                lines = file.readlines()
                updated = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        parts[0] = '0'
                        updated.append(" ".join(parts) + "\n")
                file.seek(0)
                file.writelines(updated)
                file.truncate()
        except Exception as e:
            print(f"Failed to update {file_path}: {e}")

    # ─────────────────────────────────────────────
    # CLEANING
    # ─────────────────────────────────────────────

    def remove_unwanted_files(self) -> None:
        """Delete files that are not .jpg, .png, or .txt."""
        print("Removing unwanted files...")
        allowed = {".jpg", ".png", ".txt"}
        for f in self.base_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() not in allowed:
                try:
                    f.unlink()
                    print(f"Removed {f}")
                except Exception as e:
                    print(f"Error removing {f}: {e}")

    def remove_empty_dirs(self) -> None:
        """Delete empty subfolders."""
        for root, dirs, _ in os.walk(self.base_dir, topdown=False):
            for d in dirs:
                path = Path(root) / d
                if not any(path.iterdir()):
                    try:
                        path.rmdir()
                        print(f"Removed empty directory: {path}")
                    except Exception as e:
                        print(f"Failed to remove {path}: {e}")

    # ─────────────────────────────────────────────
    # YAML GENERATION
    # ─────────────────────────────────────────────

    def create_yaml(self, class_names: list[str] | None = None):
        """Create dataset.yaml pointing to train/val images with class names."""
        class_names = class_names or []
        data = {
            "train": str(self.base_dir / "images/train"),
            "val": str(self.base_dir / "images/val"),
            "nc": len(class_names),
            "names": class_names,
        }

        yaml_path = self.base_dir / "dataset.yaml"
        try:
            with yaml_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            print(f"Created: {yaml_path}")
        except Exception as e:
            print(f"Failed to create YAML: {e}")
