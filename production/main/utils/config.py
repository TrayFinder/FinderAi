"""
For project-wide constants and sharred configurations

Pattern:
  Directories -> DIR (INPUT_DIR, OUTPUT_DIR): it ends with "/", represents the folder
  Files -> FILE (CONFIG_FILE, LOG_FILE): it contains only the filename
  Full Path -> PATH (LOG_FILE_PATH, DETECTION_MODEL_PATH): it contains the entire path (DIR + FILE)
  Numeric values/limits -> Clear descriptions (MAX_RETRIES, MIN_BUFFER_SIZE)
"""

import os

current_dir    = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR       = os.path.abspath(os.path.join(current_dir, '../../')) + os.sep
PRODUCTION_DIR = os.path.abspath(os.path.join(ROOT_DIR, "production")) + os.sep
ASSETS_DIR     = os.path.abspath(os.path.join(PRODUCTION_DIR, "assets")) + os.sep
MAIN_DIR       = os.path.abspath(os.path.join(PRODUCTION_DIR, "main")) + os.sep
LOGS_DIR       = os.path.abspath(os.path.join(ASSETS_DIR, 'logs')) + os.sep
TRAINING_DIR   = os.path.abspath(os.path.join(MAIN_DIR, "training")) + os.sep
DATASETS_DIR   = os.path.abspath(os.path.join(ASSETS_DIR, "datasets")) + os.sep
MODELS_DIR     = os.path.abspath(os.path.join(ASSETS_DIR, "models")) + os.sep
