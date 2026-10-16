"""Shared configuration for the plant signal classification project."""

from plant_signal.paths import ROOT_DIR


DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = ROOT_DIR / "data_raw"
ANNOTATIONS_DIR = ROOT_DIR / "annotations"
ARTIFACT_DIR = ROOT_DIR / "artifacts"

DATA_CSV = DATA_DIR / "plant_demo.csv"
REAL_DATA_CSV = DATA_DIR / "plant_real_dataset.csv"
ANNOTATION_CSV = ANNOTATIONS_DIR / "trial_annotations.csv"
PREPROCESS_PATH = ARTIFACT_DIR / "preprocess.json"
BEST_MODEL_PATH = ARTIFACT_DIR / "plant_signal_best.keras"
INT8_MODEL_PATH = ARTIFACT_DIR / "plant_signal_int8.tflite"
TRAINING_HISTORY_CSV = ARTIFACT_DIR / "training_history.csv"
TRAINING_CURVES_PNG = ARTIFACT_DIR / "training_curves.png"
CONFUSION_MATRIX_PNG = ARTIFACT_DIR / "confusion_matrix.png"
C_MODEL_CC_PATH = ARTIFACT_DIR / "plant_model_data.cc"
C_MODEL_H_PATH = ARTIFACT_DIR / "plant_model_data.h"

SEED = 1234
WINDOW_SIZE = 128
SAMPLING_RATE_HZ = 500
ANNOTATION_EDGE_MARGIN_SECONDS = 0.3
NUM_CLASSES = 3
CLASS_NAMES = {
    0: "Rest",
    1: "Tap",
    2: "HoldTouch",
}

BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 1.0e-3
EARLY_STOPPING_PATIENCE = 8

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

SCALE_PERCENTILE = 99.0
MIN_PREPROCESS_SCALE = 1.0e-6
REPRESENTATIVE_DATASET_SIZE = 100

DEMO_SESSION_COUNT = 12
DEMO_SAMPLES_PER_CLASS_PER_SESSION = 18
