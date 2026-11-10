"""
config.py — ALL folder paths and settings of the project, in ONE place.

Every other file imports what it needs from here, for example:
    from config import SAVED_MODEL_DIR
So if you want to move a folder or change a setting, you only change it here.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# 1. Folders
# ----------------------------------------------------------------------------
# PROJECT_DIR = the folder that contains this config.py file
PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"                 # small files that belong to the project (advice.csv)
DATASETS_DIR = DATA_DIR / "datasets"            # the big image datasets are downloaded here
PLANTVILLAGE_DIR = DATASETS_DIR / "plantvillage"   # lab photos  (training)
PLANTDOC_DIR = DATASETS_DIR / "plantdoc"           # field photos (testing only)
RICE_DIR = DATASETS_DIR / "rice"                   # a crop the model has never seen ("unknown" test)

SAVED_MODEL_DIR = PROJECT_DIR / "saved_model"   # the notebook SAVES the trained model here, the app LOADS it
OUTPUTS_DIR = PROJECT_DIR / "outputs"           # results, figures and image lists for the report / paper
DATABASE_DIR = PROJECT_DIR / "database"         # the app stores its prediction history here

# ----------------------------------------------------------------------------
# 2. Files
# ----------------------------------------------------------------------------
MODEL_FILE = SAVED_MODEL_DIR / "plant_model.onnx"        # the model used by the app
MODEL_INFO_FILE = SAVED_MODEL_DIR / "model_info.json"    # class names, thresholds, temperature ...
PYTORCH_WEIGHTS_FILE = SAVED_MODEL_DIR / "plant_model.pt"  # PyTorch copy (only needed to train again)
ADVICE_FILE = DATA_DIR / "advice.csv"                    # what to tell the farmer for each group
DATABASE_FILE = DATABASE_DIR / "predictions.db"          # SQLite database file

# ----------------------------------------------------------------------------
# 3. How images are prepared (MUST be the same in training and in the app)
# ----------------------------------------------------------------------------
RESIZE_SIZE = 256     # first make the shorter side 256 pixels ...
IMAGE_SIZE = 224      # ... then cut out the 224 x 224 centre
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # average colour of ImageNet photos (red, green, blue)
IMAGENET_STD = [0.229, 0.224, 0.225]    # spread of the colours of ImageNet photos

# ----------------------------------------------------------------------------
# 4. Conformal prediction
# ----------------------------------------------------------------------------
ALPHA = 0.10          # allowed error rate: the true group must be in the answer 90% of the time

# ----------------------------------------------------------------------------
# 5. Dataset download links (all public, on GitHub)
# ----------------------------------------------------------------------------
PLANTVILLAGE_URL = "https://github.com/spMohanty/PlantVillage-Dataset"
PLANTDOC_URL = "https://github.com/pratikkayal/PlantDoc-Dataset"
# PlantDoc is downloaded as a ZIP (not with git): some of its file names contain "?" and are too long,
# which Windows does not allow, so git fails there. The ZIP lets us rename those files while unpacking.
PLANTDOC_ZIP_URL = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
RICE_URL = "https://github.com/PawarMukesh/RiceLeaf-Disease-Detector"
