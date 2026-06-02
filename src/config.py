"""Project configuration for blood cell classification."""
from pathlib import Path
IMAGE_WIDTH = 80
IMAGE_HEIGHT = 60
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
CLASS_NAMES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
NUCLEAR_GROUPS = {"LYMPHOCYTE": 0, "MONOCYTE": 0, "EOSINOPHIL": 1, "NEUTROPHIL": 1}
NUCLEAR_GROUP_NAMES = ["Mononuclear", "Polynuclear"]
OUTPUT_DIR = Path("outputs")
