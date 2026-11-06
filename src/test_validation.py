from pathlib import Path
from data_loader import load_dataset
from data_validation import validate_dataset

project_root = Path(__file__).resolve().parent.parent

df = load_dataset(project_root)

validate_dataset(df)