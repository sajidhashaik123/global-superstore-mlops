
from pathlib import Path
import pandas as pd

def load_dataset(project_root):
    file_path = project_root / "data" / "raw" / "global_superstore.csv"

    if not file_path.exists():
        raise FileNotFoundError("Dataset not found!")

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    return df
