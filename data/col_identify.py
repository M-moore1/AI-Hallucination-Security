from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "TestData.csv"

df = pd.read_csv(csv_path)

print(df.head())