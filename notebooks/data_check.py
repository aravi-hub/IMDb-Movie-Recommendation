from pathlib import Path
import pandas as pd

# Project root folder
project_root = Path(__file__).resolve().parent.parent

# CSV file path
file_path = project_root / "data" / "imdb_movies_2024.csv"

print("Looking for file at:")
print(file_path)

if not file_path.exists():
    print("\nERROR: CSV file not found!")
    print("Please check:")
    print("1. data folder exists")
    print("2. CSV file is inside data folder")
    print("3. CSV filename is correct")
    exit()

df = pd.read_csv(file_path)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())