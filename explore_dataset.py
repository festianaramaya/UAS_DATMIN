import pandas as pd

# membaca dataset

df = pd.read_csv(
    "dataset/train.txt",
    sep=";",
    names=["text", "emotion"]
)

print("=" * 60)
print("5 Data Pertama")
print("=" * 60)

print(df.head())

print()

print("=" * 60)
print("Jumlah Data")
print("=" * 60)

print(df.shape)

print()

print("=" * 60)
print("Nama Kolom")
print("=" * 60)

print(df.columns)

print()

print("=" * 60)
print("Distribusi Label")
print("=" * 60)

print(df["emotion"].value_counts())

print()

print("=" * 60)
print("Missing Value")
print("=" * 60)

print(df.isnull().sum())