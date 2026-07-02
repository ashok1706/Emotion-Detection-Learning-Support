import pandas as pd

# Load training dataset
train_df = pd.read_csv(
    "data/raw/train.txt",
    sep=";",
    names=["text", "emotion"]
)

print("First 5 rows:")
print(train_df.head())

print("\nDataset Shape:")
print(train_df.shape)

print("\nEmotion Counts:")
print(train_df["emotion"].value_counts())