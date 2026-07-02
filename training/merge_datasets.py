import os
import pandas as pd

# ==============================
# Load Emotion Dataset for NLP
# ==============================

emotion_path = "data/raw/emotions_nlp"

train = pd.read_csv(
    os.path.join(emotion_path, "train.txt"),
    sep=";",
    names=["text", "emotion"]
)

val = pd.read_csv(
    os.path.join(emotion_path, "val.txt"),
    sep=";",
    names=["text", "emotion"]
)

test = pd.read_csv(
    os.path.join(emotion_path, "test.txt"),
    sep=";",
    names=["text", "emotion"]
)

emotion_dataset = pd.concat(
    [train, val, test],
    ignore_index=True
)

print("Emotion Dataset:", len(emotion_dataset))

# ==============================
# Load Clean GoEmotions
# ==============================

goemotion = pd.read_csv(
    "data/processed/goemotions_clean.csv"
)

print("GoEmotions:", len(goemotion))

# ==============================
# Merge
# ==============================

merged = pd.concat(
    [emotion_dataset, goemotion],
    ignore_index=True
)

# Remove duplicate texts
merged = merged.drop_duplicates(subset=["text"])

# Shuffle
merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nFinal Dataset Size:", len(merged))

print("\nEmotion Distribution:\n")
print(merged["emotion"].value_counts())

# ==============================
# Save
# ==============================

os.makedirs("data/processed", exist_ok=True)

merged.to_csv(
    "data/processed/final_dataset.csv",
    index=False
)

print("\nSaved:")
print("data/processed/final_dataset.csv")