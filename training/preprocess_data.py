import os
import pandas as pd

# ==========================================
# Emotion Dataset for NLP
# ==========================================

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

print("=" * 60)
print("EMOTION DATASET FOR NLP")
print("=" * 60)

print(f"Train Shape      : {train.shape}")
print(f"Validation Shape : {val.shape}")
print(f"Test Shape       : {test.shape}")

print("\nFirst 5 Training Samples:\n")
print(train.head())

# ==========================================
# GoEmotions Dataset
# ==========================================

goemotion_path = "data/raw/goemotions"

go_train = pd.read_csv(
    os.path.join(goemotion_path, "train.tsv"),
    sep="\t",
    header=None,
    names=["text", "label", "id"]
)

go_dev = pd.read_csv(
    os.path.join(goemotion_path, "dev.tsv"),
    sep="\t",
    header=None,
    names=["text", "label", "id"]
)

go_test = pd.read_csv(
    os.path.join(goemotion_path, "test.tsv"),
    sep="\t",
    header=None,
    names=["text", "label", "id"]
)

print("\n")
print("=" * 60)
print("GOEMOTIONS DATASET")
print("=" * 60)

print(f"Train Shape      : {go_train.shape}")
print(f"Validation Shape : {go_dev.shape}")
print(f"Test Shape       : {go_test.shape}")

print("\nFirst 5 Training Samples:\n")
print(go_train.head())

print("\n")
print("=" * 60)
print("GOEMOTIONS LABEL EXAMPLES")
print("=" * 60)

print(go_train["label"].head(20))

print("\n")
print("=" * 60)
print("PREPROCESSING STEP COMPLETED SUCCESSFULLY")
print("=" * 60)