import os
import pandas as pd

# =====================================
# Load emotion names
# =====================================

with open("data/raw/goemotions/emotions.txt", "r", encoding="utf-8") as f:
    emotion_names = [x.strip() for x in f.readlines()]

# =====================================
# Mapping
# =====================================

emotion_map = {
    "admiration": "love",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "love",
    "confusion": None,
    "curiosity": None,
    "desire": "love",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "anger",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "love",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": None,
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": None
}

# =====================================
# Load GoEmotions
# =====================================

train = pd.read_csv(
    "data/raw/goemotions/train.tsv",
    sep="\t",
    header=None,
    names=["text", "label", "id"]
)

print("Original samples:", len(train))

# =====================================
# Keep only single-label rows
# =====================================

train = train[~train["label"].str.contains(",", regex=False)]

print("Single-label samples:", len(train))

# =====================================
# Convert label id → emotion name
# =====================================

def convert_label(label):

    idx = int(label)

    emotion = emotion_names[idx]

    return emotion_map.get(emotion)

train["emotion"] = train["label"].apply(convert_label)

# =====================================
# Remove unmapped emotions
# =====================================

train = train.dropna(subset=["emotion"])

print("Mapped samples:", len(train))

print("\nEmotion Distribution:\n")

print(train["emotion"].value_counts())

# =====================================
# Save cleaned dataset
# =====================================

os.makedirs("data/processed", exist_ok=True)

train[["text", "emotion"]].to_csv(
    "data/processed/goemotions_clean.csv",
    index=False
)

print("\nSaved:")
print("data/processed/goemotions_clean.csv")