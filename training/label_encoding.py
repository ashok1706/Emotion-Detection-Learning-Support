import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
train_df = pd.read_csv(
    "data/raw/train.txt",
    sep=";",
    names=["text", "emotion"]
)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df["clean_text"] = train_df["text"].apply(clean_text)

# Encode emotion labels
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["emotion"])

print("Emotion Mapping:\n")

for emotion, label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{emotion} --> {label}")

print("\nFirst 5 Rows:\n")
print(train_df[["clean_text", "emotion", "label"]].head())