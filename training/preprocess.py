import re
import pandas as pd

# Load the training dataset
train_df = pd.read_csv(
    "data/raw/train.txt",
    sep=";",
    names=["text", "emotion"]
)

# Function to clean text
def clean_text(text):
    text = str(text).lower()                  # Convert to lowercase
    text = re.sub(r"http\S+", "", text)       # Remove URLs
    text = re.sub(r"[^a-zA-Z ]", "", text)    # Keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Apply cleaning
train_df["clean_text"] = train_df["text"].apply(clean_text)

# Display the first 5 rows
print("First 5 rows after preprocessing:\n")
print(train_df[["text", "clean_text"]].head())

# Display dataset information
print("\nDataset Shape:")
print(train_df.shape)

print("\nEmotion Counts:")
print(train_df["emotion"].value_counts())