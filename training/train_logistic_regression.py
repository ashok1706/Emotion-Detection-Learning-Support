import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================
# Load Dataset
# =====================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("data/processed/final_dataset.csv")

print(df.head())
print("\nDataset Shape:", df.shape)

# =====================================
# Encode Labels
# =====================================

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["emotion"])

print("\nEmotion Classes:")
print(label_encoder.classes_)

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# =====================================
# TF-IDF
# =====================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("\nTF-IDF Shape:", X_train.shape)

# =====================================
# Logistic Regression
# =====================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("Training Completed!")

# =====================================
# Prediction
# =====================================

y_pred = model.predict(X_test)

# =====================================
# Evaluation
# =====================================

accuracy = accuracy_score(y_test, y_pred)

print("\n")
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))

# =====================================
# Save Model
# =====================================

save_path = "models/logistic"

os.makedirs(save_path, exist_ok=True)

joblib.dump(
    model,
    os.path.join(save_path, "emotion_model.pkl")
)

joblib.dump(
    vectorizer,
    os.path.join(save_path, "tfidf_vectorizer.pkl")
)

joblib.dump(
    label_encoder,
    os.path.join(save_path, "label_encoder.pkl")
)

print("\n")
print("=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("Location:", save_path)