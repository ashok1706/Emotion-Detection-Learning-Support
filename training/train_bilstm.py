import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_csv("data/processed/final_dataset.csv")

print(df.head())
print("\nDataset Shape:", df.shape)

# ============================================================
# Label Encoding
# ============================================================

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["emotion"])

print("\nEmotion Classes:")
print(label_encoder.classes_)

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ============================================================
# Tokenization
# ============================================================

VOCAB_SIZE = 20000
MAX_LENGTH = 100

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("\nVocabulary Size:", len(tokenizer.word_index))
print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ============================================================
# Build Model
# ============================================================

print("\nBuilding BiLSTM Model...")

model = Sequential()

model.add(
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=128,
        input_length=MAX_LENGTH
    )
)

model.add(
    Bidirectional(
        LSTM(128)
    )
)

model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))

model.add(Dropout(0.3))

model.add(
    Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# Callbacks
# ============================================================

os.makedirs("models/bilstm", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/bilstm/bilstm_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ============================================================
# Train
# ============================================================

print("\nTraining Started...\n")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[checkpoint, early_stop]
)

# ============================================================
# Evaluate
# ============================================================

print("\nEvaluating...\n")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nAccuracy :", accuracy)

# ============================================================
# Predictions
# ============================================================

predictions = model.predict(X_test)

y_pred = predictions.argmax(axis=1)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))

# ============================================================
# Save Tokenizer & Label Encoder
# ============================================================

joblib.dump(
    tokenizer,
    "models/bilstm/tokenizer.pkl"
)

joblib.dump(
    label_encoder,
    "models/bilstm/label_encoder.pkl"
)

print("\nTokenizer Saved")
print("Label Encoder Saved")

# ============================================================
# Training Graph
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("BiLSTM Accuracy")

plt.legend()

plt.savefig("models/bilstm/accuracy.png")

plt.close()

plt.figure(figsize=(10,5))

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BiLSTM Loss")

plt.legend()

plt.savefig("models/bilstm/loss.png")

plt.close()

print("\nGraphs Saved")

print("\n" + "=" * 60)
print("BiLSTM TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)