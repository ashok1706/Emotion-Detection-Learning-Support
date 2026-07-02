import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

############################################################
# Reproducibility
############################################################

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

############################################################
# Load Dataset
############################################################

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_csv("data/processed/final_dataset.csv")

print(df.head())
print()

print("Dataset Shape:", df.shape)
print()

############################################################
# Encode Labels
############################################################

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["emotion"])

print("Emotion Classes:")

print(label_encoder.classes_)
print()

############################################################
# Train/Test Split
############################################################

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"],
    df["label"],
    test_size=0.20,
    random_state=SEED,
    stratify=df["label"],
)

print("Training Samples :", len(train_texts))
print("Testing Samples  :", len(test_texts))
print()

############################################################
# Tokenizer
############################################################

print("Loading DistilBERT Tokenizer...")

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=128,
)

test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=128,
)

print("Tokenization Completed")
print()
############################################################
# PyTorch Dataset
############################################################

class EmotionDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


train_dataset = EmotionDataset(
    train_encodings,
    train_labels,
)

test_dataset = EmotionDataset(
    test_encodings,
    test_labels,
)

############################################################
# Load DistilBERT Model
############################################################

print("=" * 60)
print("Loading DistilBERT Model")
print("=" * 60)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

############################################################
# Training Arguments
############################################################

training_args = TrainingArguments(

    output_dir="models/bert",

    overwrite_output_dir=True,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_steps=100,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    report_to="none",

    seed=SEED,
)

############################################################
# Accuracy Metric
############################################################

def compute_metrics(pred):

    predictions = np.argmax(pred.predictions, axis=1)

    accuracy = accuracy_score(
        pred.label_ids,
        predictions,
    )

    return {
        "accuracy": accuracy
    }

############################################################
# Trainer
############################################################

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics,

)

############################################################
# Start Training
############################################################

print("=" * 60)
print("Training DistilBERT...")
print("=" * 60)

trainer.train()
import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

############################################################
# Reproducibility
############################################################

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

############################################################
# Load Dataset
############################################################

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_csv("data/processed/final_dataset.csv")

print(df.head())
print()

print("Dataset Shape:", df.shape)
print()

############################################################
# Encode Labels
############################################################

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["emotion"])

print("Emotion Classes:")

print(label_encoder.classes_)
print()

############################################################
# Train/Test Split
############################################################

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"],
    df["label"],
    test_size=0.20,
    random_state=SEED,
    stratify=df["label"],
)

print("Training Samples :", len(train_texts))
print("Testing Samples  :", len(test_texts))
print()

############################################################
# Tokenizer
############################################################

print("Loading DistilBERT Tokenizer...")

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=128,
)

test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=128,
)

print("Tokenization Completed")
print()
############################################################
# PyTorch Dataset
############################################################

class EmotionDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


train_dataset = EmotionDataset(
    train_encodings,
    train_labels,
)

test_dataset = EmotionDataset(
    test_encodings,
    test_labels,
)

############################################################
# Load DistilBERT Model
############################################################

print("=" * 60)
print("Loading DistilBERT Model")
print("=" * 60)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

############################################################
# Training Arguments
############################################################

training_args = TrainingArguments(

    output_dir="models/bert",

    overwrite_output_dir=True,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_steps=100,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    report_to="none",

    seed=SEED,
)

############################################################
# Accuracy Metric
############################################################

def compute_metrics(pred):

    predictions = np.argmax(pred.predictions, axis=1)

    accuracy = accuracy_score(
        pred.label_ids,
        predictions,
    )

    return {
        "accuracy": accuracy
    }

############################################################
# Trainer
############################################################

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics,

)

############################################################
# Start Training
############################################################

print("=" * 60)
print("Training DistilBERT...")
print("=" * 60)

trainer.train()

############################################################
# Evaluation
############################################################

print("=" * 60)
print("Evaluating DistilBERT...")
print("=" * 60)

predictions = trainer.predict(test_dataset)

y_pred = np.argmax(predictions.predictions, axis=1)

print()

print("Accuracy :", accuracy_score(test_labels, y_pred))

print()

print("Classification Report")

print(
    classification_report(
        test_labels,
        y_pred,
        target_names=label_encoder.classes_
    )
)

print()

print("Confusion Matrix")

print(confusion_matrix(test_labels, y_pred))

############################################################
# Save Model
############################################################

print()
print("=" * 60)
print("Saving DistilBERT Model...")
print("=" * 60)

os.makedirs("models/bert", exist_ok=True)

trainer.save_model("models/bert")

tokenizer.save_pretrained("models/bert")

import joblib

joblib.dump(
    label_encoder,
    "models/bert/label_encoder.pkl"
)

print()

print("Model Saved Successfully!")

print("Location : models/bert")

print("=" * 60)
