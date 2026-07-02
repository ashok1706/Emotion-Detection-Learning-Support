import re
import os
import joblib
import pandas as pd
from datetime import datetime

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================
MODEL_PATH = "models/logistic/emotion_model.pkl"
VECTORIZER_PATH = "models/logistic/tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = "models/logistic/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)


# ==========================================================
# TEXT PREPROCESSING
# ==========================================================

def clean_text(text):
    """
    Clean user input before prediction.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove punctuation & numbers
    text = re.sub(r"[^a-zA-Z ]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================================
# EMOTION PREDICTION
# ==========================================================

def predict_emotion(text):
    """
    Predict emotion from user text.

    Returns:
        primary_emotion
        secondary_emotion
        mixed_emotion
        confidence_scores
    """

    cleaned_text = clean_text(text)

    features = vectorizer.transform([cleaned_text])

    probabilities = model.predict_proba(features)[0]

    classes = label_encoder.classes_

    # Convert probabilities into percentages
    confidence = {
        emotion: round(score * 100, 2)
        for emotion, score in zip(classes, probabilities)
    }

    # Sort emotions by confidence
    sorted_emotions = sorted(
        confidence.items(),
        key=lambda x: x[1],
        reverse=True
    )

    primary_emotion = sorted_emotions[0][0]
    primary_score = sorted_emotions[0][1]

    secondary_emotion = sorted_emotions[1][0]
    secondary_score = sorted_emotions[1][1]

    # Mixed emotion detection
    mixed_emotion = False

    if secondary_score >= 35:
        mixed_emotion = True

    result = {
        "primary_emotion": primary_emotion,
        "primary_confidence": primary_score,
        "secondary_emotion": secondary_emotion,
        "secondary_confidence": secondary_score,
        "mixed_emotion": mixed_emotion,
        "confidence": confidence
    }

    return result


# ==========================================================
# LEARNING RECOMMENDATIONS
# ==========================================================
def get_learning_recommendation(emotion):
    """
    Returns personalized learning recommendation
    based on predicted emotion.
    """

    recommendations = {

        "joy": [
            "😊 Keep your positive mindset.",
            "Challenge yourself with advanced problems.",
            "Explore new topics to improve your skills.",
            "Help your classmates understand concepts.",
            "Keep practicing consistently."
        ],

        "sadness": [
            "💙 Take a short break before continuing.",
            "Study one topic at a time.",
            "Listen to relaxing music.",
            "Reward yourself after completing small goals.",
            "Remember that every expert was once a beginner."
        ],

        "anger": [
            "😌 Take a few deep breaths.",
            "Relax before studying again.",
            "Break difficult tasks into smaller steps.",
            "Avoid rushing through lessons.",
            "Focus on steady progress."
        ],

        "fear": [
            "💪 Practice step by step.",
            "Start with easier questions.",
            "Don't fear mistakes—they help you learn.",
            "Believe in your abilities.",
            "Focus on progress instead of perfection."
        ],

        "love": [
            "❤️ Study with friends.",
            "Teach someone what you've learned.",
            "Keep your curiosity alive.",
            "Enjoy the learning journey.",
            "Stay motivated and consistent."
        ],

        "surprise": [
            "😲 Review today's lesson once more.",
            "Write short notes.",
            "Practice similar problems.",
            "Strengthen your understanding.",
            "Keep exploring new concepts."
        ]
    }

    return recommendations.get(
        emotion,
        ["📚 Keep learning every day."]
    )


# ==========================================================
# EMOTION INSIGHT
# ==========================================================

def get_emotion_insight(primary_emotion, secondary_emotion, mixed_emotion):
    """
    Generate a simple emotion insight.
    """

    if mixed_emotion:

        return (
            f"Your message mainly expresses "
            f"{primary_emotion.title()} while also showing "
            f"signs of {secondary_emotion.title()}. "
            f"It is completely normal to experience multiple emotions at the same time."
        )

    insights = {

        "joy":
        "You seem happy and positive. Keep enjoying your learning journey.",

        "sadness":
        "You may be feeling low. Take a short break and continue with smaller goals.",

        "anger":
        "You appear frustrated. Staying calm can help improve your focus.",

        "fear":
        "You might be anxious. Believe in yourself and practice one step at a time.",

        "love":
        "You seem motivated and caring. Keep supporting your learning community.",

        "surprise":
        "Unexpected situations can become valuable learning opportunities."
    }

    return insights.get(
        primary_emotion,
        "Keep learning with confidence."
    )


# ==========================================================
# MOTIVATION
# ==========================================================

def get_motivation(emotion):

    quotes = {

        "joy":
        "Success comes from consistent effort.",

        "sadness":
        "Every small step forward is progress.",

        "anger":
        "Stay calm. Great things take time.",

        "fear":
        "Believe in yourself. You are capable of amazing things.",

        "love":
        "Learning becomes meaningful when shared with others.",

        "surprise":
        "Every surprise is a chance to learn something new."
    }

    return quotes.get(
        emotion,
        "Never stop learning."
    )


# ==========================================================
# SAVE PREDICTION HISTORY
# ==========================================================
def save_prediction(text, emotion):

    history_folder = "history"

    if not os.path.exists(history_folder):
        os.makedirs(history_folder)

    history_file = os.path.join(
        history_folder,
        "prediction_history.csv"
    )

    new_data = pd.DataFrame({

        "Timestamp": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ],

        "Input Text": [
            text
        ],

        "Predicted Emotion": [
            emotion
        ]
    })

    if os.path.exists(history_file):

        try:

            old_data = pd.read_csv(history_file)

            all_data = pd.concat(
                [old_data, new_data],
                ignore_index=True
            )

        except Exception:

            all_data = new_data

    else:

        all_data = new_data

    all_data.to_csv(
        history_file,
        index=False
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    sample = "I am excited but nervous about my first interview."

    result = predict_emotion(sample)

    print("=" * 50)
    print("EMOTION ANALYSIS")
    print("=" * 50)

    print("\nPrimary Emotion :",
          result["primary_emotion"])

    print("Confidence      :",
          str(result["primary_confidence"]) + "%")

    print("\nSecondary Emotion :",
          result["secondary_emotion"])

    print("Confidence        :",
          str(result["secondary_confidence"]) + "%")

    print("\nMixed Emotion :",
          result["mixed_emotion"])

    print("\nConfidence Scores")

    for emotion, score in result["confidence"].items():

        print(
            emotion.title(),
            ":",
            str(score) + "%"
        )

    print("\nLearning Recommendation")

    recommendation = get_learning_recommendation(
        result["primary_emotion"]
    )

    for item in recommendation:

        print("-", item)

    print("\nEmotion Insight")

    print(
        get_emotion_insight(
            result["primary_emotion"],
            result["secondary_emotion"],
            result["mixed_emotion"]
        )
    )

    print("\nMotivation")

    print(
        get_motivation(
            result["primary_emotion"]
        )
    )

    save_prediction(
        sample,
        result["primary_emotion"]
    )

    print("\nPrediction Saved Successfully.")