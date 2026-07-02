from transformers import pipeline

# Load a pre-trained emotion detection model
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def predict_emotion(text):
    """
    Predict emotions from the given text.
    """
    results = emotion_classifier(text)

    # Find emotion with highest score
    best_emotion = max(results[0], key=lambda x: x["score"])

    return {
        "emotion": best_emotion["label"],
        "confidence": round(best_emotion["score"] * 100, 2),
        "all_scores": results[0]
    }