"""
============================================================
Emotion Detection Learning Support System
Recommendation Engine
============================================================
"""

recommendations = {

    "joy": {
        "title": "Keep Up the Great Work! 😊",
        "tips": [
            "Maintain your positive mindset.",
            "Challenge yourself with advanced problems.",
            "Explore new topics to improve your skills.",
            "Help your classmates understand concepts.",
            "Keep practicing consistently."
        ]
    },

    "sadness": {
        "title": "Stay Positive 💙",
        "tips": [
            "Take a short break before continuing.",
            "Study one topic at a time.",
            "Listen to relaxing music.",
            "Reward yourself after completing small goals.",
            "Remember that every expert was once a beginner."
        ]
    },

    "anger": {
        "title": "Relax and Refocus 😌",
        "tips": [
            "Take deep breaths.",
            "Walk for 5–10 minutes.",
            "Drink some water.",
            "Resume studying after calming down.",
            "Avoid making decisions while frustrated."
        ]
    },

    "fear": {
        "title": "Build Confidence 💪",
        "tips": [
            "Practice step by step.",
            "Start with easier questions.",
            "Don't fear mistakes—they help you learn.",
            "Believe in your abilities.",
            "Focus on progress instead of perfection."
        ]
    },

    "love": {
        "title": "Share Your Knowledge ❤️",
        "tips": [
            "Study with friends.",
            "Teach someone what you've learned.",
            "Keep your curiosity alive.",
            "Enjoy the learning journey.",
            "Stay motivated and consistent."
        ]
    },

    "surprise": {
        "title": "Stay Curious 😲",
        "tips": [
            "Read more about the topic.",
            "Explore related concepts.",
            "Take notes of new discoveries.",
            "Ask questions whenever you're curious.",
            "Use your curiosity to improve learning."
        ]
    }

}


def get_recommendation(emotion):
    """
    Returns recommendations based on predicted emotion.
    """

    emotion = emotion.lower()

    if emotion in recommendations:
        return recommendations[emotion]

    return {
        "title": "Keep Learning 📚",
        "tips": [
            "Stay motivated.",
            "Practice every day.",
            "Learning never stops."
        ]
    }