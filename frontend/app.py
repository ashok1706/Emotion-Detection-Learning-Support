import os
import sys

# ==========================================================
# ADD PROJECT ROOT
# ==========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==========================================================
# IMPORTS
# ==========================================================

import streamlit as st
import requests

from recommendation.recommender import get_recommendation
from analytics.dashboard import show_dashboard

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Emotion Detection Learning Support",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    padding-top:2rem;
}

.result-card{
    background:#F8F9FA;
    border-left:8px solid #1976D2;
    border-radius:12px;
    padding:20px;
    margin-bottom:20px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:50px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Navigation")

st.sidebar.success(
    "Emotion Detection Learning Support System"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Supported Emotions")

st.sidebar.write("😊 Joy")
st.sidebar.write("😢 Sadness")
st.sidebar.write("😠 Anger")
st.sidebar.write("😨 Fear")
st.sidebar.write("❤️ Love")
st.sidebar.write("😲 Surprise")

st.sidebar.markdown("---")

st.sidebar.info("""
**Model Used**

• Logistic Regression

**Dataset**

• Emotion NLP Dataset

• Google GoEmotions
""")

# ==========================================================
# HEADER
# ==========================================================

st.title("🎭 Emotion Detection Learning Support System")

st.markdown(
    """
    <h4 style='color:gray; margin-top:-10px;'>
    Artificial Intelligence Powered Emotion Recognition
    </h4>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# USER INPUT
# ==========================================================

text = st.text_area(
    "Enter your sentence",
    height=170,
    placeholder="Example: I feel very happy today."
)

col1, col2 = st.columns(2)

predict = col1.button(
    "Predict Emotion",
    use_container_width=True
)

clear = col2.button(
    "Clear",
    use_container_width=True
)

if clear:
    st.rerun()

# ==========================================================
# EMOJI MAP
# ==========================================================

emoji = {

    "joy":"😊",

    "sadness":"😢",

    "anger":"😠",

    "fear":"😨",

    "love":"❤️",

    "surprise":"😲"

}

# ==========================================================
# API
# ==========================================================

API_URL = "http://127.0.0.1:8000/predict"
# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    if text.strip() == "":

        st.warning("⚠️ Please enter some text.")

    else:

        with st.spinner("Analyzing emotion..."):

            try:

                response = requests.post(
                    API_URL,
                    json={"text": text},
                    timeout=30
                )

                if response.status_code == 200:

                    result = response.json()

                    primary_emotion = result["primary_emotion"].lower()
                    secondary_emotion = result["secondary_emotion"].lower()

                    primary_confidence = result["primary_confidence"]
                    secondary_confidence = result["secondary_confidence"]

                    mixed_emotion = result["mixed_emotion"]

                    confidence = result["confidence"]

                    recommendation = result["recommendation"]
                    motivation = result["motivation"]
                    emotion_insight = result["emotion_insight"]

                    icon = emoji.get(primary_emotion, "🙂")

                    st.success("Prediction Completed Successfully!")

                    st.markdown(
                        f"""
                        <div class="result-card">

                        <h2 style="text-align:center;">

                        {icon} {primary_emotion.upper()}

                        </h2>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.subheader("📝 Input Text")
                    st.info(text)

                    st.subheader("🎯 Primary Emotion")

                    st.success(
                        f"{icon} {primary_emotion.title()} "
                        f"({primary_confidence:.2f}%)"
                    )

                    st.subheader("🎭 Secondary Emotion")

                    st.info(
                        f"{emoji.get(secondary_emotion,'🙂')} "
                        f"{secondary_emotion.title()} "
                        f"({secondary_confidence:.2f}%)"
                    )

                    st.subheader("🧠 Mixed Emotion Detection")

                    if mixed_emotion:

                        st.warning(
                            f"Your text expresses a mixture of "
                            f"**{primary_emotion.title()}** and "
                            f"**{secondary_emotion.title()}** emotions."
                        )

                    else:

                        st.success(
                            "A single dominant emotion was detected."
                        )

                    st.subheader("📊 Confidence Scores")

                    for emotion_name, score in confidence.items():

                        st.write(
                            f"**{emotion_name.title()}** "
                            f"({score:.2f}%)"
                        )

                        st.progress(min(score / 100, 1.0))

                    st.subheader("📚 Learning Recommendations")

                    if isinstance(recommendation, list):

                        for tip in recommendation:
                            st.write("✅", tip)

                    else:

                        st.write(recommendation)

                    st.subheader("💡 Motivation")

                    st.info(motivation)

                    st.subheader("🧠 Emotion Insight")

                    st.success(emotion_insight)

                else:

                    st.error("Prediction failed.")

            except requests.exceptions.ConnectionError:

                st.error("❌ FastAPI server is not running.")

            except Exception as e:

                st.error("Unexpected Error")

                st.exception(e)
                # ==========================================================
# PREDICTION HISTORY
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if predict and text.strip() != "":

    try:

        response = requests.post(
            API_URL,
            json={"text": text},
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            emotion = result["primary_emotion"]

            st.session_state.history.insert(
                0,
                {
                    "text": text,
                    "emotion": emotion
                }
            )

            if len(st.session_state.history) > 10:

                st.session_state.history = (
                    st.session_state.history[:10]
                )

    except Exception:
        pass


# ==========================================================
# SIDEBAR HISTORY
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.subheader("Recent Predictions")

if len(st.session_state.history) == 0:

    st.sidebar.write("No predictions yet.")

else:

    for item in st.session_state.history:

        icon = emoji.get(item["emotion"], "🙂")

        st.sidebar.markdown(
            f"""
**{icon} {item['emotion'].title()}**

{item['text'][:45]}...

---
"""
        )


# ==========================================================
# PROJECT INFORMATION
# ==========================================================

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    st.subheader("✨ Project Features")

    st.markdown("""
- 🎭 Emotion Detection
- 🤖 Machine Learning Prediction
- 🧠 Mixed Emotion Recognition
- 📚 Learning Recommendations
- ⚡ FastAPI Backend
- 🎨 Streamlit Frontend
- 📊 Interactive Analytics Dashboard
""")

with col2:

    st.subheader("🛠 Technology Stack")

    st.markdown("""
- Python
- Streamlit
- FastAPI
- Scikit-Learn
- TensorFlow
- Pandas
- Plotly
""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div class='footer'>

<h3>Emotion Detection Learning Support System</h3>

<p>
AI-Powered Learning Support using Emotion Recognition
</p>

<p>
Developed for the <b>Skill Wallet Project</b>
</p>

<p>
Built using ❤️ Python, FastAPI, Streamlit & Machine Learning
</p>

</div>
""",
unsafe_allow_html=True
)

# ==========================================================
# ANALYTICS
# ==========================================================

st.markdown("---")

show_dashboard(st.session_state.history)