import streamlit as st

st.set_page_config(
    page_title="About Project",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About the Project")

st.markdown("---")

st.header("🎓 Emotion Detection & Learning Support Engine")

st.write("""
The Emotion Detection & Learning Support Engine is a Machine Learning application
that detects emotions from user-entered text and provides personalized learning
support recommendations.

This project demonstrates how Natural Language Processing (NLP) and Machine
Learning can be applied to understand human emotions and improve the learning
experience.
""")

st.markdown("---")

st.header("🎯 Project Objectives")

st.markdown("""
- Detect emotions from text
- Help learners based on their emotional state
- Apply Natural Language Processing (NLP)
- Build an interactive Streamlit application
- Visualize prediction history through analytics
""")

st.markdown("---")

st.header("🛠 Technologies Used")

st.markdown("""
- Python
- Scikit-learn
- Streamlit
- Pandas
- NumPy
- Joblib
""")

st.markdown("---")

st.header("🤖 Machine Learning Model")

st.success("""
Model : Logistic Regression

Accuracy : 86.9%
""")

st.markdown("---")

st.header("😊 Supported Emotions")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("😊 Joy")
    st.info("😢 Sadness")

with col2:
    st.info("😠 Anger")
    st.info("😨 Fear")

with col3:
    st.info("❤️ Love")
    st.info("😲 Surprise")

st.markdown("---")

st.header("✨ Features")

st.markdown("""
✅ Emotion Detection

✅ Confidence Scores

✅ Learning Support Recommendation

✅ Prediction History

✅ Analytics Dashboard

✅ CSV Export
""")

st.markdown("---")

st.success("Thank you for using the Emotion Detection & Learning Support Engine!")