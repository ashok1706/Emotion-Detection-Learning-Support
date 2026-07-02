import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analytics Dashboard")

st.markdown("---")

history_file = "history/prediction_history.csv"

try:
    df = pd.read_csv(history_file)

    total_predictions = len(df)
    most_common = df["Predicted Emotion"].mode()[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("📌 Total Predictions", total_predictions)

    with col2:
        st.metric("😊 Most Common Emotion", most_common.capitalize())

    st.markdown("---")

    emotion_counts = (
        df["Predicted Emotion"]
        .value_counts()
        .reset_index()
    )

    emotion_counts.columns = ["Emotion", "Count"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Bar Chart")
        fig_bar = px.bar(
            emotion_counts,
            x="Emotion",
            y="Count",
            color="Emotion",
            text="Count"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("🥧 Pie Chart")
        fig_pie = px.pie(
            emotion_counts,
            names="Emotion",
            values="Count",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    st.subheader("📋 Prediction History")

    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="⬇ Download Prediction History",
        data=df.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv"
    )

except FileNotFoundError:
    st.warning("Prediction history file not found.")

except pd.errors.EmptyDataError:
    st.warning("Prediction history is empty.")