import streamlit as st
from inference import predict_sentiment, predict_star

st.set_page_config(
    page_title="McDonald's Review Analysis",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 McDonald's Review Analyzer")
st.write("Enter a customer review and choose an analysis type:")

user_input = st.text_area(
    "Enter a review:",
    height=160,
    placeholder="Type your review here..."
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            result = predict_sentiment(user_input)
            st.subheader("Sentiment Analysis Result")
            pred = result["predicted_label"]
            if pred == "negative":
                st.error(pred)
            elif pred == "neutral":
                st.warning(pred)
            else:
                st.success(pred)

            st.subheader("Class Probabilities")
            for k,v in result["probabilities"].items():
                st.write(f"{k}: {v}")
                st.progress(v)
            st.caption(f"Running on device: {result['device']}")
        else:
            st.warning("Please enter a review first.")

with col2:
    if st.button("Predict Star Rating"):
        if user_input.strip():
            result = predict_star(user_input)
            st.subheader("Star Rating Result")
            st.success(f"{result['predicted_star']} - {result['suggestion']}")

            st.subheader("Class Probabilities")
            for k,v in result["probabilities"].items():
                st.write(f"{k}: {v}")
                st.progress(v)
            st.caption(f"Running on device: {result['device']}")
        else:
            st.warning("Please enter a review first.")

st.markdown("---")
st.caption("Pipeline 1 & 2 - Sentiment analysis and star rating prediction for McDonald's reviews")
