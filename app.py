import streamlit as st
from inference import predict_sentiment, predict_star

st.set_page_config(
    page_title="McDonald's Review Analyzer",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 McDonald's Review Analyzer")
st.write(
    "Enter a customer review and run either sentiment analysis "
    "or star rating prediction with suggestion."
)

example_reviews = {
    "None": "",
    "Negative example": "The fries were cold and the service was very slow.",
    "Neutral example": "The food was okay, nothing special.",
    "Positive example": "Great burger and very friendly staff."
}

selected_example = st.selectbox(
    "Choose an example review or type your own below:",
    options=list(example_reviews.keys())
)

default_text = example_reviews[selected_example]

user_input = st.text_area(
    "Enter a review:",
    value=default_text,
    height=160,
    placeholder="Type your review here..."
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            result = predict_sentiment(user_input)

            st.subheader("Sentiment Result")
            pred = result["predicted_label"]

            if pred == "negative":
                st.error(f"Predicted sentiment: {pred}")
            elif pred == "neutral":
                st.warning(f"Predicted sentiment: {pred}")
            else:
                st.success(f"Predicted sentiment: {pred}")

            st.subheader("Sentiment Probabilities")
            for label, prob in result["probabilities"].items():
                st.write(f"{label}: {prob}")
                st.progress(float(prob))

            st.caption(f"Running on device: {result['device']}")
        else:
            st.warning("Please enter a review first.")

with col2:
    if st.button("Predict Star Rating"):
        if user_input.strip():
            result = predict_star(user_input)

            st.subheader("Star Rating Result")
            st.info(f"Predicted rating: {result['predicted_star']}")
            st.write(f"**Suggestion:** {result['suggestion']}")

            st.subheader("Star Probabilities")
            for label, prob in result["probabilities"].items():
                st.write(f"{label}: {prob}")
                st.progress(float(prob))

            st.caption(f"Running on device: {result['device']}")
        else:
            st.warning("Please enter a review first.")

st.markdown("---")
st.caption("Pipeline 1 + Pipeline 2 for McDonald's review analysis")
