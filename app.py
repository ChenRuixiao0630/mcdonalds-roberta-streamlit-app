import streamlit as st
from inference import predict_sentiment

st.set_page_config(
    page_title="McDonald's Review Sentiment Analysis",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 McDonald's Review Sentiment Analysis")
st.write(
    "Enter a customer review and the fine-tuned RoBERTa model will predict "
    "whether the sentiment is negative, neutral, or positive."
)

st.markdown("### Try an example")
example_reviews = {
    "Negative example": "The fries were cold and the service was very slow.",
    "Neutral example": "The food was okay, nothing special.",
    "Positive example": "Great burger and very friendly staff."
}

selected_example = st.selectbox(
    "Choose an example review or type your own below:",
    options=["None"] + list(example_reviews.keys())
)

default_text = ""
if selected_example != "None":
    default_text = example_reviews[selected_example]

user_input = st.text_area(
    "Enter a review:",
    value=default_text,
    height=160,
    placeholder="Example: The fries were cold and the service was very slow."
)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        result = predict_sentiment(user_input)

        st.subheader("Prediction")
        predicted_label = result["predicted_label"]

        if predicted_label == "negative":
            st.error(f"Predicted sentiment: {predicted_label}")
        elif predicted_label == "neutral":
            st.warning(f"Predicted sentiment: {predicted_label}")
        else:
            st.success(f"Predicted sentiment: {predicted_label}")

        st.subheader("Class Probabilities")
        probs = result["probabilities"]

        st.write(f"**Negative:** {probs['negative']}")
        st.progress(float(probs["negative"]))

        st.write(f"**Neutral:** {probs['neutral']}")
        st.progress(float(probs["neutral"]))

        st.write(f"**Positive:** {probs['positive']}")
        st.progress(float(probs["positive"]))

        st.caption(f"Running on device: {result['device']}")
    else:
        st.warning("Please enter a review first.")

st.markdown("---")
st.caption("Pipeline 1 - Fine-tuned RoBERTa model for McDonald's review sentiment analysis")
