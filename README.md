# McDonald's Review Sentiment Analysis App

This project is Pipeline 1 of our ISOM5240 project.

We fine-tuned a RoBERTa model on McDonald's customer reviews to classify sentiment into three categories:

- negative
- neutral
- positive

## Final Model
We compared multiple fine-tuned transformer models and selected RoBERTa-base as the final model because it achieved the best trade-off between accuracy and runtime.

## Files
- `app.py` - Streamlit web app
- `inference.py` - model inference logic
- `requirements.txt` - dependencies for deployment
- `README.md` - project description

## Model Source
The final fine-tuned model is hosted on Hugging Face Hub:
`ChenRuixiao/mcdonalds-roberta-sentiment`

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
