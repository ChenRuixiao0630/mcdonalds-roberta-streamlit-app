# McDonald's Review Analysis App

This Streamlit app implements both Pipeline 1 and Pipeline 2 of our ISOM5240 project.

## Pipelines

1. **Sentiment Analysis (Pipeline 1)**  
   - Model: ChenRuixiao/mcdonalds-roberta-sentiment  
   - Output: Negative / Neutral / Positive sentiment

2. **Star Rating Prediction (Pipeline 2)**  
   - Model: ChenRuixiao/bert_base_mcd_star_model  
   - Output: 1–5 star rating with suggestions

## Files

- `app.py` - Streamlit web app  
- `inference.py` - Model inference logic for both pipelines  
- `requirements.txt` - Python dependencies  
- `README.md` - Project description

## Usage

```bash
pip install -r requirements.txt
streamlit run app.py
