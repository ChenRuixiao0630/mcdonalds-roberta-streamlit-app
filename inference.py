import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Pipeline 1：三分类情感分析
SENT_MODEL_NAME = "ChenRuixiao/mcdonalds-roberta-sentiment"

# Pipeline 2：五分类星级预测
STAR_MODEL_NAME = "ChenRuixiao/bert_base_mcd_star_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id2label_sent = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

id2label_star = {
    0: "1_star",
    1: "2_star",
    2: "3_star",
    3: "4_star",
    4: "5_star"
}

star_suggestions = {
    "1_star": "Very dissatisfied. Consider reviewing service and food quality.",
    "2_star": "Dissatisfied. Improvements in service or food are needed.",
    "3_star": "Neutral. Customer experience was average.",
    "4_star": "Satisfied. Good experience, with some room for improvement.",
    "5_star": "Very satisfied. Excellent service and food."
}

@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_star_model():
    tokenizer = AutoTokenizer.from_pretrained(STAR_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(STAR_MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model

def predict_sentiment(text: str):
    tokenizer, model = load_sentiment_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze()
        pred_id = torch.argmax(probs).item()

    return {
        "predicted_label": id2label_sent[pred_id],
        "probabilities": {
            "negative": round(probs[0].item(), 4),
            "neutral": round(probs[1].item(), 4),
            "positive": round(probs[2].item(), 4)
        },
        "device": str(device)
    }

def predict_star(text: str):
    tokenizer, model = load_star_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze()
        pred_id = torch.argmax(probs).item()
        pred_label = id2label_star[pred_id]

    return {
        "predicted_star": pred_label,
        "suggestion": star_suggestions[pred_label],
        "probabilities": {
            "1_star": round(probs[0].item(), 4),
            "2_star": round(probs[1].item(), 4),
            "3_star": round(probs[2].item(), 4),
            "4_star": round(probs[3].item(), 4),
            "5_star": round(probs[4].item(), 4)
        },
        "device": str(device)
    }
