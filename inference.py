from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Pipeline 1：三分类情感分析
SENT_MODEL_NAME = "ChenRuixiao/mcdonalds-roberta-sentiment"

# Pipeline 2：五分类星级预测
STAR_MODEL_NAME = "ChenRuixiao/bert_base_mcd_star_model"

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载情感分析模型
sent_tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL_NAME)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_NAME)
sent_model.to(device)
sent_model.eval()
id2label_sent = {0: "negative", 1: "neutral", 2: "positive"}

# 加载星级预测模型
star_tokenizer = AutoTokenizer.from_pretrained(STAR_MODEL_NAME)
star_model = AutoModelForSequenceClassification.from_pretrained(STAR_MODEL_NAME)
star_model.to(device)
star_model.eval()
id2label_star = {0: "1_star", 1: "2_star", 2: "3_star", 3: "4_star", 4: "5_star"}
star_suggestions = {
    "1_star": "Very dissatisfied. Consider reviewing service and food quality.",
    "2_star": "Dissatisfied. Improvements in service or food are needed.",
    "3_star": "Neutral. Customer experience was average.",
    "4_star": "Satisfied. Good experience, small improvements possible.",
    "5_star": "Very satisfied. Excellent service and food."
}

# 文本情感分析函数
def predict_sentiment(text: str):
    inputs = sent_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sent_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze()
        pred_id = torch.argmax(probs).item()
        pred_label = id2label_sent[pred_id]

    return {
        "predicted_label": pred_label,
        "probabilities": {k: round(v.item(),4) for k,v in zip(id2label_sent.values(), probs)},
        "device": str(device)
    }

# 星级预测函数
def predict_star(text: str):
    inputs = star_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = star_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze()
        pred_id = torch.argmax(probs).item()
        pred_label = id2label_star[pred_id]

    return {
        "predicted_star": pred_label,
        "suggestion": star_suggestions[pred_label],
        "probabilities": {k: round(v.item(),4) for k,v in zip(id2label_star.values(), probs)},
        "device": str(device)
    }
