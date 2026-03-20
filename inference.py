from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = "ChenRuixiao/mcdonalds-roberta-sentiment"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

def predict_sentiment(text: str):
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
        "predicted_label": id2label[pred_id],
        "probabilities": {
            "negative": round(probs[0].item(), 4),
            "neutral": round(probs[1].item(), 4),
            "positive": round(probs[2].item(), 4)
        },
        "device": str(device)
    }
