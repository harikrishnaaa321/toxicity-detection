from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import json
import numpy as np
from classifier import ToxicityClassifier
from langdetect import detect, LangDetectException

with open("config.json") as f:
    config = json.load(f)

app = FastAPI(title="Toxicity & Abuse Detection", version="1.0.0")

categories = ["toxic", "harassment", "obscene", "threat", "insult", "identity_hate"]


model = ToxicityClassifier(
    "toxicity_distilbert.onnx", 
    categories, 
    tokenizer_path="./tokenizer"
)

class AnalyzeRequest(BaseModel):
    user_id: str
    post_id: str
    text: str

class AnalyzeResponse(BaseModel):
    user_id: str
    post_id: str
    toxicity_score: float
    label: str
    action: str
    reasons: List[str]
    threshold: float

def decide_label_action(
    scores: dict, 
    enabled_categories: List[str], 
    toxicity_threshold: float, 
    flag_threshold: float
):
    filtered = {k: v for k, v in scores.items() if k in enabled_categories}
    max_score = max(filtered.values()) if filtered else 0.0

    if max_score >= toxicity_threshold:
        label = "toxic"
        action = "blocked"
    elif max_score >= flag_threshold:
        label = "flagged"
        action = "flagged"
    else:
        label = "safe"
        action = "approved"

    reasons = [cat for cat, score in filtered.items() if score >= flag_threshold]
    return max_score, label, action, reasons

def preprocess_text(text: str) -> str:
    import re
    import emoji

    # Convert emojis to text
    text = emoji.demojize(text)

    # Lowercase
    text = text.lower()

    # Clean special characters (retain Hindi letters)
    text = re.sub(r'[^a-zA-Z0-9\s\u0900-\u097F]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


@app.post("/analyze-text", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    try:
        # Step 1: Language Detection
        try:
            lang = detect(payload.text)
        except LangDetectException:
            lang = "unknown"

        if lang != "en":
            # Step 2: Handle non-English
            return AnalyzeResponse(
                user_id=payload.user_id,
                post_id=payload.post_id,
                toxicity_score=0.0,
                label="flagged",
                action="flagged",
                reasons=["non-English content"],
                threshold=config.get("toxicity_threshold", 0.75)
            )

        # Step 3: Preprocess + Prediction
        scores = model.predict(payload.text)
        max_score, label, action, reasons = decide_label_action(
            scores,
            config.get("enabled_categories", categories),
            config.get("toxicity_threshold", 0.75),
            config.get("flag_threshold", 0.5)
        )

        return AnalyzeResponse(
            user_id=payload.user_id,
            post_id=payload.post_id,
            toxicity_score=round(max_score, 2),
            label=label,
            action=action,
            reasons=reasons,
            threshold=config.get("toxicity_threshold", 0.75)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"model_version": "1.0.0"}
