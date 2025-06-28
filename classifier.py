import numpy as np
from transformers import DistilBertTokenizerFast
import onnxruntime

class ToxicityClassifier:
    def __init__(self, model_path: str, categories, tokenizer_path: str):
        self.session = onnxruntime.InferenceSession(model_path)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
        self.categories = categories

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        encodings = self.tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        input_feed = {
            "input_ids": encodings["input_ids"].astype(np.int64),
            "attention_mask": encodings["attention_mask"].astype(np.int64)
        }
        logits = self.session.run(None, input_feed)[0]
        probs = 1 / (1 + np.exp(-logits))
        results = [{cat: float(probs[i][j]) for j, cat in enumerate(self.categories)} for i in range(len(probs))]

        if len(results) == 1:
            return results[0]
        return results
