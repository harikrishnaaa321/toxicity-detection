🛡️ Toxicity & Abuse Detection Microservice

An AI-powered FastAPI microservice to detect toxic, abusive, bullying, or inappropriate language in user-generated posts. The service runs fully offline, leverages a DistilBERT model in ONNX format, and is Dockerized for easy deployment.

🚀 Features

Predicts toxicity scores and flags harmful content.

Returns structured JSON with:

Toxicity Score

Action (approved, flagged, blocked)

Reasons (categories like insult, harassment, etc.)

Uses DistilBERT + ONNX Runtime for fast inference.

Configurable via config.json.

REST API built with FastAPI.

Exposes /analyze-text, /health, and /version endpoints.

📁 Project Structure

.
├── classifier.py           # Core inference logic using ONNX model
├── main.py                 # FastAPI app with REST endpoints
├── toxicity_distilbert.onnx# Pretrained DistilBERT ONNX model
├── tokenizer/              # Tokenizer files from HuggingFace (saved locally)
├── config.json             # Configurable thresholds and enabled categories
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container setup
└── README.md               # Documentation

🔧 Installation (Locally)

Clone this repo:

git clone <your-repo-url>
cd toxicity-detector

Install dependencies:

pip install -r requirements.txt

Run the app:

uvicorn main:app --host 0.0.0.0 --port 8000

🐳 Docker Deployment

Build the Docker image:

docker build -t toxicity-detector .

Run the container:

docker run -p 8000:8000 toxicity-detector

🔌 API Endpoints

POST /analyze-text

Analyze text for toxicity and get structured results.

Request:

{
  "user_id": "stu_5044",
  "post_id": "post_0988",
  "text": "You are so dumb and your post is pathetic"
}

Response:

{
  "user_id": "stu_5044",
  "post_id": "post_0988",
  "toxicity_score": 0.86,
  "label": "toxic",
  "action": "blocked",
  "reasons": ["insult", "harassment"],
  "threshold": 0.75
}

GET /health

Check service health.

{ "status": "ok" }

GET /version

Get current model version.

{ "model_version": "1.0.0" }

⚙️ Config (config.json)

{
  "toxicity_threshold": 0.75,
  "flag_threshold": 0.5,
  "enabled_categories": ["toxic", "insult", "harassment"]
}

toxicity_threshold: Score above which a post is marked toxic.

flag_threshold: Score range for flagged content.

enabled_categories: Categories to evaluate in prediction.

🧪 Testing Scenarios

Text Example

Expected Score

Label

Action

"You're a failure and a disgrace."

> 0.80

toxic

blocked

"This is kind of dumb tbh"

~0.60

flagged

flagged

"Great job on your article!"

< 0.30

safe

approved

"बकवास पोस्ट है" (in Hindi)

~0.50

flagged

flagged

📆 Requirements

fastapi
uvicorn
numpy
transformers
onnxruntime

Make sure your tokenizer is saved locally (tokenizer/) using:

from transformers import DistilBertTokenizerFast
DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased").save_pretrained("./tokenizer")

📌 Notes

Offline-Only: No external API calls; uses ONNX model and local tokenizer.

Performance: Inference time < 1s per text input.

No UI: JSON-based API only.

📜 License

MIT License (or specify yours)

