Toxicity & Abuse Detection Microservice
An AI-powered FastAPI microservice for detecting toxic, abusive, bullying, or inappropriate language in user-generated content.
The service runs fully offline, leverages a DistilBERT model in ONNX format, and is Dockerized for easy deployment.

Features
Predicts toxicity scores and flags harmful content.

Returns structured JSON with:

Toxicity Score

Action (approved, flagged, blocked)

Reasons (categories like insult, harassment, etc.)

Uses DistilBERT + ONNX Runtime for fast inference.

Configurable via config.json.

REST API built with FastAPI.

Exposes the following endpoints:

POST /analyze-text

GET /health

GET /version

Project Structure
graphql
Copy
Edit
.
├── classifier.py           # Core inference logic using ONNX model
├── main.py                 # FastAPI app with REST endpoints
├── toxicity_distilbert.onnx# Pretrained DistilBERT ONNX model
├── tokenizer/              # Tokenizer files from HuggingFace (saved locally)
├── config.json             # Configurable thresholds and enabled categories
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container setup
└── README.md               # Documentation
Installation (Local Setup)
Clone the repository:

bash
Copy
Edit
git clone <your-repo-url>
cd toxicity-detector
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the application:

bash
Copy
Edit
uvicorn main:app --host 0.0.0.0 --port 8000
Docker Deployment
Build the Docker image:

bash
Copy
Edit
docker build -t toxicity-detector .
Run the Docker container:

bash
Copy
Edit
docker run -p 8000:8000 toxicity-detector
API Endpoints
POST /analyze-text
Analyzes input text for toxicity and returns structured results.

Request Example:

json
Copy
Edit
{
  "user_id": "stu_5044",
  "post_id": "post_0988",
  "text": "You are so dumb and your post is pathetic"
}
Response Example:

json
Copy
Edit
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
Returns service health status.

Response:

json
Copy
Edit
{ "status": "ok" }
GET /version
Returns current model version.

Response:

json
Copy
Edit
{ "model_version": "1.0.0" }
Configuration (config.json)
json
Copy
Edit
{
  "toxicity_threshold": 0.75,
  "flag_threshold": 0.5,
  "enabled_categories": ["toxic", "insult", "harassment"]
}
toxicity_threshold: Score above which content is marked as toxic.

flag_threshold: Score range for flagged (potentially harmful) content.

enabled_categories: Categories to evaluate in the prediction.

Testing Scenarios
Text Example	Expected Score	Label	Action
"You're a failure and a disgrace."	> 0.80	toxic	blocked
"This is kind of dumb tbh"	~0.60	flagged	flagged
"Great job on your article!"	< 0.30	safe	approved
"बकवास पोस्ट है" (in Hindi)	~0.50	flagged	flagged

Requirements
fastapi

uvicorn

numpy

transformers

onnxruntime

Make sure the tokenizer is saved locally:

python
Copy
Edit
from transformers import DistilBertTokenizerFast
DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased").save_pretrained("./tokenizer")
Notes
Offline-Only: Fully self-contained, no external API calls.

Performance: Inference time < 1 second per text input.

No UI: JSON-based API only.

