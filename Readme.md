# Toxicity & Abuse Detection Microservice

An **AI-powered FastAPI microservice** for detecting toxic, abusive, bullying, or inappropriate language in user-generated content.
The service runs **fully offline**, leverages a **DistilBERT model in ONNX format**, and is **Dockerized** for seamless deployment.

---

## Features

* Predicts toxicity scores and flags harmful content.
* Returns structured JSON with:

  * **Toxicity Score**
  * **Action** (approved, flagged, blocked)
  * **Reasons** (categories like insult, harassment, etc.)
* Uses **DistilBERT + ONNX Runtime** for fast, efficient inference.
* Configurable via `config.json`.
* REST API built with **FastAPI**.
* Exposes:

  * `POST /analyze-text`
  * `GET /health`
  * `GET /version`

---

## Project Structure

```text
.
├── classifier.py            # Core inference logic using ONNX model
├── main.py                  # FastAPI app with REST endpoints
├── toxicity_distilbert.onnx # Pretrained DistilBERT ONNX model
├── tokenizer/               # Tokenizer files from HuggingFace (saved locally)
├── config.json              # Configurable thresholds and enabled categories
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker container setup
└── README.md                # Documentation
```

---

## Compulsory Installation

1. **Clone the Repository**

   ```bash
   git clone <https://github.com/harikrishnaaa321/toxicity-detection.git>
   cd toxicity-detector
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Tokenizer Setup (Must Do)** You must save the tokenizer files locally:

   ```python
   from transformers import DistilBertTokenizerFast
   DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased").save_pretrained("./tokenizer")
   ```

4. **Run the Application**

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

---

## Docker Deployment

1. **Build the Docker Image**

   ```bash
   docker build -t toxicity-detector .
   ```

2. **Run the Docker Container**

   ```bash
   docker run -p 8000:8000 toxicity-detector
   ```

---

## Model Details

* **Model:** DistilBERT (ONNX format)
* **Tokenizer:** DistilBERT Tokenizer (saved locally)
* **ONNX Runtime:** Used for fast, offline inference
* **Inference Time:** Less than 1 second per input
* **Categories:** toxic, insult, harassment (configurable)

---

## API Endpoints

### POST `/analyze-text`

Analyzes input text for toxicity.

**Request Example:**

```json
{
  "user_id": "stu_5044",
  "post_id": "post_0988",
  "text": "You are so dumb and your post is pathetic"
}
```

**Response Example:**

```json
{
  "user_id": "stu_5044",
  "post_id": "post_0988",
  "toxicity_score": 0.86,
  "label": "toxic",
  "action": "blocked",
  "reasons": ["insult", "harassment"],
  "threshold": 0.75
}
```

### GET `/health`

Returns the service health status.

```json
{ "status": "ok" }
```

### GET `/version`

Returns the current model version.

```json
{ "model_version": "1.0.0" }
```

---

## Configuration (config.json)

```json
{
  "toxicity_threshold": 0.75,
  "flag_threshold": 0.5,
  "enabled_categories": ["toxic", "insult", "harassment"]
}
```

* **toxicity\_threshold:** Posts above this score are blocked.
* **flag\_threshold:** Posts in this range are flagged.
* **enabled\_categories:** Categories to monitor and detect.

---

---

## Test Results

| Sentence                                               | Toxicity Score | Label   | Action   | Reasons                            |
| ------------------------------------------------------ | -------------- | ------- | -------- | ---------------------------------- |
| You are a worthless idiot                              | 1.00           | toxic   | blocked  | toxic, harassment, insult          |
| Your existence is disgusting and unwanted.             | 0.97           | toxic   | blocked  | toxic                              |
| Go back to where you came from, no one wants you here. | 0.92           | toxic   | blocked  | toxic                              |
| You are a dumb, ugly loser.                            | 1.00           | toxic   | blocked  | toxic, harassment, insult          |
| Nobody likes your pathetic face.                       | 0.98           | toxic   | blocked  | toxic, insult                      |
| I hope you have a great day!                           | 0.02           | safe    | approved | \[]                                |
| Let’s work together on this project                    | 0.01           | safe    | approved | \[]                                |
| I appreciate your kindness and support.                | 0.04           | safe    | approved | \[]                                |
| Wow, you really know how to mess things up, don’t you? | 0.51           | flagged | flagged  | toxic                              |
| You’re not as smart as you think.                      | 0.62           | flagged | flagged  | toxic                              |
| You never listen, it’s annoying                        | 0.72           | flagged | flagged  | toxic                              |
| Get the hell out, you moron                            | 1.00           | toxic   | blocked  | toxic, harassment, obscene, insult |

---

## Key Requirements

* fastapi
* uvicorn
* numpy
* transformers
* onnxruntime

---

## Notes

* **Offline-Only:** Entirely self-contained, no external API calls.
* **Performance:** Less than 1 second per inference.
* **API-Only:** No user interface provided.

---

## License

MIT License (or specify your own license).
