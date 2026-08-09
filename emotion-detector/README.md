# Final Project: Emotion Detector

A Flask web application that analyzes text with the IBM Watson NLP emotion API and reports anger, disgust, fear, joy, sadness, and the dominant emotion.

## Run locally

```bash
python -m pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:5000` and enter a sentence. Unit tests use mocked API responses and do not require network access.
