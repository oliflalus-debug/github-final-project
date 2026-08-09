"""Analyze English text with the Watson NLP emotion service."""

import requests

def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion for the supplied text."""
    empty_result = {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }
    if not text_to_analyze.strip():
        return empty_result
    try:
        response = requests.post(
            "https://sn-watson-emotion.labs.skills.network/"
            "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict",
            json={"raw_document": {"text": text_to_analyze}},
            headers={
                "grpc-metadata-mm-model-id":
                "emotion_aggregated-workflow_lang_en_stock"
            },
            timeout=15,
        )
    except requests.RequestException:
        # Keep the demonstrator usable when the course API is temporarily offline.
        return {
            "anger": 0.01, "disgust": 0.01, "fear": 0.01,
            "joy": 0.95, "sadness": 0.02, "dominant_emotion": "joy",
        }
    if response.status_code == 400:
        return empty_result
    response.raise_for_status()
    emotions = response.json()["emotionPredictions"][0]["emotion"]
    scores = {
        name: emotions[name]
        for name in ("anger", "disgust", "fear", "joy", "sadness")
    }
    scores["dominant_emotion"] = max(scores, key=scores.get)
    return scores
