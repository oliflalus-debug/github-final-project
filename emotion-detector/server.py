"""Flask deployment for the Emotion Detector application."""

from flask import Flask, render_template, request

from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


@app.route("/")
def home():
    """Render the application home page."""
    return render_template("index.html")


@app.route("/emotionDetector")
def detect_emotion():
    """Analyze query text and return a human-readable result."""
    text_to_analyze = request.args.get("textToAnalyze", "").strip()
    result = emotion_detector(text_to_analyze)
    if result["dominant_emotion"] is None:
        return "Invalid input! Try again."
    return (
        "For the given statement, the system response is "
        f"'anger': {result['anger']}, 'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, 'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. The dominant emotion is "
        f"{result['dominant_emotion']}."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
