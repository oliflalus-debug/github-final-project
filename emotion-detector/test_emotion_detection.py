"""Unit tests for the emotion detection package."""

import unittest
from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Validate the dominant emotion for five representative statements."""

    def run_emotion_test(self, scores):
        """Mock Watson NLP and return the formatted result."""
        response = Mock(status_code=200)
        response.json.return_value = {
            "emotionPredictions": [{"emotion": scores}]
        }
        response.raise_for_status.return_value = None
        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=response,
        ):
            return emotion_detector("Test statement")

    def test_joy(self):
        """Joy is detected as the dominant emotion."""
        result = self.run_emotion_test(
            {"anger": 0.01, "disgust": 0.01, "fear": 0.02,
             "joy": 0.94, "sadness": 0.02}
        )
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_anger(self):
        """Anger is detected as the dominant emotion."""
        result = self.run_emotion_test(
            {"anger": 0.90, "disgust": 0.03, "fear": 0.02,
             "joy": 0.01, "sadness": 0.04}
        )
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_disgust(self):
        """Disgust is detected as the dominant emotion."""
        result = self.run_emotion_test(
            {"anger": 0.03, "disgust": 0.91, "fear": 0.02,
             "joy": 0.01, "sadness": 0.03}
        )
        self.assertEqual(result["dominant_emotion"], "disgust")

    def test_sadness(self):
        """Sadness is detected as the dominant emotion."""
        result = self.run_emotion_test(
            {"anger": 0.02, "disgust": 0.01, "fear": 0.03,
             "joy": 0.02, "sadness": 0.92}
        )
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_fear(self):
        """Fear is detected as the dominant emotion."""
        result = self.run_emotion_test(
            {"anger": 0.02, "disgust": 0.01, "fear": 0.93,
             "joy": 0.01, "sadness": 0.03}
        )
        self.assertEqual(result["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()
