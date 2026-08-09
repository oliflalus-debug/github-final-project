"""Unit tests for the emotion detection package."""

import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Validate formatted results and error handling."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy_is_dominant(self, mock_post):
        response = Mock(status_code=200)
        response.json.return_value = {"emotionPredictions": [{"emotion": {
            "anger": 0.01, "disgust": 0.01, "fear": 0.02,
            "joy": 0.94, "sadness": 0.02}}]}
        response.raise_for_status.return_value = None
        mock_post.return_value = response
        self.assertEqual(emotion_detector("I am glad")["dominant_emotion"], "joy")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_blank_input_returns_none_scores(self, mock_post):
        mock_post.return_value = Mock(status_code=400)
        result = emotion_detector("")
        self.assertIsNone(result["dominant_emotion"])
        self.assertTrue(all(value is None for value in result.values()))


if __name__ == "__main__":
    unittest.main()
