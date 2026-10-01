"""Offline checks for the simple Gemini translation function."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from google.genai import errors

from avd.errors import DubbingError
from avd.gemini_translation import translate_text


class GeminiTests(unittest.TestCase):
    def test_translates_with_requested_languages_and_model(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.return_value = SimpleNamespace(text="  Hello, world!  ")
        with patch("avd.gemini_translation.genai.Client", return_value=client), patch(
                "avd.gemini_translation.load_environment"), patch(
                "avd.gemini_translation.gemini_api_key", return_value="test-key"):
            result = translate_text("Bonjour le monde!", "French", "English", "test-model")
        self.assertEqual(result, "Hello, world!")
        args = client.models.generate_content.call_args.kwargs
        self.assertEqual(args["model"], "test-model")
        self.assertIn("French into natural, accurate English", args["contents"])
        self.assertIn("Bonjour le monde!", args["contents"])
        self.assertEqual(args["config"].temperature, 0.3)
        client.__exit__.assert_called_once()

    def test_target_language_can_be_changed(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.return_value = SimpleNamespace(text="Bonjour")
        with patch("avd.gemini_translation.genai.Client", return_value=client), patch(
                "avd.gemini_translation.load_environment"), patch(
                "avd.gemini_translation.gemini_api_key", return_value="test-key"):
            self.assertEqual(translate_text("Hello", "English", "French"), "Bonjour")
        self.assertIn("English into natural, accurate French", client.models.generate_content.call_args.kwargs["contents"])

    def test_empty_input_does_not_call_gemini(self):
        with patch("avd.gemini_translation.genai.Client") as client:
            with self.assertRaises(DubbingError):
                translate_text(" ")
        client.assert_not_called()

    def test_empty_response_is_rejected(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.return_value = SimpleNamespace(text=None)
        with patch("avd.gemini_translation.genai.Client", return_value=client), patch(
                "avd.gemini_translation.load_environment"), patch(
                "avd.gemini_translation.gemini_api_key", return_value="test-key"):
            with self.assertRaisesRegex(DubbingError, "empty translation"):
                translate_text("Bonjour", "French")

    def test_api_error_does_not_expose_key(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.side_effect = errors.APIError(503, {"message": "private-key"})
        with patch("avd.gemini_translation.genai.Client", return_value=client), patch(
                "avd.gemini_translation.load_environment"), patch(
                "avd.gemini_translation.gemini_api_key", return_value="test-key"):
            with self.assertRaisesRegex(DubbingError, "HTTP 503") as caught:
                translate_text("Bonjour", "French")
        self.assertNotIn("private-key", str(caught.exception))
        client.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
