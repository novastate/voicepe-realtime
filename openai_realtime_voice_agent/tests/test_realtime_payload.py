import unittest

from app.realtime_payload import transform_gpt_transcription_language


class TestGptTranscriptionLanguage(unittest.TestCase):
    def test_replaces_singular_language_on_pipecat_session_payload(self):
        payload = {
            "type": "session.update",
            "session": {
                "input_audio_transcription": {
                    "model": "gpt-live-transcribe",
                    "language": "en",
                }
            },
        }

        transform_gpt_transcription_language(payload)

        self.assertEqual(
            payload["session"]["input_audio_transcription"],
            {"model": "gpt-live-transcribe", "languages": ["en"]},
        )

    def test_replaces_singular_language_for_gpt_transcribe(self):
        payload = {
            "type": "session.update",
            "session": {
                "input_audio_transcription": {
                    "model": "gpt-transcribe",
                    "language": "en",
                }
            },
        }

        transform_gpt_transcription_language(payload)

        self.assertEqual(
            payload["session"]["input_audio_transcription"],
            {"model": "gpt-transcribe", "languages": ["en"]},
        )

    def test_leaves_other_transcription_models_unchanged(self):
        payload = {
            "type": "session.update",
            "session": {
                "input_audio_transcription": {
                    "model": "gpt-4o-transcribe",
                    "language": "en",
                }
            },
        }

        transform_gpt_transcription_language(payload)

        self.assertEqual(
            payload["session"]["input_audio_transcription"],
            {"model": "gpt-4o-transcribe", "language": "en"},
        )
