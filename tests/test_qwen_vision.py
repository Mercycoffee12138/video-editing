import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib import error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cutting_pipeline.qwen_vision import (  # noqa: E402
    QwenVisionConfig,
    QwenVisionContentBlockedError,
    analyze_images,
)


class QwenVisionTests(unittest.TestCase):
    def test_analyze_images_retries_transient_url_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "frame.jpg"
            image_path.write_bytes(b"fake-image")

            response = MagicMock()
            response.read.return_value = json.dumps(
                {
                    "output": {
                        "choices": [
                            {
                                "message": {
                                    "content": [{"text": '{"ok": true}'}],
                                }
                            }
                        ]
                    }
                }
            ).encode("utf-8")
            context_manager = MagicMock()
            context_manager.__enter__.return_value = response
            context_manager.__exit__.return_value = False

            config = QwenVisionConfig(
                api_key="test-key",
                max_retries=3,
                retry_backoff_seconds=0.0,
            )

            with patch(
                "cutting_pipeline.qwen_vision.request.urlopen",
                side_effect=[error.URLError(ConnectionResetError(54, "Connection reset by peer")), context_manager],
            ) as mock_urlopen:
                payload = analyze_images([image_path], "test prompt", config)

        self.assertEqual(payload["text"], '{"ok": true}')
        self.assertEqual(mock_urlopen.call_count, 2)

    def test_analyze_images_raises_after_retry_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "frame.jpg"
            image_path.write_bytes(b"fake-image")

            config = QwenVisionConfig(
                api_key="test-key",
                max_retries=2,
                retry_backoff_seconds=0.0,
            )

            with patch(
                "cutting_pipeline.qwen_vision.request.urlopen",
                side_effect=error.URLError(ConnectionResetError(54, "Connection reset by peer")),
            ):
                with self.assertRaises(RuntimeError) as context:
                    analyze_images([image_path], "test prompt", config)

        self.assertIn("Connection reset by peer", str(context.exception))

    def test_analyze_images_raises_content_blocked_error_for_inspection_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "frame.jpg"
            image_path.write_bytes(b"fake-image")

            response = MagicMock()
            response.read.return_value = json.dumps(
                {
                    "request_id": "test",
                    "code": "DataInspectionFailed",
                    "message": "<400> InternalError.Algo.DataInspectionFailed: blocked",
                }
            ).encode("utf-8")
            context_manager = MagicMock()
            context_manager.__enter__.return_value = response
            context_manager.__exit__.return_value = False

            config = QwenVisionConfig(
                api_key="test-key",
                max_retries=2,
                retry_backoff_seconds=0.0,
            )

            with patch("cutting_pipeline.qwen_vision.request.urlopen", return_value=context_manager):
                with self.assertRaises(QwenVisionContentBlockedError) as context:
                    analyze_images([image_path], "test prompt", config)

        self.assertIn("content inspection", str(context.exception))


if __name__ == "__main__":
    unittest.main()
