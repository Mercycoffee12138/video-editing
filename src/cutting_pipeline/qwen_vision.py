from __future__ import annotations

import base64
import json
import mimetypes
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_BASE_URL = "https://api.zhizengzeng.com/alibaba"
DEFAULT_MODEL = "qwen-vl-max-latest"
DEFAULT_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class QwenVisionConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


def load_config_from_env() -> QwenVisionConfig | None:
    api_key = (
        os.getenv("ZZZ_API_KEY")
        or os.getenv("ZHIZENGZENG_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    if not api_key:
        return None

    base_url = os.getenv("ZZZ_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.getenv("QWEN_VL_MODEL", DEFAULT_MODEL)
    timeout_seconds = int(os.getenv("QWEN_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    return QwenVisionConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def _path_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_guess_mime_type(path)};base64,{encoded}"


def build_payload(image_paths: list[Path], prompt: str, model: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"image": _path_to_data_url(path)} for path in image_paths]
    content.append({"text": prompt})
    return {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        },
        "parameters": {
            "result_format": "message",
        },
    }


def _endpoint_url(config: QwenVisionConfig) -> str:
    return f"{config.base_url}/api/v1/services/aigc/multimodal-generation/generation"


def _extract_text(payload: dict[str, Any]) -> str:
    output = payload.get("output") or {}
    choices = output.get("choices") or []
    for choice in choices:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                text = item.get("text")
                if text:
                    text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)
    raise RuntimeError(f"No text found in vision response: {json.dumps(payload, ensure_ascii=False)}")


def analyze_images(
    image_paths: list[Path],
    prompt: str,
    config: QwenVisionConfig,
) -> dict[str, Any]:
    payload = build_payload(image_paths=image_paths, prompt=prompt, model=config.model)
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        _endpoint_url(config),
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=config.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except socket.timeout as exc:
        raise RuntimeError(
            f"Qwen vision request timed out after {config.timeout_seconds}s."
        ) from exc
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen vision request failed with HTTP {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Qwen vision request failed: {exc.reason}") from exc

    parsed = json.loads(raw)
    return {
        "text": _extract_text(parsed),
        "raw_response": parsed,
        "request_url": _endpoint_url(config),
    }
