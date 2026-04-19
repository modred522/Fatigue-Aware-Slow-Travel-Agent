from __future__ import annotations

import json
import re
from uuid import uuid4


def slugify(value: str) -> str:
    value = re.sub(r"\s+", "-", value.strip().lower())
    value = re.sub(r"[^a-z0-9\-\u4e00-\u9fff]", "", value)
    return value or uuid4().hex[:8]


def extract_json(value: str):
    value = value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```json\s*(.*?)\s*```", value, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    start = min((idx for idx in [value.find("["), value.find("{")] if idx != -1), default=-1)
    end = max(value.rfind("]"), value.rfind("}"))
    if start != -1 and end != -1 and end > start:
        return json.loads(value[start : end + 1])

    raise ValueError("Unable to parse JSON from model output.")
