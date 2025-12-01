import re
import json
from json_repair import loads
from pydantic import BaseModel, Field
from typing import Any, Dict, Tuple, Optional


def transform_response(
        content:Any,
        return_dict: bool = False
)-> int | Tuple[int, Dict[str, Any]]:
    
    next_content = loads(content)
    # 1) If it's already a dict, use it directly
    if isinstance(content, dict):
        data = next_content

    else:
        args = None
        tool_calls = getattr(content, "tool_calls", None)
        if tool_calls and isinstance(tool_calls, list):
            tc0 = tool_calls[0]
            if isinstance(tc0, dict):
                args = tc0.get("args")
            else:
                args = getattr(tc0, "args", None)
        if isinstance(args, dict):
            data = args
        else:
            text = None
            if hasattr(content, "content"):
                text = content.content
            if not text and hasattr(content, "response_metadata"):
                cands = content.response_metadata.get("candidates", [])
                if cands:
                    parts = cands[0].get("content", {}).get("parts", [])
                    text = "".join(
                        p.get("text", "")
                        for p in parts
                        if isinstance(p, dict) and "text" in p
                    )
            if isinstance(content, str):
                text = content if not text else text

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Failed to extract parsable text or structured args from response.")

            raw = text.strip()
            raw = re.sub(r'^[^\{\[]*?:\s*', '', raw, count=1)

            if raw.startswith("```"):
                raw = raw.strip("`").strip()
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

            m = re.search(r"\{.*\}", raw, flags=re.S)
            if not m:
                raise ValueError(f"No JSON object found in response. Snippet: {raw[:200]!r}")
            json_str = m.group(0)

            data = json.loads(json_str)

    # 2) Extract trading decision and validate
    if "trading decision" not in data:
        raise KeyError("'trading decision' field is missing.")

    decision = data["trading decision"]
    if not isinstance(decision, int):
        raise TypeError("'trading decision' must be an integer.")
    if decision not in (-1, 0, 1):
        raise ValueError(f"'trading decision' must be -1, 0, or 1, but got {decision}")

    return (decision, data) if return_dict else decision



    return data
    
