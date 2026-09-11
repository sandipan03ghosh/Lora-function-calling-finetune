"""Prompt template and target formatting for the function-calling task.

The same template is used for the base model and the fine-tuned model so the comparison is fair.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are a function-calling assistant. You are given a list of available tools as JSON. "
    "Decide which tool call(s) are needed to fulfil the user's request and respond with ONLY a "
    'JSON array of calls, where each call is {"name": <tool name>, "arguments": <object>}. '
    "Use only tools from the provided list and only their documented parameters. "
    "If no tool is appropriate, respond with []."
)


def build_messages(query: str, tools: list[dict]) -> list[dict]:
    """Chat messages for a single request."""
    tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
    user = f"Available tools:\n{tools_json}\n\nUser request: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def format_target(calls: list[dict]) -> str:
    """Canonical assistant target: a compact JSON array of calls."""
    norm = [{"name": c["name"], "arguments": c.get("arguments", {})} for c in calls]
    return json.dumps(norm, ensure_ascii=False)
