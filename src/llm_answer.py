from __future__ import annotations

import json
from typing import List, Dict, Optional

import requests

from src.schema import PedalRecord


def _format_candidates(records: List[PedalRecord], max_items: int = 10) -> str:
    lines: List[str] = []
    for r in records[:max_items]:
        v = f"{r.power.voltage_v:.0f}V" if r.power.voltage_v is not None else "unknown V"
        ma = f"{r.power.current_ma}mA" if r.power.current_ma is not None else "unknown mA"
        stereo_out = "yes" if r.io.stereo_out is True else ("no" if r.io.stereo_out is False else "unknown")
        midi = "yes" if r.control.midi is True else ("no" if r.control.midi is False else "unknown")
        midi_type = r.control.midi_type
        trs_type = r.control.trs_midi_type
        expr = "yes" if r.control.expression is True else ("no" if r.control.expression is False else "unknown")
        top = "yes" if r.io.top_jacks is True else ("no" if r.io.top_jacks is False else "unknown")

        # include a couple key sources if available
        src_bits: List[str] = []
        for k in [
            "power.voltage_v",
            "power.current_ma",
            "io.stereo_out",
            "control.midi",
            "control.midi_type",
            "control.trs_midi_type",
            "control.expression",
            "io.top_jacks",
            "bypass",
        ]:
            if k in r.sources:
                src_bits.append(f"{k}: {r.sources[k]}")
        src_text = "\n    ".join(src_bits) if src_bits else "(no snippets captured)"

        lines.append(
            f"- id: {r.id}\n"
            f"  name: {r.name}\n"
            f"  category: {r.category}\n"
            f"  voltage/current: {v}, {ma}\n"
            f"  stereo_out: {stereo_out}\n"
            f"  midi: {midi} (type: {midi_type}, trs: {trs_type})\n"
            f"  expression: {expr}\n"
            f"  top_jacks: {top}\n"
            f"  bypass: {r.bypass}\n"
            f"  sources:\n    {src_text}\n"
        )
    return "\n".join(lines)


def ollama_narrate(
    question: str,
    matches: List[PedalRecord],
    *,
    model: str = "llama3.2:3b",
    base_url: str = "http://127.0.0.1:11434",
    timeout_s: int = 60,
) -> str:
    """
    Uses Ollama /api/chat to produce a concise, friendly answer.
    The LLM MUST only use extracted facts + snippets provided here.
    """
    candidates = _format_candidates(matches, max_items=10)

    system = (
        "You are a careful assistant helping with guitar pedalboard planning.\n"
        "You MUST only use the facts provided in the candidate records and their source snippets.\n"
        "If a required fact is unknown/missing, say it's unknown and don't guess.\n"
        "Be concise. Prefer bullet points.\n"
        "When you mention a spec (e.g., current draw, MIDI type), include a short snippet citation in parentheses.\n"
    )

    user = (
        f"Question:\n{question}\n\n"
        f"Candidate matches (already filtered deterministically):\n{candidates}\n\n"
        "Write the final answer.\n"
        "- Start with a 1-sentence summary.\n"
        "- Then list matches with key specs relevant to the question.\n"
        "- If there are zero matches, say so and suggest what constraint might be relaxed.\n"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    url = f"{base_url.rstrip('/')}/api/chat"
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    # Ollama returns: {"message": {"role":"assistant","content":"..."} ...}
    msg = data.get("message", {})
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        return "LLM returned an empty response."
    return content.strip()