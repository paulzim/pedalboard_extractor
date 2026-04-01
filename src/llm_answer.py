from __future__ import annotations

from typing import List, Dict, Optional, Tuple

import requests

from src.schema import PedalRecord


# Keys we want the model to cite when relevant
CITABLE_KEYS: List[str] = [
    "power.voltage_v",
    "power.current_ma",
    "power.polarity",
    "io.mono_in",
    "io.stereo_in",
    "io.mono_out",
    "io.stereo_out",
    "io.top_jacks",
    "control.midi",
    "control.midi_type",
    "control.trs_midi_type",
    "control.expression",
    "control.tap_tempo",
    "bypass",
    "size_mm.width",
    "size_mm.depth",
]


def _short_snip(s: str, limit: int = 140) -> str:
    s = " ".join(s.split())
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _candidate_fact_lines(r: PedalRecord) -> List[str]:
    """
    Create a deterministic, compact fact block per pedal.
    We keep it consistent and small-model-friendly.
    """
    v = f"{r.power.voltage_v:.0f}V" if r.power.voltage_v is not None else "unknown"
    ma = f"{r.power.current_ma}mA" if r.power.current_ma is not None else "unknown"
    pol = r.power.polarity if r.power.polarity != "unknown" else "unknown"

    stereo_out = (
        "yes" if r.io.stereo_out is True else ("no" if r.io.stereo_out is False else "unknown")
    )
    stereo_in = (
        "yes" if r.io.stereo_in is True else ("no" if r.io.stereo_in is False else "unknown")
    )

    midi = "yes" if r.control.midi is True else ("no" if r.control.midi is False else "unknown")
    midi_type = r.control.midi_type
    trs_type = r.control.trs_midi_type

    expr = (
        "yes"
        if r.control.expression is True
        else ("no" if r.control.expression is False else "unknown")
    )
    tap = (
        "yes"
        if r.control.tap_tempo is True
        else ("no" if r.control.tap_tempo is False else "unknown")
    )
    top = "yes" if r.io.top_jacks is True else ("no" if r.io.top_jacks is False else "unknown")

    width = f"{r.size_mm.width:.0f}mm" if r.size_mm.width is not None else "unknown"
    depth = f"{r.size_mm.depth:.0f}mm" if r.size_mm.depth is not None else "unknown"

    return [
        f"id: {r.id}",
        f"name: {r.name}",
        f"category: {r.category}",
        f"power: voltage={v}, current={ma}, polarity={pol}",
        f"io: stereo_in={stereo_in}, stereo_out={stereo_out}, top_jacks={top}",
        f"control: midi={midi}, midi_type={midi_type}, trs_midi_type={trs_type}, expression={expr}, tap_tempo={tap}",
        f"size: width={width}, depth={depth}",
        f"bypass: {r.bypass}",
    ]


def _candidate_snippets(r: PedalRecord) -> Dict[str, str]:
    """
    Provide only the relevant snippet keys we recognize (stable ordering later).
    """
    out: Dict[str, str] = {}
    for k in CITABLE_KEYS:
        if k in r.sources and isinstance(r.sources[k], str) and r.sources[k].strip():
            out[k] = _short_snip(r.sources[k])
    return out


def _format_candidates(records: List[PedalRecord], max_items: int = 10) -> str:
    """
    Produce a compact, structured payload for the LLM.
    """
    blocks: List[str] = []
    for r in records[:max_items]:
        facts = "\n  ".join(_candidate_fact_lines(r))
        snips = _candidate_snippets(r)
        if snips:
            snip_lines = "\n  ".join([f"{k}: {v}" for k, v in snips.items()])
        else:
            snip_lines = "(none)"
        blocks.append(
            f"PEDAL\n"
            f"  {facts}\n"
            f"SNIPPETS\n"
            f"  {snip_lines}\n"
        )
    return "\n".join(blocks)


def _ollama_chat(
    *,
    model: str,
    base_url: str,
    system: str,
    user: str,
    timeout_s: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
        },
    }

    url = f"{base_url.rstrip('/')}/api/chat"
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    msg = data.get("message", {})
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        return "LLM returned an empty response."
    return content.strip()


def ollama_narrate(
    question: str,
    matches: List[PedalRecord],
    *,
    model: str = "llama3.2:3b",
    base_url: str = "http://127.0.0.1:11434",
    timeout_s: int = 60,
) -> str:
    """
    Uses Ollama /api/chat to narrate deterministic results.
    The LLM MUST ONLY use facts + snippets provided here.
    """
    candidates = _format_candidates(matches, max_items=10)

    # Make the model behave like a grounded "formatter"
    system = (
        "You are a careful assistant helping with guitar pedalboard planning.\n"
        "RULES (must follow):\n"
        "1) Use ONLY the facts and SNIPPETS provided. Do NOT use outside knowledge.\n"
        "2) If a detail is unknown/missing, explicitly say 'unknown' and do not guess.\n"
        "3) When you mention a spec, include a citation in parentheses using the provided snippet key.\n"
        "   Example: 'Draws 500mA (power.current_ma)'\n"
        "4) Output MUST be concise markdown with this exact structure:\n"
        "   ## Summary\n"
        "   ## Matches\n"
        "   ## Unknowns or Notes\n"
        "   ## Snippets used\n"
        "5) In 'Matches', use bullets; each bullet begins with the pedal name.\n"
        "6) In 'Snippets used', list only the snippet keys you actually cited.\n"
    )

    # We also give a tiny “template” to copy, which helps small models a lot.
    template = (
        "## Summary\n"
        "- <one sentence>\n\n"
        "## Matches\n"
        "- <Pedal Name>: <why it matches>. <Key specs> (snippet.key) (snippet.key)\n\n"
        "## Unknowns or Notes\n"
        "- <call out any unknowns that matter>\n\n"
        "## Snippets used\n"
        "- snippet.key\n"
    )

    user = (
        f"Question:\n{question}\n\n"
        f"Candidate matches (already filtered deterministically):\n{candidates}\n\n"
        "Now write the final answer following the rules exactly.\n"
        "If there are zero matches, say so in Summary and Matches, and suggest one constraint to relax.\n\n"
        f"Template:\n{template}"
    )

    return _ollama_chat(
        model=model,
        base_url=base_url,
        system=system,
        user=user,
        timeout_s=timeout_s,
    )