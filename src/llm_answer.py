from __future__ import annotations

from typing import List, Dict, Set, Tuple
import re
import requests

from src.schema import PedalRecord

# Keys we allow the model to cite (stable, predictable)
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
    return s if len(s) <= limit else (s[: limit - 1].rstrip() + "…")


def _candidate_fact_lines(r: PedalRecord) -> List[str]:
    v = f"{r.power.voltage_v:.0f}V" if r.power.voltage_v is not None else "unknown"
    ma = f"{r.power.current_ma}mA" if r.power.current_ma is not None else "unknown"
    pol = r.power.polarity if r.power.polarity != "unknown" else "unknown"

    stereo_out = "yes" if r.io.stereo_out is True else ("no" if r.io.stereo_out is False else "unknown")
    stereo_in = "yes" if r.io.stereo_in is True else ("no" if r.io.stereo_in is False else "unknown")

    midi = "yes" if r.control.midi is True else ("no" if r.control.midi is False else "unknown")
    expr = "yes" if r.control.expression is True else ("no" if r.control.expression is False else "unknown")
    tap = "yes" if r.control.tap_tempo is True else ("no" if r.control.tap_tempo is False else "unknown")
    top = "yes" if r.io.top_jacks is True else ("no" if r.io.top_jacks is False else "unknown")

    width = f"{r.size_mm.width:.0f}mm" if r.size_mm.width is not None else "unknown"
    depth = f"{r.size_mm.depth:.0f}mm" if r.size_mm.depth is not None else "unknown"

    return [
        f"id: {r.id}",
        f"name: {r.name}",
        f"category: {r.category}",
        f"power: voltage={v}, current={ma}, polarity={pol}",
        f"io: stereo_in={stereo_in}, stereo_out={stereo_out}, top_jacks={top}",
        f"control: midi={midi}, midi_type={r.control.midi_type}, trs_midi_type={r.control.trs_midi_type}, expression={expr}, tap_tempo={tap}",
        f"size: width={width}, depth={depth}",
        f"bypass: {r.bypass}",
    ]


def _candidate_snippets(r: PedalRecord) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in CITABLE_KEYS:
        if k in r.sources and isinstance(r.sources[k], str) and r.sources[k].strip():
            out[k] = _short_snip(r.sources[k])
    return out


def _format_candidates(records: List[PedalRecord], max_items: int = 10) -> str:
    blocks: List[str] = []
    for r in records[:max_items]:
        facts = "\n  ".join(_candidate_fact_lines(r))
        snips = _candidate_snippets(r)
        snip_lines = "\n  ".join([f"{k}: {v}" for k, v in snips.items()]) if snips else "(none)"
        blocks.append(
            "PEDAL\n"
            f"  {facts}\n"
            "SNIPPETS\n"
            f"  {snip_lines}\n"
        )
    return "\n".join(blocks)


def _ollama_chat(*, model: str, base_url: str, system: str, user: str, timeout_s: int, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 4096,
        },
    }

    url = f"{base_url.rstrip('/')}/api/chat"
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        return "LLM returned an empty response."
    return content.strip()


def _extract_used_keys(text: str) -> Set[str]:
    # pull anything that looks like (some.key)
    keys = set(re.findall(r"\(([a-z0-9_]+\.[a-z0-9_]+)\)", text))
    return keys


def _looks_inconsistent(text: str, match_count: int) -> bool:
    """
    Heuristic: if match_count > 0 but Summary claims "no" / "none" / "0",
    if match_count == 0 but it lists something under Matches, or if a
    positive match set omits the chain recommendation sections.
    """
    lower = text.lower()

    # find Summary section
    # not perfect, but bounded so later "none" notes do not trigger repair
    summary_section = re.search(
        r"##\s*summary([\s\S]*?)(##\s*signal chain|##\s*matches|##\s*unknowns|##\s*snippets|$)",
        text,
        re.IGNORECASE,
    )
    summary_text = summary_section.group(1).lower() if summary_section else lower
    summary_claims_none = bool(re.search(r"\b(no|none|0)\b", summary_text))

    # find number of bullets in Matches
    matches_section = re.search(r"##\s*matches([\s\S]*?)(##\s*unknowns|##\s*snippets|$)", text, re.IGNORECASE)
    bullets = 0
    if matches_section:
        bullets = len(re.findall(r"^\s*-\s+", matches_section.group(1), re.MULTILINE))

    if match_count > 0 and summary_claims_none:
        return True
    if match_count == 0 and bullets > 0:
        return True
    # also: if match_count > 0 but bullets == 0
    if match_count > 0 and bullets == 0:
        return True
    if re.search(r"<[^>\n]+>", text):
        return True
    if match_count > 0:
        chain_section = re.search(
            r"##\s*signal chain([\s\S]*?)(##\s*why this order|##\s*matches|$)",
            text,
            re.IGNORECASE,
        )
        if not chain_section:
            return True
        has_chain_bullet = bool(
            re.search(r"^\s*-\s+(?!\(none\)\s*$).+", chain_section.group(1), re.MULTILINE | re.IGNORECASE)
        )
        if not has_chain_bullet:
            return True
        if not re.search(r"##\s*why this order", text, re.IGNORECASE):
            return True
    return False


def ollama_narrate(
    question: str,
    matches: List[PedalRecord],
    *,
    model: str = "llama3.2:3b",
    base_url: str = "http://127.0.0.1:11434",
    timeout_s: int = 60,
) -> str:
    match_count = len(matches)
    match_ids = [m.id for m in matches[:10]]  # keep short
    candidates = _format_candidates(matches, max_items=10)
    allowed_keys = ", ".join(CITABLE_KEYS)

    system = (
        "You are a careful assistant helping with guitar pedalboard planning.\n"
        "You MUST follow these rules:\n"
        "1) Use ONLY the facts and SNIPPETS provided for pedal specs, capabilities, and constraints. Do NOT use outside specs.\n"
        "2) Never contradict MATCH_COUNT. If MATCH_COUNT is 0, there are no matches.\n"
        "   If MATCH_COUNT is >0, Summary must say how many matches were found.\n"
        "3) If a detail is unknown/missing, explicitly say 'unknown' and do not guess.\n"
        "4) When you mention a spec, include a citation key in parentheses like (power.current_ma).\n"
        "5) You may use ordinary signal-chain reasoning to choose pedal order, but roles must come from the provided category/facts and the user's request.\n"
        f"6) Only cite keys from this allowlist: {allowed_keys}\n"
        "7) Output MUST be concise markdown with EXACT headings:\n"
        "   ## Summary\n"
        "   ## Signal Chain\n"
        "   ## Why This Order\n"
        "   ## Matches\n"
        "   ## Unknowns or Notes\n"
        "   ## Snippets used\n"
        "8) In 'Signal Chain', provide one bullet with matched pedal names in recommended order, separated by ' -> '. If no matches, use '- (none)'.\n"
        "9) In 'Why This Order', use 1-2 bullets explaining how the order supports the requested tone without inventing pedal behavior.\n"
        "10) In 'Matches', use bullets; each bullet starts with the pedal name and states its role in the chain.\n"
        "11) In 'Unknowns or Notes', note relevant tradeoffs, uncertainties, or missing details; use '- none' if there are no useful notes.\n"
        "12) In 'Snippets used', list only keys you actually cited (one per line, with a leading '-').\n"
        "13) Replace template placeholders with actual matched pedals and facts; never output angle-bracket placeholder text.\n"
    )

    template = (
        "## Summary\n"
        f"- Found {match_count} match(es). Recommended chain: <Pedal A> -> <Pedal B>.\n\n"
        "## Signal Chain\n"
        "- <Pedal A> -> <Pedal B> -> <Pedal C>\n\n"
        "## Why This Order\n"
        "- <how this order supports the requested tone, using only provided categories/facts>\n"
        "- <tradeoff or constraint note if relevant; cite any spec mentioned>\n\n"
        "## Matches\n"
        "- <Pedal Name>: <role in the chain>. <key specs with citations when specs are mentioned>\n\n"
        "## Unknowns or Notes\n"
        "- <tradeoffs, uncertainties, missing details, or none>\n\n"
        "## Snippets used\n"
        "- <keys cited above>\n"
    )

    user = (
        f"MATCH_COUNT: {match_count}\n"
        f"MATCH_IDS: {match_ids}\n\n"
        f"Question:\n{question}\n\n"
        f"Candidate matches (already filtered deterministically):\n{candidates}\n\n"
        "Write the final answer following the rules exactly.\n"
        "If MATCH_COUNT is 0:\n"
        "- Summary must say no matches.\n"
        "- Matches section must contain exactly one bullet: '- (none)'.\n"
        "If MATCH_COUNT is >0:\n"
        "- Summary must say how many matches.\n"
        "- Signal Chain section must recommend an order using matched pedal names only.\n"
        "- Why This Order must explain how that order supports the requested tone.\n"
        "- Matches section must list those pedals and their role in the chain.\n\n"
        f"Template:\n{template}"
    )

    # First attempt: small temperature
    out = _ollama_chat(
        model=model,
        base_url=base_url,
        system=system,
        user=user,
        timeout_s=timeout_s,
        temperature=0.2,
    )

    # Self-heal if it contradicts itself (common with small models)
    if _looks_inconsistent(out, match_count):
        repair = (
            "Your previous answer contradicted MATCH_COUNT or omitted required chain guidance.\n"
            "Regenerate the entire response.\n"
            "Hard requirements:\n"
            f"- MATCH_COUNT is {match_count}; your Summary MUST match it.\n"
            "- Do not say 'no matches' if MATCH_COUNT > 0.\n"
            "- Include a Signal Chain section with matched pedal names in recommended order.\n"
            "- Include a Why This Order section explaining the order without inventing specs.\n"
            "- Use correct citation formatting: (key) with keys from allowlist only.\n"
        )
        out = _ollama_chat(
            model=model,
            base_url=base_url,
            system=system,
            user=user + "\n\n" + repair,
            timeout_s=timeout_s,
            temperature=0.0,
        )

    # Optional: you can also enforce Snippets used == cited keys,
    # but for now we just nudge it and keep output readable.

    return out
