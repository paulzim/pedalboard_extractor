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
    Heuristic: catch obvious format misses so small local models get one
    retry with stricter instructions.
    """
    if re.search(r"<[^>\n]+>", text):
        return True

    recommended_section = re.search(
        r"##\s*(?:1\.\s*)?recommended chain([\s\S]*?)(##\s*(?:2\.\s*)?why this fits|##\s*(?:3\.\s*)?constraints honored|##\s*(?:4\.\s*)?unknowns\s*/\s*tradeoffs|##\s*(?:\d+[.)]\s*)?snippets|$)",
        text,
        re.IGNORECASE,
    )
    fit_section = re.search(
        r"##\s*(?:2\.\s*)?why this fits([\s\S]*?)(##\s*(?:3\.\s*)?constraints honored|##\s*(?:4\.\s*)?unknowns\s*/\s*tradeoffs|##\s*(?:\d+[.)]\s*)?snippets|$)",
        text,
        re.IGNORECASE,
    )
    constraints_section = re.search(
        r"##\s*(?:3\.\s*)?constraints honored([\s\S]*?)(##\s*(?:4\.\s*)?unknowns\s*/\s*tradeoffs|##\s*(?:\d+[.)]\s*)?snippets|$)",
        text,
        re.IGNORECASE,
    )
    unknowns_section = re.search(
        r"##\s*(?:4\.\s*)?unknowns\s*/\s*tradeoffs([\s\S]*?)(##\s*(?:\d+[.)]\s*)?snippets|$)",
        text,
        re.IGNORECASE,
    )

    if not recommended_section or not fit_section or not constraints_section or not unknowns_section:
        return True

    recommended_text = recommended_section.group(1)
    fit_text = fit_section.group(1)
    constraints_text = constraints_section.group(1)
    unknowns_text = unknowns_section.group(1)
    has_recommended_bullet = bool(re.search(r"^\s*-\s+", recommended_text, re.MULTILINE))
    has_non_none_chain = bool(
        re.search(r"^\s*-\s+(?!\(none\)\s*$).+", recommended_text, re.MULTILINE | re.IGNORECASE)
    )
    has_fit_bullet = bool(
        re.search(r"^\s*-\s+(?!\(none\)\s*$).+", fit_text, re.MULTILINE | re.IGNORECASE)
    )
    has_constraints_bullet = bool(re.search(r"^\s*-\s+", constraints_text, re.MULTILINE))
    has_unknowns_bullet = bool(re.search(r"^\s*-\s+", unknowns_text, re.MULTILINE))

    if not has_recommended_bullet or not has_constraints_bullet or not has_unknowns_bullet:
        return True
    if match_count == 0 and has_non_none_chain:
        return True
    if match_count > 0:
        if not has_non_none_chain or not has_fit_bullet:
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
        "   If MATCH_COUNT is >0, recommend an order using only matched pedal names.\n"
        "3) If a detail is unknown/missing, explicitly say 'unknown' and do not guess.\n"
        "4) When you mention a spec, include a citation key in parentheses like (power.current_ma).\n"
        "5) You may use ordinary signal-chain reasoning to choose pedal order, but roles must come from the provided category/facts and the user's request.\n"
        "6) Prefer pedals whose categories directly support the requested tone or effect. If matching delay/drive/reverb/modulation candidates are available, do not omit those core effect types in favor of unrelated categories.\n"
        "7) Do NOT include tuner or utility pedals in the recommended chain unless the user explicitly asked for them or they are clearly necessary to satisfy the request.\n"
        "8) Prefer chain-level reasoning over per-pedal summaries. Make the value of structured context obvious: explicit requirements honored, explicit unknowns, and reduced guesswork.\n"
        f"9) Only cite keys from this allowlist: {allowed_keys}\n"
        "10) Output MUST be concise markdown with EXACT headings:\n"
        "   ## 1. Recommended chain\n"
        "   ## 2. Why this fits\n"
        "   ## 3. Constraints honored\n"
        "   ## 4. Unknowns / tradeoffs\n"
        "   ## Snippets used\n"
        "11) In 'Recommended chain', provide one bullet with matched pedal names in recommended order, separated by ' -> '. If no matches, use '- (none)'.\n"
        "12) In 'Why this fits', use 1-2 bullets tied to the requested tone and the selected chain.\n"
        "13) In 'Constraints honored', use 1-4 bullets. Mention only requirements or capabilities that are explicitly supported by provided facts. If a requested detail is not confirmed, do NOT put it here.\n"
        "14) In 'Unknowns / tradeoffs', use 1-3 bullets for missing values, mono/stereo limitations, power uncertainty, or chain compromises. Use '- none' only if there are no meaningful unknowns or tradeoffs.\n"
        "15) Avoid generic one-bullet-per-pedal summaries unless a pedal-specific limitation is important to the chain.\n"
        "16) In 'Snippets used', list only keys you actually cited (one per line, with a leading '-').\n"
        "17) Replace template placeholders with actual matched pedals and facts; never output angle-bracket placeholder text.\n"
    )

    template = (
        "## 1. Recommended chain\n"
        "- <Pedal A> -> <Pedal B> -> <Pedal C>\n\n"
        "## 2. Why this fits\n"
        "- <brief explanation tied to the requested tone and provided facts>\n\n"
        "## 3. Constraints honored\n"
        "- <explicit requirement or capability supported by provided facts; cite any specs mentioned>\n\n"
        "## 4. Unknowns / tradeoffs\n"
        "- <missing value, limitation, or tradeoff; cite any specs mentioned>\n\n"
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
        "- Recommended chain section must contain exactly one bullet: '- (none)'.\n"
        "- Why this fits must say no chain can be recommended because MATCH_COUNT is 0.\n"
        "- Constraints honored must say none could be confirmed because there were no matching pedals.\n"
        "- Unknowns / tradeoffs must briefly say no matching pedals were available.\n"
        "If MATCH_COUNT is >0:\n"
        "- Recommended chain section must recommend an order using matched pedal names only.\n"
        "- Why this fits must explain how that order supports the requested tone.\n"
        "- Constraints honored must clearly surface explicit requirements that the chain satisfies, such as power, stereo, MIDI, or other parsed constraints, but only when they are supported by provided facts.\n"
        "- Unknowns / tradeoffs must clearly surface missing values, limitations, or compromises instead of hiding them.\n\n"
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
            f"- MATCH_COUNT is {match_count}; your chain guidance MUST match it.\n"
            "- Do not say 'no matches' if MATCH_COUNT > 0.\n"
            "- Include exactly these visible sections: Recommended chain, Why this fits, Constraints honored, Unknowns / tradeoffs.\n"
            "- Use matched pedal names only in the recommended chain.\n"
            "- Do not include tuner or utility pedals unless the request explicitly calls for them.\n"
            "- Do not omit core requested effect types when matching candidates are available.\n"
            "- Make constraints honored and unknowns / tradeoffs explicit.\n"
            "- Explain the order without inventing specs.\n"
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
