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

_CORE_EFFECT_CATEGORIES = {"drive", "delay", "reverb"}
_PROMPT_DEBUG_MARKER = "GROUND_PROMPT_TRACE_V1"


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


def _extract_recommended_chain_items(text: str) -> List[str]:
    recommended_section = re.search(
        r"##\s*(?:1\.\s*)?recommended chain([\s\S]*?)(##\s*(?:2\.\s*)?why this fits|##\s*(?:3\.\s*)?constraints honored|##\s*(?:4\.\s*)?unknowns\s*/\s*tradeoffs|##\s*(?:\d+[.)]\s*)?snippets|$)",
        text,
        re.IGNORECASE,
    )
    if not recommended_section:
        return []

    bullet = re.search(r"^\s*-\s+(.+)$", recommended_section.group(1), re.MULTILINE)
    if not bullet:
        return []

    chain_text = bullet.group(1).strip()
    if re.fullmatch(r"\(none\)", chain_text, re.IGNORECASE):
        return []

    parts = re.split(r"\s*(?:->|→|=>|➡|⟶)\s*", chain_text)
    items: List[str] = []
    for part in parts:
        item = part.strip()
        item = item.strip("`").strip()
        item = item.strip("*_ ").strip()
        item = item.strip("\"'“”‘’").strip()
        item = item.rstrip(".,;:").strip()
        if item:
            items.append(item)
    return items


def _names_appear_in_order(text: str, names: List[str]) -> bool:
    lower = text.lower()
    pos = -1
    for name in names:
        idx = lower.find(name.lower(), pos + 1)
        if idx < 0:
            return False
        pos = idx
    return True


def _mentioned_candidate_names(text: str, candidate_names: List[str]) -> List[str]:
    lower = text.lower()
    return [name for name in candidate_names if name.lower() in lower]


def _duplicate_core_category_items(chain_items: List[str], name_to_category: Dict[str, str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for item in chain_items:
        category = name_to_category.get(item)
        if category in _CORE_EFFECT_CATEGORIES:
            grouped.setdefault(category, []).append(item)
    return {category: items for category, items in grouped.items() if len(items) > 1}


def _fit_mentions_only_chain_names(fit_text: str, chain_items: List[str], candidate_names: List[str]) -> bool:
    mentioned = _mentioned_candidate_names(fit_text, candidate_names)
    chain_name_set = {name.lower() for name in chain_items}
    return all(name.lower() in chain_name_set for name in mentioned)


def _fit_explicitly_justifies_stacking(
    fit_text: str,
    duplicate_items_by_category: Dict[str, List[str]],
) -> bool:
    lower = fit_text.lower()
    if not duplicate_items_by_category:
        return True
    for category, items in duplicate_items_by_category.items():
        if category not in lower:
            return False
        justified = False
        for idx, left in enumerate(items):
            for right in items[idx + 1 :]:
                pair_pattern = re.compile(
                    rf"(?:{re.escape(left.lower())}[\s\S]{{0,120}}(?:stack|stacking|layer|layering)[\s\S]{{0,120}}{re.escape(right.lower())})"
                    rf"|(?:{re.escape(right.lower())}[\s\S]{{0,120}}(?:stack|stacking|layer|layering)[\s\S]{{0,120}}{re.escape(left.lower())})"
                )
                if pair_pattern.search(lower):
                    justified = True
                    break
            if justified:
                break
        if not justified:
            return False
    return True


def _looks_inconsistent(
    text: str,
    match_count: int,
    candidate_names: List[str],
    name_to_category: Dict[str, str],
) -> bool:
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
    chain_items = _extract_recommended_chain_items(text)
    has_recommended_bullet = bool(re.search(r"^\s*-\s+", recommended_text, re.MULTILINE))
    has_non_none_chain = bool(
        re.search(r"^\s*-\s+(?!\(none\)\s*$).+", recommended_text, re.MULTILINE | re.IGNORECASE)
    )
    has_fit_bullet = bool(
        re.search(r"^\s*-\s+(?!\(none\)\s*$).+", fit_text, re.MULTILINE | re.IGNORECASE)
    )
    has_constraints_bullet = bool(re.search(r"^\s*-\s+", constraints_text, re.MULTILINE))
    has_debug_marker = bool(
        re.search(
            rf"^\s*-\s+Debug marker:\s*{re.escape(_PROMPT_DEBUG_MARKER)}\s*$",
            constraints_text,
            re.MULTILINE | re.IGNORECASE,
        )
    )
    has_unknowns_bullet = bool(re.search(r"^\s*-\s+", unknowns_text, re.MULTILINE))

    if not has_recommended_bullet or not has_constraints_bullet or not has_debug_marker or not has_unknowns_bullet:
        return True
    if match_count == 0 and has_non_none_chain:
        return True
    if match_count > 0:
        if not has_non_none_chain or not has_fit_bullet:
            return True
        if not chain_items:
            return True
        if any(item not in set(candidate_names) for item in chain_items):
            return True
        if len(set(chain_items)) != len(chain_items):
            return True
        if not _names_appear_in_order(fit_text, chain_items):
            return True
        if not _fit_mentions_only_chain_names(fit_text, chain_items, candidate_names):
            return True
        if not _fit_explicitly_justifies_stacking(
            fit_text,
            _duplicate_core_category_items(chain_items, name_to_category),
        ):
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
    candidate_names = [m.name for m in matches[:10]]
    name_to_category = {m.name: m.category for m in matches[:10]}
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
        "12) Recommend the smallest coherent set of pedals that satisfies the request. Prefer one best match per role, usually 1-3 pedals total.\n"
        "13) Every chain item MUST be copied exactly from CANDIDATE_NAMES. Do not paraphrase, abbreviate, translate, or replace a pedal name with a description.\n"
        "14) Each pedal may appear at most once in the chain. Recommending fewer pedals is better than padding with duplicates or weak fits.\n"
        "15) Do not use more than one drive, delay, or reverb pedal unless stacking is clearly necessary. If you stack, explicitly justify it in 'Why this fits' using the duplicated pedal names and the word 'stack' or 'layer'.\n"
        "16) In 'Why this fits', use 1-2 bullets that walk through the exact chain from left to right using the exact pedal names in order. Do not mention candidate pedals that are not in the recommended chain.\n"
        f"17) The first bullet in 'Constraints honored' MUST be exactly: '- Debug marker: {_PROMPT_DEBUG_MARKER}'\n"
        "18) After the debug marker, 'Constraints honored' may use 0-3 additional bullets. Mention only requirements or capabilities that are explicitly supported by provided facts. If a requested detail is not confirmed, do NOT put it here.\n"
        "19) In 'Unknowns / tradeoffs', use 1-3 bullets for missing values, mono/stereo limitations, power uncertainty, or chain compromises. Use '- none' only if there are no meaningful unknowns or tradeoffs.\n"
        "20) Avoid generic one-bullet-per-pedal summaries unless a pedal-specific limitation is important to the chain.\n"
        "21) In 'Snippets used', list only keys you actually cited (one per line, with a leading '-').\n"
        "22) Replace template placeholders with actual matched pedals and facts; never output angle-bracket placeholder text.\n"
    )

    template = (
        "## 1. Recommended chain\n"
        "- <Pedal A> -> <Pedal B> -> <Pedal C>\n\n"
        "## 2. Why this fits\n"
        "- <brief explanation tied to the requested tone and provided facts>\n\n"
        "## 3. Constraints honored\n"
        f"- Debug marker: {_PROMPT_DEBUG_MARKER}\n"
        "- <explicit requirement or capability supported by provided facts; cite any specs mentioned>\n\n"
        "## 4. Unknowns / tradeoffs\n"
        "- <missing value, limitation, or tradeoff; cite any specs mentioned>\n\n"
        "## Snippets used\n"
        "- <keys cited above>\n"
    )

    user = (
        f"MATCH_COUNT: {match_count}\n"
        f"MATCH_IDS: {match_ids}\n\n"
        f"CANDIDATE_NAMES: {candidate_names}\n\n"
        f"CANDIDATE_NAME_TO_CATEGORY: {name_to_category}\n\n"
        f"Question:\n{question}\n\n"
        f"Candidate matches (already filtered deterministically):\n{candidates}\n\n"
        "Write the final answer following the rules exactly.\n"
        "If MATCH_COUNT is 0:\n"
        "- Recommended chain section must contain exactly one bullet: '- (none)'.\n"
        "- Why this fits must say no chain can be recommended because MATCH_COUNT is 0.\n"
        f"- Constraints honored must include the exact debug marker bullet '- Debug marker: {_PROMPT_DEBUG_MARKER}'.\n"
        "- Constraints honored may then say none could be confirmed because there were no matching pedals.\n"
        "- Unknowns / tradeoffs must briefly say no matching pedals were available.\n"
        "If MATCH_COUNT is >0:\n"
        "- Recommended chain section must recommend an order using matched pedal names only.\n"
        "- Why this fits must explain that exact chain from left to right using the same pedal names in order.\n"
        "- Prefer the smallest coherent chain rather than listing every relevant candidate.\n"
        f"- Constraints honored must include the exact debug marker bullet '- Debug marker: {_PROMPT_DEBUG_MARKER}'.\n"
        "- Constraints honored must then clearly surface explicit requirements that the chain satisfies, such as power, stereo, MIDI, or other parsed constraints, but only when they are supported by provided facts.\n"
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
    if _looks_inconsistent(out, match_count, candidate_names, name_to_category):
        repair = (
            "Your previous answer contradicted MATCH_COUNT or omitted required chain guidance.\n"
            "Regenerate the entire response.\n"
            "Hard requirements:\n"
            f"- MATCH_COUNT is {match_count}; your chain guidance MUST match it.\n"
            "- Do not say 'no matches' if MATCH_COUNT > 0.\n"
            "- Include exactly these visible sections: Recommended chain, Why this fits, Constraints honored, Unknowns / tradeoffs.\n"
            f"- Include this exact visible bullet in Constraints honored: '- Debug marker: {_PROMPT_DEBUG_MARKER}'.\n"
            "- Use matched pedal names only in the recommended chain.\n"
            "- Every chain item must be copied exactly from CANDIDATE_NAMES.\n"
            "- Do not paraphrase pedal names into descriptive phrases.\n"
            "- Do not repeat a pedal in the chain.\n"
            "- It is better to recommend fewer pedals than to pad the chain with duplicates.\n"
            "- Prefer one best pedal per role instead of redundant drive/delay/reverb stacking.\n"
            "- If you stack pedals from the same core category, explicitly justify that stack in Why this fits using the exact duplicated pedal names and the word 'stack' or 'layer'.\n"
            "- Why this fits must walk through the exact chain in order using the same pedal names.\n"
            "- Do not mention candidate pedals in Why this fits unless they are in the recommended chain.\n"
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
