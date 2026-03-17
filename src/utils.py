from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

from .schema import Category


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def find_first(patterns: Iterable[str], text: str, flags: int = re.IGNORECASE) -> Optional[re.Match]:
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return m
    return None


def snippet_around(text: str, start: int, end: int, window: int = 90) -> str:
    a = max(0, start - window)
    b = min(len(text), end + window)
    return normalize(text[a:b])


def bool_from_keywords(text: str, true_patterns: Iterable[str], false_patterns: Iterable[str]) -> Optional[bool]:
    t = text.lower()
    for p in true_patterns:
        if re.search(p, t):
            return True
    for p in false_patterns:
        if re.search(p, t):
            return False
    return None


def parse_mm_value(raw: str) -> Optional[float]:
    """
    Accepts strings like:
      '125mm', '125 mm', '12.5 cm', '4.9 in'
    Returns mm.
    """
    s = raw.strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm|in|inch|inches)\b", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "mm":
        return val
    if unit == "cm":
        return val * 10.0
    # inches
    return val * 25.4


def guess_category(text: str) -> Category:
    """
    Return a schema.Category (Literal union) so type checkers
    don't complain when assigning into PedalRecord.category.
    """
    t = text.lower()
    if "multi" in t and ("fx" in t or "effects" in t):
        return "multi_fx"
    if "reverb" in t or "verb" in t:
        return "reverb"
    if "delay" in t or "echo" in t:
        return "delay"
    if "drive" in t or "overdrive" in t or "distortion" in t or "fuzz" in t:
        return "drive"
    if "tuner" in t:
        return "tuner"
    if "looper" in t or "loop" in t:
        return "looper"
    if "split" in t or "buffer" in t or "di" in t:
        return "utility"
    if "chorus" in t or "phaser" in t or "flanger" in t or "tremolo" in t:
        return "modulation"
    return "unknown"


def parse_name_brand(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Looks for a header line like:
      Name: Aurora Delay
      Brand: NightSky Audio
    Returns (name, brand)
    """
    name = None
    brand = None
    m_name = re.search(r"(?im)^\s*name\s*:\s*(.+?)\s*$", text)
    if m_name:
        name = m_name.group(1).strip()
    m_brand = re.search(r"(?im)^\s*brand\s*:\s*(.+?)\s*$", text)
    if m_brand:
        brand = m_brand.group(1).strip()
    return name, brand