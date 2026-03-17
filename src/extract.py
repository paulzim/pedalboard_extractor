from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional

from .schema import MidiType, PedalRecord, Polarity, TrsMidiType, BypassType
from .utils import (
    find_first,
    guess_category,
    parse_mm_value,
    parse_name_brand,
    snippet_around,
)

# ---------- Source helper ----------

def add_source(sources: Dict[str, str], field_path: str, text: str, match: re.Match) -> None:
    sources[field_path] = snippet_around(text, match.start(), match.end())

def add_source_first(sources: Dict[str, str], field_path: str, text: str, patterns: list[str]) -> None:
    m = find_first(patterns, text)
    if m:
        add_source(sources, field_path, text, m)

# ---------- Field extractors ----------

def extract_polarity(text: str) -> Polarity:
    t = text.lower()
    if "center-negative" in t or "centre-negative" in t or "center negative" in t:
        return "center_negative"
    if "center-positive" in t or "centre-positive" in t or "center positive" in t:
        return "center_positive"
    return "unknown"

def extract_midi(text: str) -> Optional[bool]:
    t = text.lower()
    if re.search(r"\bmidi\b", t):
        if re.search(r"\bno\s+midi\b|\bwithout\s+midi\b|\bmidi\s*:\s*no\b", t):
            return False
        return True
    return None

def extract_midi_type(text: str) -> MidiType:
    t = text.lower()
    if not re.search(r"\bmidi\b", t):
        return "unknown"
    if "5-pin" in t or "5 pin" in t or "din" in t:
        return "5_pin"
    if "trs" in t or "1/8" in t or "3.5mm" in t:
        return "trs"
    if "usb" in t:
        return "usb"
    return "unknown"

def extract_trs_midi_type(text: str) -> TrsMidiType:
    t = text.lower()
    # only meaningful if TRS MIDI exists in text, but safe to parse regardless
    if re.search(r"\btype\s*a\b|\btrs\s*type\s*a\b", t):
        return "type_a"
    if re.search(r"\btype\s*b\b|\btrs\s*type\s*b\b", t):
        return "type_b"
    return "unknown"

def extract_expression(text: str) -> Optional[bool]:
    t = text.lower()
    if re.search(r"\bexpression\b|\bexp\s+in\b|\bexp\b\s*(in|input)\b", t):
        if re.search(r"\bno\s+expression\b|\bexpression\s*:\s*no\b", t):
            return False
        return True
    return None

def extract_tap_tempo(text: str) -> Optional[bool]:
    t = text.lower()
    if "tap tempo" in t or (re.search(r"\btap\b", t) and "tempo" in t):
        if re.search(r"\bno\s+tap\b|\btap\s*:\s*no\b", t):
            return False
        return True
    return None

def extract_top_jacks(text: str) -> Optional[bool]:
    t = text.lower()
    if re.search(r"\btop[\s\-]mounted\b|\btop\s+jacks?\b|\btop\s+jack\b", t):
        return True
    if re.search(r"\bside[\s\-]mounted\b|\bside\s+jacks?\b", t):
        return False
    return None

def extract_bypass(text: str) -> BypassType:
    t = text.lower()
    # be conservative and prefer explicit phrases
    if "true bypass" in t or "true-bypass" in t:
        return "true_bypass"
    if "buffered bypass" in t or re.search(r"\bbuffered\b", t) and "bypass" in t:
        return "buffered"
    # switchable buffer / buffer option / selectable buffer
    if re.search(r"\bswitchable\b.*\bbuffer\b|\bbuffer\s+option\b|\bselectable\b.*\bbuffer\b", t):
        return "switchable"
    return "unknown"

def extract_stereo_flags(text: str) -> Dict[str, Optional[bool]]:
    """
    Handles phrases like:
      - "mono in, stereo out"
      - "stereo out only"
      - "stereo I/O"
      - "dual mono out"
      - "mono only"
    Conservative defaults: don't guess unless explicitly stated.
    """
    t = text.lower()
    flags: Dict[str, Optional[bool]] = {"mono_in": None, "stereo_in": None, "mono_out": None, "stereo_out": None}

    # hard negatives / mono-only
    if re.search(r"\bmono\s+only\b", t):
        flags["mono_in"] = True
        flags["stereo_in"] = False
        flags["mono_out"] = True
        flags["stereo_out"] = False
        return flags

    # explicit stereo I/O
    if re.search(r"\bstereo\s+i/o\b|\bstereo\s+io\b", t):
        flags["stereo_in"] = True
        flags["stereo_out"] = True
        flags["mono_in"] = False
        flags["mono_out"] = False
        return flags

    # common patterns
    if re.search(r"\bmono\s+in\b", t):
        flags["mono_in"] = True
        if flags["stereo_in"] is None:
            flags["stereo_in"] = False

    if re.search(r"\bstereo\s+in\b", t):
        flags["stereo_in"] = True
        flags["mono_in"] = False

    # outputs
    if re.search(r"\bmono\s+out\b", t):
        flags["mono_out"] = True
        if flags["stereo_out"] is None:
            flags["stereo_out"] = False

    if re.search(r"\bstereo\s+out\b|\bstereo\s+output\b", t):
        flags["stereo_out"] = True
        flags["mono_out"] = False

    # "stereo out only" implies input unspecified; do NOT assume stereo in
    if re.search(r"\bstereo\s+out\s+only\b|\bstereo\s+output\s+only\b", t):
        flags["stereo_out"] = True
        flags["mono_out"] = False

    # "mono in / stereo out" shorthand
    if re.search(r"\bmono\s+in\s*[,/]\s*stereo\s+out\b", t):
        flags["mono_in"] = True
        flags["stereo_in"] = False
        flags["stereo_out"] = True
        flags["mono_out"] = False

    # dual mono out (still mono out, not stereo)
    if re.search(r"\bdual\s+mono\b.*\bout\b|\btwo\s+mono\b.*\bout\b", t):
        flags["mono_out"] = True
        if flags["stereo_out"] is None:
            flags["stereo_out"] = False

    return flags

def io_source_patterns(field: str) -> list[str]:
    # patterns tuned for explicit citations
    if field == "mono_in":
        return [r"\bmono\s+in\b", r"\binput\s*:\s*mono\b"]
    if field == "stereo_in":
        return [r"\bstereo\s+in\b", r"\binput\s*:\s*stereo\b", r"\bstereo\s+i/o\b"]
    if field == "mono_out":
        return [r"\bmono\s+out\b", r"\boutput\s*:\s*mono\b", r"\bdual\s+mono\b.*\bout\b"]
    if field == "stereo_out":
        return [r"\bstereo\s+out\b", r"\bstereo\s+output\b", r"\bstereo\s+out\s+only\b", r"\bstereo\s+i/o\b"]
    if field == "top_jacks":
        return [r"\btop[\s\-]mounted\b", r"\btop\s+jacks?\b", r"\bside[\s\-]mounted\b", r"\bside\s+jacks?\b"]
    return []

def extract_dimensions_mm(text: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"width": None, "depth": None}
    t = text.lower()

    m = re.search(r"(?:size|dimensions)\s*[:\-]\s*([^\n]+)", t)
    if m:
        line = m.group(1)
        nums = re.findall(r"(\d+(?:\.\d+)?)\s*(mm|cm|in|inch|inches)\b", line)
        if len(nums) >= 2:
            w_raw = f"{nums[0][0]} {nums[0][1]}"
            d_raw = f"{nums[1][0]} {nums[1][1]}"
            out["width"] = parse_mm_value(w_raw)
            out["depth"] = parse_mm_value(d_raw)
            return out

    mw = re.search(r"\bwidth\s*[:\-]\s*([0-9.]+\s*(?:mm|cm|in|inch|inches))\b", t)
    md = re.search(r"\bdepth\s*[:\-]\s*([0-9.]+\s*(?:mm|cm|in|inch|inches))\b", t)
    if mw:
        out["width"] = parse_mm_value(mw.group(1))
    if md:
        out["depth"] = parse_mm_value(md.group(1))
    return out

# ---------- Main extraction ----------

def extract_one(path: Path) -> PedalRecord:
    raw = path.read_text(encoding="utf-8", errors="replace")
    name, brand = parse_name_brand(raw)
    inferred_name = name or path.stem.replace("_", " ").title()

    rec = PedalRecord(
        id=path.stem,
        name=inferred_name,
        brand=brand,
        category=guess_category(raw),
        source_file=str(path),
    )

    # Voltage
    mv = find_first(
        [r"\b(\d+(?:\.\d+)?)\s*v(?:dc)?\b", r"\bpower\s*[:\-]\s*(\d+(?:\.\d+)?)\s*v\b"],
        raw,
    )
    if mv:
        rec.power.voltage_v = float(mv.group(1))
        add_source(rec.sources, "power.voltage_v", raw, mv)

    # Current
    mc = find_first(
        [r"\b(\d{2,4})\s*ma\b", r"\bcurrent\s+draw\s*[:\-]\s*(\d{2,4})\s*ma\b", r"\bcurrent\s*:\s*(\d{2,4})\s*ma\b"],
        raw,
    )
    if mc:
        rec.power.current_ma = int(mc.group(1))
        add_source(rec.sources, "power.current_ma", raw, mc)

    # Polarity
    pol = extract_polarity(raw)
    rec.power.polarity = pol
    if pol != "unknown":
        add_source_first(
            rec.sources,
            "power.polarity",
            raw,
            [r"center[\s\-]negative", r"center[\s\-]positive", r"centre[\s\-]negative", r"centre[\s\-]positive"],
        )

    # MIDI
    midi = extract_midi(raw)
    if midi is not None:
        rec.control.midi = midi
        add_source_first(rec.sources, "control.midi", raw, [r"\bmidi\b", r"\bno\s+midi\b"])

    mt = extract_midi_type(raw)
    rec.control.midi_type = mt
    if mt != "unknown":
        add_source_first(rec.sources, "control.midi_type", raw, [r"5[\s\-]?pin", r"\bdin\b", r"\btrs\b", r"3\.5mm", r"1/8", r"\busb\b"])

    # TRS MIDI Type A/B
    trs_type = extract_trs_midi_type(raw)
    rec.control.trs_midi_type = trs_type
    if trs_type != "unknown":
        add_source_first(rec.sources, "control.trs_midi_type", raw, [r"\btype\s*a\b", r"\btype\s*b\b", r"\btrs\s*type\s*a\b", r"\btrs\s*type\s*b\b"])

    # Expression
    exp = extract_expression(raw)
    if exp is not None:
        rec.control.expression = exp
        add_source_first(rec.sources, "control.expression", raw, [r"\bexpression\b", r"\bexp\s+in\b", r"\bexp\s*input\b"])

    # Tap tempo
    tap = extract_tap_tempo(raw)
    if tap is not None:
        rec.control.tap_tempo = tap
        add_source_first(rec.sources, "control.tap_tempo", raw, [r"tap\s+tempo", r"\btap\b"])

    # IO stereo/mono
    stereo_flags = extract_stereo_flags(raw)
    for k, v in stereo_flags.items():
        setattr(rec.io, k, v)
        if v is not None:
            pats = io_source_patterns(k)
            if pats:
                add_source_first(rec.sources, f"io.{k}", raw, pats)

    # Top jacks
    tj = extract_top_jacks(raw)
    if tj is not None:
        rec.io.top_jacks = tj
        add_source_first(rec.sources, "io.top_jacks", raw, io_source_patterns("top_jacks"))

    # Bypass type
    bp = extract_bypass(raw)
    rec.bypass = bp
    if bp != "unknown":
        add_source_first(
            rec.sources,
            "bypass",
            raw,
            [r"true[\s\-]bypass", r"buffered\s+bypass", r"\bswitchable\b.*\bbuffer\b", r"\bbuffer\s+option\b"],
        )

    # Dimensions
    dims = extract_dimensions_mm(raw)
    width_val = dims.get("width")
    if width_val is not None:
        rec.size_mm.width = float(width_val)
        add_source_first(rec.sources, "size_mm.width", raw, [r"\bsize\b", r"\bdimensions\b", r"\bwidth\b"])

    depth_val = dims.get("depth")
    if depth_val is not None:
        rec.size_mm.depth = float(depth_val)
        add_source_first(rec.sources, "size_mm.depth", raw, [r"\bsize\b", r"\bdimensions\b", r"\bdepth\b"])

    return rec

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/pedals", help="Directory of pedal note .txt files")
    ap.add_argument("--out_dir", default="out", help="Output directory")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / "pedals.jsonl"

    paths = sorted(p for p in data_dir.glob("*.txt") if p.is_file())
    if not paths:
        raise SystemExit(f"No .txt files found in {data_dir}")

    with records_path.open("w", encoding="utf-8") as f:
        for p in paths:
            rec = extract_one(p)
            f.write(rec.model_dump_json())
            f.write("\n")

    print(f"Wrote {len(paths)} records to {records_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())