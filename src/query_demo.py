from __future__ import annotations

import argparse
import re
from typing import List

from src.schema import PedalRecord
from src.llm_answer import ollama_narrate


def load_records(path: str) -> List[PedalRecord]:
    out: List[PedalRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(PedalRecord.model_validate_json(line))
    return out


def print_shortlist(title: str, items: List[PedalRecord], max_items: int = 50) -> None:
    print(f"\n=== {title} ({len(items)}) ===")
    for r in items[:max_items]:
        v = f"{r.power.voltage_v:.0f}V" if r.power.voltage_v is not None else "V?"
        ma = f"{r.power.current_ma}mA" if r.power.current_ma is not None else "mA?"
        w = f"{r.size_mm.width:.0f}mm" if r.size_mm.width is not None else "w?"
        midi = f"MIDI({r.control.midi_type})" if r.control.midi else "no-midi/unk"
        stereo = "st-out" if r.io.stereo_out else ("mono" if r.io.mono_out else "out?")
        top = "top" if r.io.top_jacks else ("side" if r.io.top_jacks is False else "jacks?")
        bp = r.bypass
        print(
            f"- {r.id:18s} | {r.name:22s} | {r.category:8s} | {v:3s} {ma:6s} | "
            f"{midi:12s} | {stereo:6s} | {top:6s} | {bp:10s} | width {w}"
        )


def power_check(records: List[PedalRecord], pedal_id: str, supply_ma: int = 100, voltage_v: float = 9.0) -> None:
    rec = next((r for r in records if r.id == pedal_id), None)
    if not rec:
        print(f"[power_check] pedal_id not found: {pedal_id}")
        return

    print(f"\n=== Power check: {rec.name} ===")
    if rec.power.voltage_v is None or rec.power.current_ma is None:
        print("Not enough extracted info to decide (voltage/current missing).")
        return

    ok = (abs(rec.power.voltage_v - voltage_v) < 0.01) and (rec.power.current_ma <= supply_ma)
    verdict = "OK" if ok else "NO"
    print(f"Supply: {voltage_v:.0f}V {supply_ma}mA  |  Pedal: {rec.power.voltage_v:.0f}V {rec.power.current_ma}mA  =>  {verdict}")

    src_v = rec.sources.get("power.voltage_v")
    src_c = rec.sources.get("power.current_ma")
    if src_v:
        print(f"  source voltage: {src_v}")
    if src_c:
        print(f"  source current: {src_c}")


# ---------- Mini assistant: parse question -> constraints ----------

def parse_question(q: str) -> dict:
    ql = q.lower()
    c: dict = {}

    # category hints
    for cat in ["reverb", "delay", "drive", "tuner", "looper", "utility", "modulation", "multi"]:
        if cat in ql:
            c["category"] = "multi_fx" if cat == "multi" else cat
            break

    # booleans
    if "midi" in ql:
        if re.search(r"\bno\s+midi\b|\bwithout\s+midi\b", ql):
            c["midi"] = False
        else:
            c["midi"] = True

    if re.search(r"\bstereo\s+i/o\b|\bstereo\s+io\b", ql):
        c["stereo_in"] = True
        c["stereo_out"] = True
    else:
        if "stereo" in ql and ("out" in ql or "output" in ql):
            c["stereo_out"] = True
        if "stereo" in ql and ("in" in ql or "input" in ql):
            c["stereo_in"] = True

    if "expression" in ql or re.search(r"\bexp\b", ql):
        c["expression"] = True

    if "tap tempo" in ql or ("tap" in ql and "tempo" in ql):
        c["tap_tempo"] = True

    if "top jacks" in ql or "top-mounted" in ql:
        c["top_jacks"] = True

    if "true bypass" in ql:
        c["bypass"] = "true_bypass"
    if "buffered" in ql and "bypass" in ql:
        c["bypass"] = "buffered"

    if "type a" in ql:
        c["trs_midi_type"] = "type_a"
    if "type b" in ql:
        c["trs_midi_type"] = "type_b"

    # numeric constraints
    mv = re.search(r"\b(\d+(?:\.\d+)?)\s*v\b", ql)
    if mv:
        c["voltage_v"] = float(mv.group(1))

    mma = re.search(r"\b(\d{2,4})\s*ma\b", ql)
    if mma:
        if re.search(r"\bunder\b|\b<=\b|\bless\b|\bmax\b", ql):
            c["max_current_ma"] = int(mma.group(1))

    mw = re.search(r"\bunder\s*(\d+(?:\.\d+)?)\s*mm\b|\b<=\s*(\d+(?:\.\d+)?)\s*mm\b", ql)
    if mw:
        val = float(mw.group(1) or mw.group(2))
        c["max_width_mm"] = val

    return c


def apply_constraints(records: List[PedalRecord], c: dict) -> List[PedalRecord]:
    out: List[PedalRecord] = []
    for r in records:
        if "category" in c and r.category != c["category"]:
            continue

        if "midi" in c:
            if r.control.midi is None:
                continue
            if r.control.midi != c["midi"]:
                continue

        if c.get("trs_midi_type") and r.control.trs_midi_type != c["trs_midi_type"]:
            continue

        if c.get("bypass") and r.bypass != c["bypass"]:
            continue

        if c.get("stereo_out") is True and r.io.stereo_out is not True:
            continue
        if c.get("stereo_in") is True and r.io.stereo_in is not True:
            continue

        if c.get("expression") is True and r.control.expression is not True:
            continue
        if c.get("tap_tempo") is True and r.control.tap_tempo is not True:
            continue
        if c.get("top_jacks") is True and r.io.top_jacks is not True:
            continue

        if "voltage_v" in c:
            if r.power.voltage_v is None or abs(r.power.voltage_v - c["voltage_v"]) > 0.01:
                continue

        if "max_current_ma" in c:
            if r.power.current_ma is None or r.power.current_ma > c["max_current_ma"]:
                continue

        if "max_width_mm" in c:
            if r.size_mm.width is None or r.size_mm.width > c["max_width_mm"]:
                continue

        out.append(r)

    return out


# ---------- Original demo filters ----------

def filter_midi_stereo_9v(records: List[PedalRecord]) -> List[PedalRecord]:
    out: List[PedalRecord] = []
    for r in records:
        if r.control.midi is not True:
            continue
        if r.io.stereo_out is not True:
            continue
        if r.power.voltage_v is None or abs(r.power.voltage_v - 9.0) > 0.01:
            continue
        out.append(r)
    return out


def filter_reverbs_expression_width(records: List[PedalRecord], max_width_mm: float = 125.0) -> List[PedalRecord]:
    out: List[PedalRecord] = []
    for r in records:
        if r.category != "reverb":
            continue
        if r.control.expression is not True:
            continue
        if r.size_mm.width is None:
            continue
        if r.size_mm.width <= max_width_mm:
            out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="out/pedals.jsonl")
    ap.add_argument("--power_pedal_id", default="nexus_multi_fx", help="Pedal id for power demo")

    ap.add_argument("--question", default=None, help='Natural language question, e.g. "reverbs with expression under 125mm"')

    # LLM narration (Ollama)
    ap.add_argument("--ollama", action="store_true", help="Use Ollama to narrate results (filtering remains deterministic)")
    ap.add_argument("--model", default="llama3.2:3b", help="Ollama model name, e.g. llama3.2:3b or llama3.1:8b")
    ap.add_argument("--ollama_url", default="http://127.0.0.1:11434", help="Ollama base URL")

    args = ap.parse_args()
    records = load_records(args.records)

    if args.question:
        c = parse_question(args.question)
        results = apply_constraints(records, c)

        # Always print shortlist (useful even if LLM fails)
        print_shortlist(f'Question: "{args.question}"', results)

        if args.ollama:
            print("\n=== LLM narration (Ollama) ===")
            try:
                answer = ollama_narrate(
                    args.question,
                    results,
                    model=args.model,
                    base_url=args.ollama_url,
                )
                print(answer)
            except Exception as e:
                print(f"[ollama] error: {e}")

        return 0

    # Default demo mode
    power_check(records, args.power_pedal_id, supply_ma=100, voltage_v=9.0)

    shortlist = filter_midi_stereo_9v(records)
    print_shortlist("MIDI + stereo out + 9V", shortlist)

    shortlist2 = filter_reverbs_expression_width(records, max_width_mm=125.0)
    print_shortlist("Reverbs + expression + width<=125mm", shortlist2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())