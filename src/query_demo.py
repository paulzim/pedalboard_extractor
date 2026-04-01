from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

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
    print(
        f"Supply: {voltage_v:.0f}V {supply_ma}mA  |  Pedal: {rec.power.voltage_v:.0f}V {rec.power.current_ma}mA  =>  {verdict}"
    )

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

    # category hints (improved: avoid forcing a single category when the prompt implies multiple effects)
    cats = set()

    # reverb-ish
    if ("reverb" in ql) or ("washy" in ql) or ("wash" in ql) or ("ambient" in ql) or ("shimmer" in ql):
        cats.add("reverb")

    # delay-ish
    if ("delay" in ql) or ("echo" in ql) or ("repeats" in ql):
        cats.add("delay")

    # drive-ish
    if (
        ("drive" in ql)
        or ("distort" in ql)
        or ("distortion" in ql)
        or ("gain" in ql)
        or ("fuzz" in ql)
        or ("overdrive" in ql)
    ):
        cats.add("drive")

    # other categories
    if "tuner" in ql:
        cats.add("tuner")
    if ("looper" in ql) or ("loop" in ql):
        cats.add("looper")
    if (
        ("modulation" in ql)
        or ("chorus" in ql)
        or ("phaser" in ql)
        or ("flanger" in ql)
        or ("tremolo" in ql)
    ):
        cats.add("modulation")

    # multi-fx hint
    if "multi" in ql and ("fx" in ql or "effects" in ql):
        cats.add("multi_fx")

    # Only set category if it is unambiguous (exactly one category).
    # If user implies multiple (e.g., drive + delay + reverb), do NOT hard constrain.
    if len(cats) == 1:
        c["category"] = next(iter(cats))

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

    if "top jacks" in ql or "top-mounted" in ql or "top mounted" in ql:
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
        # interpret as max current unless question suggests otherwise
        if re.search(r"\bunder\b|\b<=\b|\bless\b|\bmax\b", ql):
            c["max_current_ma"] = int(mma.group(1))

    mw = re.search(r"\bunder\s*(\d+(?:\.\d+)?)\s*mm\b|\b<=\s*(\d+(?:\.\d+)?)\s*mm\b", ql)
    if mw:
        val = float(mw.group(1) or mw.group(2))
        c["max_width_mm"] = val

    return c


# ---------- Explainable filtering ----------

@dataclass(frozen=True)
class EvalResult:
    passed: bool
    failures: List[str]           # human-readable reasons
    first_failure: Optional[str]  # first failing rule (for quick scan)


def _cmp_float(a: Optional[float], b: float, tol: float = 0.01) -> bool:
    return a is not None and abs(a - b) <= tol


def evaluate_constraints(r: PedalRecord, c: dict) -> EvalResult:
    failures: List[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)

    # Category
    if "category" in c and r.category != c["category"]:
        fail(f"category != {c['category']} (got {r.category})")

    # MIDI required/forbidden
    if "midi" in c:
        if r.control.midi is None:
            fail(f"midi unknown (need {c['midi']})")
        elif r.control.midi != c["midi"]:
            fail(f"midi != {c['midi']} (got {r.control.midi})")

    # TRS MIDI type
    if c.get("trs_midi_type"):
        if r.control.trs_midi_type == "unknown":
            fail(f"trs_midi_type unknown (need {c['trs_midi_type']})")
        elif r.control.trs_midi_type != c["trs_midi_type"]:
            fail(f"trs_midi_type != {c['trs_midi_type']} (got {r.control.trs_midi_type})")

    # Bypass
    if c.get("bypass"):
        if r.bypass == "unknown":
            fail(f"bypass unknown (need {c['bypass']})")
        elif r.bypass != c["bypass"]:
            fail(f"bypass != {c['bypass']} (got {r.bypass})")

    # Stereo requirements
    if c.get("stereo_out") is True and r.io.stereo_out is not True:
        got = r.io.stereo_out
        fail(f"stereo_out not true (got {got})")
    if c.get("stereo_in") is True and r.io.stereo_in is not True:
        got = r.io.stereo_in
        fail(f"stereo_in not true (got {got})")

    # Expression / tap / top jacks
    if c.get("expression") is True and r.control.expression is not True:
        got = r.control.expression
        fail(f"expression not true (got {got})")
    if c.get("tap_tempo") is True and r.control.tap_tempo is not True:
        got = r.control.tap_tempo
        fail(f"tap_tempo not true (got {got})")
    if c.get("top_jacks") is True and r.io.top_jacks is not True:
        got = r.io.top_jacks
        fail(f"top_jacks not true (got {got})")

    # Voltage
    if "voltage_v" in c:
        if r.power.voltage_v is None:
            fail(f"voltage unknown (need {c['voltage_v']}V)")
        elif not _cmp_float(r.power.voltage_v, float(c["voltage_v"])):
            fail(f"voltage != {c['voltage_v']}V (got {r.power.voltage_v}V)")

    # Max current draw
    if "max_current_ma" in c:
        if r.power.current_ma is None:
            fail(f"current draw unknown (need <= {c['max_current_ma']}mA)")
        elif r.power.current_ma > int(c["max_current_ma"]):
            fail(f"current draw > {c['max_current_ma']}mA (got {r.power.current_ma}mA)")

    # Max width
    if "max_width_mm" in c:
        if r.size_mm.width is None:
            fail(f"width unknown (need <= {c['max_width_mm']}mm)")
        elif r.size_mm.width > float(c["max_width_mm"]):
            fail(f"width > {c['max_width_mm']}mm (got {r.size_mm.width}mm)")

    passed = len(failures) == 0
    return EvalResult(passed=passed, failures=failures, first_failure=(failures[0] if failures else None))


def apply_constraints(records: List[PedalRecord], c: dict) -> List[PedalRecord]:
    out: List[PedalRecord] = []
    for r in records:
        if evaluate_constraints(r, c).passed:
            out.append(r)
    return out


def explain_constraints(records: List[PedalRecord], c: dict, max_items: int = 200) -> None:
    print("\n=== Explain mode ===")
    print("Constraints:")
    for k in sorted(c.keys()):
        print(f"- {k}: {c[k]}")
    print("")

    rows: List[Tuple[str, str, str, str]] = []
    passed_count = 0

    for r in records:
        ev = evaluate_constraints(r, c)
        if ev.passed:
            passed_count += 1
            continue

        all_reasons = "; ".join(ev.failures)
        rows.append((r.id, r.name, ev.first_failure or "", all_reasons))

    print(f"Matched: {passed_count} / {len(records)}")
    print(f"Eliminated: {len(records) - passed_count} / {len(records)}\n")

    if not rows:
        print("Nothing eliminated — all records matched the constraints (unlikely, but possible).")
        return

    rows = rows[:max_items]

    id_w = max(len("id"), max(len(x[0]) for x in rows))
    name_w = max(len("name"), max(len(x[1]) for x in rows))
    first_w = max(len("first_failure"), min(48, max(len(x[2]) for x in rows)))

    header = f"{'id'.ljust(id_w)}  {'name'.ljust(name_w)}  {'first_failure'.ljust(first_w)}  reasons"
    print(header)
    print("-" * len(header))

    for pid, pname, first, reasons in rows:
        first_short = (first[:45] + "…") if len(first) > 48 else first
        print(f"{pid.ljust(id_w)}  {pname.ljust(name_w)}  {first_short.ljust(first_w)}  {reasons}")

    print("\nTip: This output is great for demos. You can literally show why a pedal was filtered out.")


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

    ap.add_argument("--explain", action="store_true", help="Print why each pedal did/didn't match constraints")

    ap.add_argument("--ollama", action="store_true", help="Use Ollama to narrate results (filtering remains deterministic)")
    ap.add_argument("--model", default="llama3.2:3b", help="Ollama model name, e.g. llama3.2:3b")
    ap.add_argument("--ollama_url", default="http://127.0.0.1:11434", help="Ollama base URL")

    args = ap.parse_args()
    records = load_records(args.records)

    if args.question:
        c = parse_question(args.question)

        if args.explain:
            explain_constraints(records, c)

        results = apply_constraints(records, c)
        print_shortlist(f'Question: "{args.question}"', results)

        if args.ollama:
            print("\n=== LLM narration (Ollama) ===")
            try:
                answer = ollama_narrate(
                    question=args.question,
                    matches=results,
                    model=args.model,
                    base_url=args.ollama_url,
                )
                print(answer)
            except Exception as e:
                print(f"[ollama] error: {e}")

        return 0

    power_check(records, args.power_pedal_id, supply_ma=100, voltage_v=9.0)

    shortlist = filter_midi_stereo_9v(records)
    print_shortlist("MIDI + stereo out + 9V", shortlist)

    shortlist2 = filter_reverbs_expression_width(records, max_width_mm=125.0)
    print_shortlist("Reverbs + expression + width<=125mm", shortlist2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())