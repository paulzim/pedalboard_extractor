from __future__ import annotations

from typing import Optional

from src.query_demo import apply_constraints, evaluate_constraints, parse_question
from src.schema import PedalRecord


def make_record(
    *,
    pedal_id: str,
    name: str,
    category: str = "delay",
    voltage_v: Optional[float] = 9.0,
    current_ma: Optional[int] = 75,
    midi: Optional[bool] = True,
    stereo_out: Optional[bool] = True,
    trs_midi_type: str = "type_b",
) -> PedalRecord:
    rec = PedalRecord(id=pedal_id, name=name, category=category)
    rec.power.voltage_v = voltage_v
    rec.power.current_ma = current_ma
    rec.control.midi = midi
    rec.control.trs_midi_type = trs_midi_type
    rec.io.stereo_out = stereo_out
    return rec


def test_parse_question_extracts_specs_without_llm() -> None:
    constraints = parse_question("midi stereo out 9v under 100mA type b")

    assert constraints == {
        "midi": True,
        "stereo_out": True,
        "trs_midi_type": "type_b",
        "voltage_v": 9.0,
        "max_current_ma": 100,
    }


def test_parse_question_does_not_overconstrain_multi_effect_prompt() -> None:
    constraints = parse_question(
        "I want a distorted, washy delay tone. Stereo if possible. I use 9V isolated power."
    )

    assert constraints["voltage_v"] == 9.0
    assert "category" not in constraints


def test_evaluate_constraints_reports_explainable_failure() -> None:
    rec = make_record(
        pedal_id="mono_delay",
        name="Mono Delay",
        current_ma=120,
        stereo_out=False,
    )
    constraints = parse_question("midi stereo out 9v under 100mA")

    result = evaluate_constraints(rec, constraints)

    assert result.passed is False
    assert result.first_failure == "stereo_out not true (got False)"
    assert "current draw > 100mA (got 120mA)" in result.failures


def test_apply_constraints_returns_only_matching_records() -> None:
    matching = make_record(pedal_id="aurora_delay", name="Aurora Delay")
    too_power_hungry = make_record(
        pedal_id="big_delay",
        name="Big Delay",
        current_ma=250,
    )
    mono_only = make_record(
        pedal_id="mono_delay",
        name="Mono Delay",
        stereo_out=False,
    )

    results = apply_constraints(
        [matching, too_power_hungry, mono_only],
        parse_question("midi stereo out 9v under 100mA"),
    )

    assert [r.id for r in results] == ["aurora_delay"]
