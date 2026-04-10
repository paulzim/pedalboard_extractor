from __future__ import annotations

from pathlib import Path

import pytest

from src.extract import extract_one


def test_extract_one_normalizes_core_fields_and_sources(tmp_path: Path) -> None:
    note = tmp_path / "aurora_delay.txt"
    note.write_text(
        """
        Name: Aurora Delay
        Brand: NightSky Audio

        Washy stereo delay with trails.
        Power: 9V DC, 75mA, center-negative.
        I/O: mono in / stereo out.
        Control: MIDI via TRS Type B, expression input, and tap tempo.
        Top-mounted jacks. Buffered bypass.
        Dimensions: 4.9 in x 2.5 in.
        """,
        encoding="utf-8",
    )

    rec = extract_one(note)

    assert rec.id == "aurora_delay"
    assert rec.name == "Aurora Delay"
    assert rec.brand == "NightSky Audio"
    assert rec.category == "delay"
    assert rec.power.voltage_v == 9.0
    assert rec.power.current_ma == 75
    assert rec.power.polarity == "center_negative"
    assert rec.io.mono_in is True
    assert rec.io.stereo_out is True
    assert rec.control.midi is True
    assert rec.control.midi_type == "trs"
    assert rec.control.trs_midi_type == "type_b"
    assert rec.control.expression is True
    assert rec.control.tap_tempo is True
    assert rec.io.top_jacks is True
    assert rec.bypass == "buffered"
    assert rec.size_mm.width == pytest.approx(124.46)
    assert rec.size_mm.depth == pytest.approx(63.5)

    assert "75mA" in rec.sources["power.current_ma"]
    assert "stereo out" in rec.sources["io.stereo_out"].lower()
    assert "Type B" in rec.sources["control.trs_midi_type"]
