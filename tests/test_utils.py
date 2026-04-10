from __future__ import annotations

import pytest

from src.utils import guess_category, parse_mm_value, parse_name_brand


def test_parse_mm_value_normalizes_common_units() -> None:
    assert parse_mm_value("125 mm") == pytest.approx(125.0)
    assert parse_mm_value("12.5 cm") == pytest.approx(125.0)
    assert parse_mm_value("4.9 in") == pytest.approx(124.46)
    assert parse_mm_value("wide-ish") is None


def test_parse_name_brand_reads_header_lines() -> None:
    text = """
    Name: Aurora Delay
    Brand: NightSky Audio

    Big modulated delay notes go here.
    """

    assert parse_name_brand(text) == ("Aurora Delay", "NightSky Audio")


def test_guess_category_prefers_obvious_pedal_family() -> None:
    assert guess_category("lush ambient shimmer reverb") == "reverb"
    assert guess_category("tape echo delay repeats") == "delay"
    assert guess_category("transparent overdrive with grit") == "drive"
