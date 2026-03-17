from __future__ import annotations
from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field

Category = Literal[
    "delay", "reverb", "drive", "modulation", "multi_fx", "utility", "tuner", "looper", "unknown"
]

Polarity = Literal["center_negative", "center_positive", "unknown"]
MidiType = Literal["5_pin", "trs", "usb", "unknown"]
TrsMidiType = Literal["type_a", "type_b", "unknown"]
BypassType = Literal["true_bypass", "buffered", "switchable", "unknown"]

class Power(BaseModel):
    voltage_v: Optional[float] = None
    current_ma: Optional[int] = None
    polarity: Polarity = "unknown"

class IO(BaseModel):
    mono_in: Optional[bool] = None
    stereo_in: Optional[bool] = None
    mono_out: Optional[bool] = None
    stereo_out: Optional[bool] = None

    # musician-friendly
    top_jacks: Optional[bool] = None

class Control(BaseModel):
    midi: Optional[bool] = None
    midi_type: MidiType = "unknown"

    # only meaningful when midi_type == "trs"
    trs_midi_type: TrsMidiType = "unknown"

    expression: Optional[bool] = None
    tap_tempo: Optional[bool] = None

class SizeMM(BaseModel):
    width: Optional[float] = None
    depth: Optional[float] = None

class PedalRecord(BaseModel):
    id: str
    name: str
    brand: Optional[str] = None
    category: Category = "unknown"

    power: Power = Field(default_factory=Power)
    io: IO = Field(default_factory=IO)
    control: Control = Field(default_factory=Control)
    size_mm: SizeMM = Field(default_factory=SizeMM)

    # musician-friendly
    bypass: BypassType = "unknown"

    # field_path -> snippet
    sources: Dict[str, str] = Field(default_factory=dict)

    source_file: Optional[str] = None