from __future__ import annotations

import streamlit as st
from typing import List, Dict

from src.query_demo import load_records, parse_question, apply_constraints
from src.schema import PedalRecord

st.set_page_config(page_title="Pedalboard Extractor Demo", layout="wide")


@st.cache_data
def cached_records(path: str) -> List[PedalRecord]:
    return load_records(path)


def record_row(r: PedalRecord) -> Dict:
    return {
        "id": r.id,
        "name": r.name,
        "category": r.category,
        "voltage_v": r.power.voltage_v,
        "current_ma": r.power.current_ma,
        "polarity": r.power.polarity,
        "stereo_out": r.io.stereo_out,
        "midi": r.control.midi,
        "midi_type": r.control.midi_type,
        "trs_midi_type": r.control.trs_midi_type,
        "expression": r.control.expression,
        "tap_tempo": r.control.tap_tempo,
        "top_jacks": r.io.top_jacks,
        "bypass": r.bypass,
        "width_mm": r.size_mm.width,
        "depth_mm": r.size_mm.depth,
    }


def show_sources(r: PedalRecord):
    st.markdown("**Sources (snippets)**")
    if not r.sources:
        st.caption("No snippets captured.")
        return
    for k in sorted(r.sources.keys()):
        st.write(f"- `{k}`: {r.sources[k]}")


st.title("Pedalboard Extractor Demo")
records_path = st.sidebar.text_input("Records path", value="out/pedals.jsonl")
records = cached_records(records_path)

tabs = st.tabs(["Search (Question Mode)", "Board Builder", "Browse Data"])

# --- Tab 1: Question mode ---
with tabs[0]:
    st.subheader("Ask a constraint-style question")
    q = st.text_input(
        'Try: "reverbs with expression under 125mm" or "midi stereo out 9v" or "top jacks true bypass"'
    )
    if q:
        c = parse_question(q)
        results = apply_constraints(records, c)
        st.caption(f"Parsed constraints: {c}")
        st.write(f"Matches: {len(results)}")

        if results:
            st.dataframe([record_row(r) for r in results], use_container_width=True)

            st.markdown("### Inspect sources")
            pick = st.selectbox("Pick a result", options=[r.id for r in results])
            rr = next(r for r in results if r.id == pick)
            show_sources(rr)
    else:
        st.info("Enter a question above to route it to structured filters.")

# --- Tab 2: Board builder ---
with tabs[1]:
    st.subheader("Board Builder")
    st.caption("Pick pedals → see power budget + MIDI chain + quick red flags.")

    options = {r.id: f"{r.name} ({r.category})" for r in records}
    selected_ids = st.multiselect("Select pedals", options=list(options.keys()), format_func=lambda x: options[x])
    selected = [r for r in records if r.id in selected_ids]

    col1, col2, col3 = st.columns(3)

    total_ma = sum((r.power.current_ma or 0) for r in selected)
    unknown_current = [r for r in selected if r.power.current_ma is None]

    with col1:
        st.metric("Total current draw (mA)", total_ma)
        if unknown_current:
            st.warning(f"{len(unknown_current)} pedal(s) missing current draw: " + ", ".join(r.id for r in unknown_current))

    with col2:
        midi_pedals = [r for r in selected if r.control.midi]
        st.metric("MIDI-capable pedals", len(midi_pedals))
        if midi_pedals:
            st.write("MIDI list:")
            for r in midi_pedals:
                st.write(f"- {r.name} — {r.control.midi_type} ({r.control.trs_midi_type})")

    with col3:
        stereo_out = [r for r in selected if r.io.stereo_out]
        st.metric("Stereo-out pedals", len(stereo_out))

    st.markdown("### Selected pedals")
    if selected:
        st.dataframe([record_row(r) for r in selected], use_container_width=True)

        st.markdown("### Inspect sources for a selected pedal")
        pick2 = st.selectbox("Pick a pedal", options=[r.id for r in selected])
        rr2 = next(r for r in selected if r.id == pick2)
        show_sources(rr2)
    else:
        st.info("Select pedals to build a board.")

# --- Tab 3: Browse ---
with tabs[2]:
    st.subheader("All extracted records")
    st.dataframe([record_row(r) for r in records], use_container_width=True)

    st.markdown("### Inspect sources")
    pick3 = st.selectbox("Pick any pedal", options=[r.id for r in records], key="pick_all")
    rr3 = next(r for r in records if r.id == pick3)
    show_sources(rr3)