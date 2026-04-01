from __future__ import annotations

import streamlit as st
from typing import List, Dict

from src.query_demo import load_records, parse_question, apply_constraints
from src.schema import PedalRecord
from src.llm_answer import ollama_narrate

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


def build_structured_context(
    selected: List[PedalRecord], preferences: str = "", constraints: str = ""
) -> str:
    lines: List[str] = []
    lines.append("Known pedal candidates:")

    for i, r in enumerate(selected, start=1):
        pedal_lines = [f"{i}. {r.name}"]
        pedal_lines.append(f"   Category: {r.category}")

        power_bits: List[str] = []
        if r.power.voltage_v is not None:
            power_bits.append(f"{r.power.voltage_v:.0f}V")
        if r.power.polarity != "unknown":
            power_bits.append(r.power.polarity.replace("_", "-"))
        if r.power.current_ma is not None:
            power_bits.append(f"{r.power.current_ma}mA")
        if power_bits:
            pedal_lines.append(f"   Power: {', '.join(power_bits)}")

        io_bits: List[str] = []
        if r.io.mono_in:
            io_bits.append("mono in")
        if r.io.stereo_in:
            io_bits.append("stereo in")
        if r.io.mono_out:
            io_bits.append("mono out")
        if r.io.stereo_out:
            io_bits.append("stereo out")
        if io_bits:
            pedal_lines.append(f"   I/O: {', '.join(io_bits)}")

        control_bits: List[str] = []
        if r.control.midi:
            if r.control.midi_type != "unknown":
                midi_label = f"MIDI ({r.control.midi_type})"
                if r.control.trs_midi_type != "unknown":
                    midi_label += f" {r.control.trs_midi_type}"
                control_bits.append(midi_label)
            else:
                control_bits.append("MIDI")
        if r.control.expression:
            control_bits.append("expression")
        if r.control.tap_tempo:
            control_bits.append("tap tempo")
        if control_bits:
            pedal_lines.append(f"   Control: {', '.join(control_bits)}")

        if r.size_mm.width is not None and r.size_mm.depth is not None:
            pedal_lines.append(f"   Size: {r.size_mm.width:.0f}mm x {r.size_mm.depth:.0f}mm")

        lines.extend(pedal_lines)

    total_ma = sum((r.power.current_ma or 0) for r in selected)

    if preferences.strip():
        lines.append("")
        lines.append("User preferences:")
        lines.append(f"- {preferences.strip()}")

    if constraints.strip():
        lines.append("")
        lines.append("Constraints:")
        lines.append(f"- {constraints.strip()}")

    if selected:
        lines.append("")
        lines.append("Planning notes:")
        lines.append(f"- Total current draw: {total_ma}mA")
        lines.append("- Use isolated 9V outputs where needed")

    lines.append("")
    lines.append("Task:")
    lines.append("- Recommend a coherent signal chain and explain the reasoning.")

    return "\n".join(lines)


def build_raw_context(raw_request: str, selected: List[PedalRecord]) -> str:
    lines: List[str] = [raw_request.strip()]

    if selected:
        lines.append("")
        lines.append("Reference notes:")
        for r in selected:
            note_bits: List[str] = []
            if r.category:
                note_bits.append(r.category)
            if r.power.current_ma is not None:
                note_bits.append(f"{r.power.current_ma}mA")
            if r.io.stereo_out:
                note_bits.append("stereo out")
            if r.control.expression:
                note_bits.append("expression")
            if r.control.midi:
                note_bits.append("MIDI")
            suffix = f" ({', '.join(note_bits)})" if note_bits else ""
            lines.append(f"- {r.name}{suffix}")

    return "\n".join(lines)


def ollama_controls(scope_key: str) -> Dict[str, object]:
    """
    Shared controls for Ollama usage across tabs.
    Returns a dict containing:
      use_llm (bool), model (str), base_url (str)
    """
    with st.expander("LLM narration (Ollama)", expanded=False):
        use_llm = st.checkbox("Use Ollama to narrate", value=False, key=f"{scope_key}_use_ollama")
        model = st.text_input("Ollama model", value="llama3.2:3b", key=f"{scope_key}_ollama_model")
        base_url = st.text_input("Ollama base URL", value="http://127.0.0.1:11434", key=f"{scope_key}_ollama_url")
        st.caption("Tip: `ollama pull llama3.2:3b` (or `llama3.1:8b`). Filtering stays deterministic; Ollama just narrates.")
    return {"use_llm": use_llm, "model": model, "base_url": base_url}


st.title("Pedalboard Extractor Demo")
records_path = st.sidebar.text_input("Records path", value="out/pedals.jsonl")
records = cached_records(records_path)

tabs = st.tabs(
    [
        "Search (Question Mode)",
        "Board Builder",
        "Prompt Comparison",
        "Browse Data",
    ]
)

# --- Tab 1: Question mode ---
with tabs[0]:
    st.subheader("Ask a constraint-style question")
    q = st.text_input(
        'Try: "reverbs with expression under 125mm" or "midi stereo out 9v" or "top jacks true bypass"',
        key="question_mode_input",
    )

    llm_cfg = ollama_controls("question_mode")

    if q:
        c = parse_question(q)
        results = apply_constraints(records, c)
        st.caption(f"Parsed constraints: {c}")
        st.write(f"Matches: {len(results)}")

        if results:
            st.dataframe([record_row(r) for r in results], use_container_width=True)

            st.markdown("### Inspect sources")
            pick = st.selectbox("Pick a result", options=[r.id for r in results], key="question_mode_pick")
            rr = next(r for r in results if r.id == pick)
            show_sources(rr)
        else:
            st.info("No matches. Try relaxing one constraint (e.g., drop width limit or MIDI requirement).")

        if llm_cfg["use_llm"]:
            st.markdown("### LLM narration")
            with st.spinner("Asking Ollama..."):
                try:
                    ans = ollama_narrate(
                        question=q,
                        matches=results,
                        model=str(llm_cfg["model"]),
                        base_url=str(llm_cfg["base_url"]),
                    )
                    st.write(ans)
                except Exception as e:
                    st.error(f"Ollama error: {e}")
                    st.caption("Sanity checks: is Ollama running? does the model exist? try `ollama list`.")
    else:
        st.info("Enter a question above to route it to structured filters.")

# --- Tab 2: Board builder ---
with tabs[1]:
    st.subheader("Board Builder")
    st.caption("Pick pedals → see power budget + MIDI chain + quick red flags.")

    options = {r.id: f"{r.name} ({r.category})" for r in records}
    selected_ids = st.multiselect(
        "Select pedals",
        options=list(options.keys()),
        format_func=lambda x: options[x],
        key="board_builder_select",
    )
    selected = [r for r in records if r.id in selected_ids]

    col1, col2, col3 = st.columns(3)

    total_ma = sum((r.power.current_ma or 0) for r in selected)
    unknown_current = [r for r in selected if r.power.current_ma is None]

    with col1:
        st.metric("Total current draw (mA)", total_ma)
        if unknown_current:
            st.warning(
                f"{len(unknown_current)} pedal(s) missing current draw: "
                + ", ".join(r.id for r in unknown_current)
            )

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
        pick2 = st.selectbox("Pick a pedal", options=[r.id for r in selected], key="board_builder_pick")
        rr2 = next(r for r in selected if r.id == pick2)
        show_sources(rr2)
    else:
        st.info("Select pedals to build a board.")

# --- Tab 3: Prompt Comparison ---
with tabs[2]:
    st.subheader("Prompt Comparison")
    st.caption("See how the same user intent looks as raw context versus structured context.")

    options_pc = {r.id: f"{r.name} ({r.category})" for r in records}
    default_prompt_ids = ["grit_drive", "canyon_delay", "cloudburst_reverb"]
    prompt_defaults = [pid for pid in default_prompt_ids if pid in options_pc]

    selected_ids_pc = st.multiselect(
        "Select pedals for comparison",
        options=list(options_pc.keys()),
        format_func=lambda x: options_pc[x],
        default=prompt_defaults,
        key="prompt_compare_select",
    )
    selected_pc = [r for r in records if r.id in selected_ids_pc]

    raw_request = st.text_area(
        "Raw user request",
        value=(
            "I want something with a Grit Drive, maybe the Canyon delay, "
            "and a huge ambient verb. I use a CIOKS. I still want to stack gain sometimes."
        ),
        height=120,
        key="prompt_compare_raw_request",
    )

    preferences = st.text_input(
        "Preferences",
        value="ambient textures, gain stacking",
        key="prompt_compare_preferences",
    )

    constraints = st.text_input(
        "Constraints",
        value="9V isolated power planning",
        key="prompt_compare_constraints",
    )

    raw_context = build_raw_context(raw_request, selected_pc)
    structured_context = build_structured_context(selected_pc, preferences, constraints)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Raw context")
        st.code(raw_context, language="text")

    with col_right:
        st.markdown("### Structured context")
        st.code(structured_context, language="text")

    st.markdown("### Prompt-ready task block")
    prompt_task = structured_context + "\n\nReturn a signal chain recommendation with a short explanation."
    st.code(prompt_task, language="text")

    st.markdown("### What changed")
    st.markdown(
        """
- Less ambiguity
- Explicit constraints
- Cleaner task framing
- More inspectable model context
"""
    )

    # Optional: Let Ollama narrate the plan using the structured context
    llm_cfg_pc = ollama_controls("prompt_compare")
    if llm_cfg_pc["use_llm"]:
        st.markdown("### LLM narration (from selected pedals)")
        with st.spinner("Asking Ollama..."):
            try:
                # We treat the selected pedals as the "matches" list.
                question_for_llm = (
                    "Given these pedals and the user's preferences/constraints, recommend a coherent signal chain "
                    "and explain the reasoning. Stick to known facts; don't invent specs."
                )
                ans = ollama_narrate(
                    question=question_for_llm + "\n\nPreferences: " + preferences + "\nConstraints: " + constraints,
                    matches=selected_pc,
                    model=str(llm_cfg_pc["model"]),
                    base_url=str(llm_cfg_pc["base_url"]),
                )
                st.write(ans)
            except Exception as e:
                st.error(f"Ollama error: {e}")
                st.caption("Sanity checks: is Ollama running? does the model exist? try `ollama list`.")

# --- Tab 4: Browse ---
with tabs[3]:
    st.subheader("All extracted records")
    st.dataframe([record_row(r) for r in records], use_container_width=True)

    st.markdown("### Inspect sources")
    pick3 = st.selectbox("Pick any pedal", options=[r.id for r in records], key="pick_all")
    rr3 = next(r for r in records if r.id == pick3)
    show_sources(rr3)