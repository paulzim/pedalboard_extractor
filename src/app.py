from __future__ import annotations

import streamlit as st
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.query_demo import (
    load_records,
    parse_question,
    apply_constraints,
    evaluate_constraints,
)
from src.schema import PedalRecord
from src.llm_answer import ollama_narrate

st.set_page_config(page_title="Pedalboard Extractor Demo", layout="wide")


# ---------------------------
# Records + display helpers
# ---------------------------

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


def build_elimination_rows(records: List[PedalRecord], constraints: dict) -> List[Dict]:
    rows: List[Dict] = []
    for r in records:
        ev = evaluate_constraints(r, constraints)
        if ev.passed:
            continue
        rows.append(
            {
                "id": r.id,
                "name": r.name,
                "category": r.category,
                "first_failure": ev.first_failure or "",
                "all_failures": "; ".join(ev.failures),
            }
        )
    return rows


# ---------------------------
# Prompt comparison helpers
# ---------------------------

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
            pedal_lines.append(
                f"   Size: {r.size_mm.width:.0f}mm x {r.size_mm.depth:.0f}mm"
            )

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


# ---------------------------
# Ollama controls (shared)
# ---------------------------

def ollama_controls(scope_key: str) -> Dict[str, object]:
    with st.expander("LLM narration (Ollama)", expanded=False):
        use_llm = st.checkbox("Use Ollama to narrate", value=False, key=f"{scope_key}_use_ollama")
        model = st.text_input("Ollama model", value="llama3.2:3b", key=f"{scope_key}_ollama_model")
        base_url = st.text_input("Ollama base URL", value="http://127.0.0.1:11434", key=f"{scope_key}_ollama_url")
        st.caption("Filtering stays deterministic; Ollama just narrates from extracted facts + snippets.")
    return {"use_llm": use_llm, "model": model, "base_url": base_url}


# ---------------------------
# Embeddings (sound search)
# ---------------------------

@st.cache_data
def load_embeddings(index_path: str) -> Tuple[List[str], np.ndarray]:
    p = Path(index_path)
    data = np.load(p, allow_pickle=False)
    ids = data["ids"].tolist()
    embs = data["embs"]  # normalized float32 [N, D]
    return ids, embs


@st.cache_resource
def load_embed_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n


def sound_search(
    query: str,
    ids: List[str],
    embs: np.ndarray,
    *,
    model_name: str,
    top_k: int,
    allowed_ids: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    model = load_embed_model(model_name)
    q = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
    q = normalize_vec(q)

    sims = embs @ q  # cosine similarity (normalized)
    if allowed_ids is not None:
        allowed = set(allowed_ids)
        mask = np.array([pid in allowed for pid in ids], dtype=bool)
        if mask.sum() == 0:
            return []
        sims = np.where(mask, sims, -1e9)

    idx = np.argsort(-sims)[:top_k]
    results: List[Tuple[str, float]] = []
    for i in idx:
        if sims[i] < -1e8:
            continue
        results.append((ids[i], float(sims[i])))
    return results


# ---------------------------
# Merge mode (combined query -> sound + specs)
# ---------------------------

_SPEC_STOPWORDS = {
    # constraint-ish keywords we strip from sound query
    "midi", "stereo", "mono", "in", "out", "input", "output",
    "expression", "exp", "tap", "tempo",
    "true", "bypass", "buffered", "top", "jacks", "top-mounted", "topmounted",
    "type", "a", "b", "trs", "usb", "din",
    "under", "max", "<=", ">=", "less", "more",
    "v", "vdc", "volt", "volts", "ma", "mm", "cm", "inch", "inches",
    "reverb", "delay", "drive", "tuner", "looper", "utility", "modulation", "multi", "fx",
}

def split_merge_query(merged: str) -> Tuple[str, str, dict]:
    """
    Returns (sound_query, spec_query, parsed_constraints)

    - spec_query is the original merged string (we let parse_question extract what it can)
    - sound_query is the merged string with obvious constraint tokens removed
    """
    merged = merged.strip()
    parsed = parse_question(merged)

    # crude token cleanup for sound query: remove numbers+units and stopwords
    lower = merged.lower()
    # remove numeric constraints (e.g., 9v, 500ma, 125mm)
    lower = re.sub(r"\b\d+(\.\d+)?\s*(v|vdc|ma|mm|cm|in|inch|inches)\b", " ", lower)
    # remove symbols
    lower = re.sub(r"[^\w\s-]", " ", lower)
    tokens = [t for t in lower.split() if t.strip()]

    kept = []
    for t in tokens:
        if t in _SPEC_STOPWORDS:
            continue
        # drop pure numbers
        if re.fullmatch(r"\d+(\.\d+)?", t):
            continue
        kept.append(t)

    sound_query = " ".join(kept).strip()
    spec_query = merged
    return sound_query, spec_query, parsed


# ---------------------------
# UI
# ---------------------------

st.title("Pedalboard Extractor Demo")
records_path = st.sidebar.text_input("Records path", value="out/pedals.jsonl")
records = cached_records(records_path)

tabs = st.tabs(
    [
        "Search (Question Mode)",
        "Board Builder",
        "Prompt Comparison",
        "Sound Search (Embeddings)",
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

    explain = st.checkbox("Explain eliminations", value=False, key="question_mode_explain")
    llm_cfg = ollama_controls("question_mode")

    if q:
        c = parse_question(q)
        results = apply_constraints(records, c)
        st.caption(f"Parsed constraints: {c}")

        if explain:
            eliminated_rows = build_elimination_rows(records, c)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Matched", len(results))
            with col_b:
                st.metric("Eliminated", len(eliminated_rows))
            with col_c:
                st.metric("Total", len(records))

            if eliminated_rows:
                st.markdown("### Why pedals were eliminated")
                st.dataframe(eliminated_rows, use_container_width=True)

        st.write(f"Matches: {len(results)}")

        if results:
            st.dataframe([record_row(r) for r in results], use_container_width=True)
            st.markdown("### Inspect sources")
            pick = st.selectbox("Pick a result", options=[r.id for r in results], key="question_mode_pick")
            rr = next(r for r in results if r.id == pick)
            show_sources(rr)
        else:
            st.info("No matches. Try relaxing one constraint.")

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

    preferences = st.text_input("Preferences", value="ambient textures, gain stacking", key="prompt_compare_preferences")
    constraints = st.text_input("Constraints", value="9V isolated power planning", key="prompt_compare_constraints")

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
    st.code(structured_context + "\n\nReturn a signal chain recommendation with a short explanation.", language="text")

    llm_cfg_pc = ollama_controls("prompt_compare")
    if llm_cfg_pc["use_llm"]:
        st.markdown("### LLM narration (from selected pedals)")
        with st.spinner("Asking Ollama..."):
            ans = ollama_narrate(
                question="Recommend a coherent signal chain given these pedals and constraints.",
                matches=selected_pc,
                model=str(llm_cfg_pc["model"]),
                base_url=str(llm_cfg_pc["base_url"]),
            )
            st.write(ans)

# --- Tab 4: Sound Search (Embeddings) ---
with tabs[3]:
    st.subheader("Sound Search (Embeddings)")
    st.caption("Free-text vibe search over the *entire pedal notes*. Specs remain deterministic elsewhere.")

    index_path = st.text_input("Embeddings index path", value="out/embeddings.npz", key="emb_index_path")
    embed_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2", key="emb_model_name")

    st.markdown("### Merge mode (combined query)")
    merged = st.text_input(
        'Try: "huge ambient wash reverb, but must be stereo out + midi + 9v"',
        key="merge_query",
    )
    use_merge = st.checkbox("Use merge mode", value=False, key="merge_enable")

    sound_q = ""
    spec_q = ""
    parsed_constraints = None

    if use_merge and merged.strip():
        sound_q, spec_q, parsed_constraints = split_merge_query(merged)
        st.caption(f"Parsed constraints: {parsed_constraints}")
        st.write(f"Sound query extracted: **{sound_q or '(empty)'}**")
        st.write(f"Spec query (for constraints parser): **{spec_q}**")

    st.markdown("### Sound query")
    colx, coly = st.columns([2, 1])
    with colx:
        sound_q_input = st.text_input(
            'Try: "huge ambient wash, modulated trails" or "tight mid gain that stacks well"',
            value=sound_q,
            key="sound_query",
        )
    with coly:
        top_k = st.slider("Top K", min_value=3, max_value=20, value=8, step=1, key="sound_topk")

    st.markdown("### Optional: spec constraints first")
    apply_specs_first = st.checkbox(
        "Apply spec constraints first",
        value=bool(use_merge and merged.strip()),
        key="sound_apply_specs_first",
    )

    spec_query = st.text_input(
        'Spec constraints (same format as Question Mode), e.g. "midi stereo out 9v" or "reverb expression"',
        value=(spec_q if apply_specs_first else ""),
        key="sound_spec_query",
    )

    run = st.button("Run sound search", key="sound_run")

    if run:
        p = Path(index_path)
        if not p.exists():
            st.error(f"Embeddings index not found: {index_path}")
            st.code("python -m src.rag_index --data_dir data/pedals --out_dir out", language="bash")
        elif not sound_q_input.strip():
            st.warning("Enter a sound/vibe query first.")
        else:
            ids, embs = load_embeddings(index_path)

            allowed_ids = None
            if apply_specs_first and spec_query.strip():
                c = parse_question(spec_query)
                filtered = apply_constraints(records, c)
                allowed_ids = [r.id for r in filtered]
                st.caption(f"Candidate set after deterministic constraints: {len(allowed_ids)}")
                if len(allowed_ids) == 0:
                    st.warning("No candidates left after constraints. Relax the spec constraints.")
                    st.stop()

            results = sound_search(
                query=sound_q_input.strip(),
                ids=ids,
                embs=embs,
                model_name=embed_model,
                top_k=int(top_k),
                allowed_ids=allowed_ids,
            )

            if not results:
                st.info("No results (or all candidates filtered out).")
            else:
                by_id = {r.id: r for r in records}
                rows: List[Dict] = []
                for pid, score in results:
                    r = by_id.get(pid)
                    if not r:
                        continue
                    row = record_row(r)
                    row["score"] = round(score, 4)
                    rows.append(row)

                st.markdown("### Ranked results")
                st.dataframe(rows, use_container_width=True)

                pick_sound = st.selectbox("Inspect sources for a result", options=[r["id"] for r in rows], key="sound_pick")
                rr = by_id[pick_sound]
                show_sources(rr)

                st.markdown("### Notes")
                st.markdown(
                    "- Sound search is fuzzy ranking over the full note text.\n"
                    "- Constraints remain deterministic (extractors + filters).\n"
                    "- Merge mode is a heuristic split: good for demos, not perfect NLP."
                )

# --- Tab 5: Browse ---
with tabs[4]:
    st.subheader("All extracted records")
    st.dataframe([record_row(r) for r in records], use_container_width=True)

    st.markdown("### Inspect sources")
    pick3 = st.selectbox("Pick any pedal", options=[r.id for r in records], key="pick_all")
    rr3 = next(r for r in records if r.id == pick3)
    show_sources(rr3)