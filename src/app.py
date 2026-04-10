from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.llm_answer import ollama_narrate
from src.query_demo import apply_constraints, evaluate_constraints, load_records, parse_question
from src.schema import PedalRecord

st.set_page_config(page_title="Pedalboard Extractor Demo", layout="wide")


# ---------------------------
# Caching / loading
# ---------------------------

@st.cache_data
def cached_records(path: str) -> List[PedalRecord]:
    return load_records(path)


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


# ---------------------------
# Display helpers
# ---------------------------

def record_row(r: PedalRecord) -> Dict[str, object]:
    return {
        "id": r.id,
        "name": r.name,
        "category": r.category,
        "voltage_v": r.power.voltage_v,
        "current_ma": r.power.current_ma,
        "polarity": r.power.polarity,
        "stereo_out": r.io.stereo_out,
        "stereo_in": r.io.stereo_in,
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


def show_sources(r: PedalRecord) -> None:
    st.markdown("**Sources (snippets)**")
    if not r.sources:
        st.caption("No snippets captured.")
        return
    for k in sorted(r.sources.keys()):
        st.write(f"- `{k}`: {r.sources[k]}")


def build_elimination_rows(records: List[PedalRecord], constraints: dict) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
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


def strip_snippets_used_section(text: str) -> str:
    """
    Presentation-friendly cleanup for grounded LLM output.

    Removes the trailing '## Snippets used' section so the audience sees
    a cleaner answer in the side-by-side comparison. Provenance is still
    available in the app via 'Inspect sources'.
    """
    cleaned = re.sub(
        r"\n##\s*Snippets used[\s\S]*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


# ---------------------------
# Context builders (raw vs structured)
# ---------------------------

def build_structured_context(selected: List[PedalRecord], preferences: str = "", constraints: str = "") -> str:
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
        if r.io.top_jacks is True:
            io_bits.append("top jacks")
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

        if r.bypass and r.bypass != "unknown":
            pedal_lines.append(f"   Bypass: {r.bypass}")

        if r.size_mm.width is not None and r.size_mm.depth is not None:
            pedal_lines.append(f"   Size: {r.size_mm.width:.0f}mm x {r.size_mm.depth:.0f}mm")

        pedal_lines.append(f"   Record id: {r.id}")
        lines.extend(pedal_lines)

    total_ma = sum((r.power.current_ma or 0) for r in selected)
    unknown_current = [r.id for r in selected if r.power.current_ma is None]

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
        lines.append(f"- Total current draw (known only): {total_ma}mA")
        if unknown_current:
            lines.append(f"- Missing current draw for: {', '.join(unknown_current)}")
        lines.append("- Use isolated 9V outputs where needed (when applicable)")

    lines.append("")
    lines.append("Task:")
    lines.append("- Recommend a coherent signal chain and explain the reasoning.")
    lines.append("- If something is unknown, say so—do not invent specs.")

    return "\n".join(lines)


def load_raw_note_text(pedal_id: str, data_dir: str = "data/pedals") -> str:
    """
    Loads the raw note text from data/pedals/{id}.txt
    """
    p = Path(data_dir) / f"{pedal_id}.txt"
    if not p.exists():
        return f"(raw note not found: {p})"
    return p.read_text(encoding="utf-8", errors="replace")


def build_raw_context(user_prompt: str, selected: List[PedalRecord], excerpt_chars: int = 900) -> str:
    """
    TRUE 'raw' baseline:
    - includes the user prompt
    - includes raw note excerpts (not extracted fields)
    """
    lines: List[str] = []
    lines.append("User prompt:")
    lines.append(user_prompt.strip())

    if selected:
        lines.append("")
        lines.append("Raw pedal notes (excerpts):")
        for r in selected:
            raw = load_raw_note_text(r.id)
            raw_clean = " ".join(raw.split())
            excerpt = raw_clean[:excerpt_chars] + ("…" if len(raw_clean) > excerpt_chars else "")
            lines.append("")
            lines.append(f"--- {r.name} (id: {r.id}) ---")
            lines.append(excerpt)

    return "\n".join(lines)


# ---------------------------
# Ollama controls (shared)
# ---------------------------

def ollama_controls(scope_key: str) -> Dict[str, object]:
    with st.expander("LLM settings (Ollama)", expanded=False):
        use_llm = st.checkbox("Run Ollama comparison", value=True, key=f"{scope_key}_use_ollama")
        model = st.text_input("Ollama model", value="llama3.2:3b", key=f"{scope_key}_ollama_model")
        base_url = st.text_input("Ollama base URL", value="http://127.0.0.1:11434", key=f"{scope_key}_ollama_url")
        st.caption("Raw side uses a naive prompt over raw text. Structured side uses grounded narration + snippet citations.")
    return {"use_llm": use_llm, "model": model, "base_url": base_url}


def ollama_naive_answer(
    *,
    model: str,
    base_url: str,
    user_prompt: str,
    raw_context: str,
    timeout_s: int = 60,
) -> str:
    """
    Naive baseline:
    - sends raw prompt + raw note excerpts
    - does NOT include extracted fields or snippet keys
    """
    system = (
        "You are a helpful guitar pedalboard assistant.\n"
        "You will be given a user prompt and raw pedal notes.\n"
        "Make a reasonable recommendation for a signal chain and explain why.\n"
        "If information is missing, you may make best-effort assumptions, but be clear when you are assuming.\n"
    )

    user = (
        f"{raw_context}\n\n"
        "Task:\n"
        "- Recommend a coherent signal chain.\n"
        "- Explain the reasoning.\n"
        "- Call out any assumptions.\n"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.4, "num_ctx": 4096},
    }

    url = f"{base_url.rstrip('/')}/api/chat"
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        return "LLM returned an empty response."
    return content.strip()


# ---------------------------
# Embeddings search
# ---------------------------

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

    sims = embs @ q
    if allowed_ids is not None:
        allowed = set(allowed_ids)
        mask = np.array([pid in allowed for pid in ids], dtype=bool)
        if mask.sum() == 0:
            return []
        sims = np.where(mask, sims, -1e9)

    idx = np.argsort(-sims)[:top_k]
    out: List[Tuple[str, float]] = []
    for i in idx:
        if sims[i] < -1e8:
            continue
        out.append((ids[i], float(sims[i])))
    return out


# ---------------------------
# Merge mode splitter (combined prompt -> sound query + constraints)
# ---------------------------

_FILLER_PHRASES = [
    r"\bi want\b",
    r"\bi would like\b",
    r"\bi'm looking for\b",
    r"\bi am looking for\b",
    r"\bcan you\b",
    r"\bgive me\b",
    r"\bhelp me\b",
    r"\bif possible\b",
    r"\bideally\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bplease\b",
]

_SOUND_STOPWORDS = {
    "i", "im", "i'm", "me", "my", "we", "our", "you",
    "want", "like", "looking", "for", "a", "an", "the",
    "something", "tone", "sound", "vibe", "kind", "sort",
    "with", "and", "or", "but", "that", "this",
    "to", "of", "in", "on", "at", "from",
    "if", "possible", "ideally", "maybe",
    "please", "thanks",
    "it", "its", "it's",
    "keep", "make", "get",
}

_SPEC_STOPWORDS = {
    "midi", "stereo", "mono", "in", "out", "input", "output",
    "expression", "exp", "tap", "tempo",
    "true", "bypass", "buffered", "switchable",
    "top", "jacks", "top-mounted", "topmounted", "side",
    "type", "a", "b", "trs", "usb", "din",
    "under", "max", "<=", ">=", "less", "more",
    "v", "vdc", "volt", "volts", "ma", "mm", "cm", "inch", "inches",
    "power", "isolated", "supply", "supplies", "cioks",
    "tuner", "utility", "modulation", "multi", "fx", "effects",
}

def split_merge_query(merged: str) -> Tuple[str, dict]:
    merged = merged.strip()
    parsed = parse_question(merged)

    lower = merged.lower()

    for pat in _FILLER_PHRASES:
        lower = re.sub(pat, " ", lower)

    lower = re.sub(r"\b\d+(\.\d+)?\s*(v|vdc|ma|mm|cm|in|inch|inches)\b", " ", lower)
    lower = re.sub(r"[^\w\s-]", " ", lower)

    tokens = [t for t in lower.split() if t.strip()]

    kept: List[str] = []
    for t in tokens:
        if t in _SOUND_STOPWORDS:
            continue
        if t in _SPEC_STOPWORDS:
            continue
        if re.fullmatch(r"\d+(\.\d+)?", t):
            continue
        kept.append(t)

    sound_query = " ".join(kept).strip()
    if not sound_query:
        sound_query = merged

    return sound_query, parsed


def pick_fallback_pedals(records: List[PedalRecord], prompt: str, k: int) -> List[PedalRecord]:
    by_id = {r.id: r for r in records}
    defaults = [pid for pid in ["grit_drive", "canyon_delay", "cloudburst_reverb", "nexus_multi_fx"] if pid in by_id]
    picked: List[PedalRecord] = [by_id[pid] for pid in defaults][:k]

    want = prompt.lower()
    cat_order = []
    if "reverb" in want or "wash" in want or "ambient" in want:
        cat_order.append("reverb")
    if "delay" in want or "echo" in want:
        cat_order.append("delay")
    if "drive" in want or "distort" in want or "gain" in want:
        cat_order.append("drive")

    for cat in cat_order:
        for r in records:
            if len(picked) >= k:
                return picked
            if r in picked:
                continue
            if r.category == cat:
                picked.append(r)

    for r in records:
        if len(picked) >= k:
            break
        if r not in picked:
            picked.append(r)
    return picked[:k]


def auto_select_pedals(
    records: List[PedalRecord],
    user_prompt: str,
    *,
    index_path: str,
    embed_model: str,
    top_k: int,
    apply_constraints_first: bool,
) -> Tuple[List[PedalRecord], Dict, str, List[Tuple[str, float]]]:
    constraints = parse_question(user_prompt)
    sound_q, _ = split_merge_query(user_prompt)
    sound_query = sound_q if sound_q else user_prompt.strip()

    allowed_ids: Optional[List[str]] = None
    if apply_constraints_first and constraints:
        filtered = apply_constraints(records, constraints)
        allowed_ids = [r.id for r in filtered]

    p = Path(index_path)
    if p.exists():
        ids, embs = load_embeddings(index_path)
        scored = sound_search(
            query=sound_query,
            ids=ids,
            embs=embs,
            model_name=embed_model,
            top_k=max(top_k, 10),
            allowed_ids=allowed_ids,
        )

        by_id = {r.id: r for r in records}
        chosen: List[PedalRecord] = []
        for pid, _score in scored:
            if pid in by_id and by_id[pid] not in chosen:
                chosen.append(by_id[pid])
            if len(chosen) >= top_k:
                break

        if not chosen:
            chosen = pick_fallback_pedals(
                apply_constraints(records, constraints) if constraints else records,
                user_prompt,
                top_k,
            )
        return chosen, constraints, sound_query, scored

    chosen = pick_fallback_pedals(
        apply_constraints(records, constraints) if constraints else records,
        user_prompt,
        top_k,
    )
    return chosen, constraints, sound_query, []


# ---------------------------
# UI
# ---------------------------

st.title("Pedalboard Extractor Demo")
records_path = st.sidebar.text_input("Records path", value="out/pedals.jsonl")
records = cached_records(records_path)

tabs = st.tabs(
    [
        "Demo: One-Prompt (Comparison)",
        "Constraint Finder",
        "Vibe Search (Embeddings)",
        "Board Builder",
        "Browse Data",
    ]
)

# -------------------------------------------------------------------
# TAB 0: HERO DEMO (real comparison)
# -------------------------------------------------------------------
with tabs[0]:
    st.subheader("Demo: One-Prompt (Comparison)")
    st.caption("This is a real baseline vs structured test: raw notes + naive LLM vs extracted fields + grounded narration.")

    user_prompt = st.text_area(
        "Your prompt",
        value="I want a distorted, washy delay tone. Stereo if possible. I use 9V isolated power.",
        height=90,
        key="one_prompt_input",
    )

    with st.expander("Advanced (optional)", expanded=False):
        index_path = st.text_input("Embeddings index path", value="out/embeddings.npz", key="one_prompt_index")
        embed_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2", key="one_prompt_emb_model")
        top_k = st.slider("How many pedals to select", min_value=2, max_value=8, value=5, step=1, key="one_prompt_topk")
        apply_constraints_first = st.checkbox(
            "Apply parsed spec constraints before embeddings",
            value=True,
            key="one_prompt_apply_constraints_first",
            help="If enabled: deterministic filters narrow candidates, then embeddings rank only within that set.",
        )
        show_explain = st.checkbox("Show elimination explain table", value=False, key="one_prompt_explain")
        raw_excerpt_chars = st.slider("Raw note excerpt length", 300, 2000, 900, 100, key="raw_excerpt_chars")

    llm_cfg = ollama_controls("one_prompt_llm")
    run = st.button("Run comparison", key="one_prompt_run")

    if run:
        index_path = st.session_state.get("one_prompt_index", "out/embeddings.npz")
        embed_model = st.session_state.get("one_prompt_emb_model", "sentence-transformers/all-MiniLM-L6-v2")
        top_k = int(st.session_state.get("one_prompt_topk", 5))
        apply_constraints_first = bool(st.session_state.get("one_prompt_apply_constraints_first", True))
        show_explain = bool(st.session_state.get("one_prompt_explain", False))
        raw_excerpt_chars = int(st.session_state.get("raw_excerpt_chars", 900))

        selected, constraints, sound_query, scored = auto_select_pedals(
            records,
            user_prompt,
            index_path=index_path,
            embed_model=embed_model,
            top_k=top_k,
            apply_constraints_first=apply_constraints_first,
        )

        st.markdown("### What the code inferred")
        st.write(f"- Parsed constraints: `{constraints}`")
        st.write(f"- Sound query used for embeddings: **{sound_query}**")
        if scored:
            preview = [{"id": pid, "score": round(score, 4)} for pid, score in scored[:10]]
            st.dataframe(preview, use_container_width=True)
        else:
            st.caption("No embeddings scores (index missing or fallback selection used).")

        if show_explain and constraints:
            st.markdown("### Explain eliminations (based on parsed constraints)")
            eliminated_rows = build_elimination_rows(records, constraints)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Matched (constraints)", len(apply_constraints(records, constraints)))
            with col_b:
                st.metric("Eliminated", len(eliminated_rows))
            with col_c:
                st.metric("Total", len(records))
            if eliminated_rows:
                st.dataframe(eliminated_rows, use_container_width=True)

        st.markdown("### Auto-selected pedals")
        st.dataframe([record_row(r) for r in selected], use_container_width=True)

        preferences_text = sound_query
        constraints_text = ", ".join([f"{k}={v}" for k, v in constraints.items()]) if constraints else ""

        raw_context = build_raw_context(user_prompt, selected, excerpt_chars=raw_excerpt_chars)
        structured_context = build_structured_context(selected, preferences_text, constraints_text)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Raw context (true baseline)")
            st.code(raw_context, language="text")
        with c2:
            st.markdown("### Structured context (extracted)")
            st.code(structured_context, language="text")

        if llm_cfg["use_llm"]:
            st.markdown("### Ollama outputs (Naive vs Grounded)")
            o1, o2 = st.columns(2)

            with o1:
                with st.spinner("Asking Ollama (naive, raw notes)..."):
                    try:
                        ans_raw = ollama_naive_answer(
                            model=str(llm_cfg["model"]),
                            base_url=str(llm_cfg["base_url"]),
                            user_prompt=user_prompt,
                            raw_context=raw_context,
                        )
                        st.markdown("#### Naive (raw notes) → Ollama")
                        st.write(ans_raw)
                    except Exception as e:
                        st.error(f"Ollama error (naive/raw): {e}")

            with o2:
                with st.spinner("Asking Ollama (grounded, extracted fields)..."):
                    try:
                        ans_struct = ollama_narrate(
                            question="Recommend a coherent signal chain and explain the reasoning.\n\n" + structured_context,
                            matches=selected,
                            model=str(llm_cfg["model"]),
                            base_url=str(llm_cfg["base_url"]),
                        )
                        ans_struct_clean = strip_snippets_used_section(ans_struct)
                        st.markdown("#### Grounded (structured) → Ollama")
                        st.write(ans_struct_clean)
                    except Exception as e:
                        st.error(f"Ollama error (grounded/structured): {e}")

        st.markdown("### Inspect sources")
        if selected:
            pick_src = st.selectbox("Pick a pedal", options=[r.id for r in selected], key="one_prompt_pick_sources")
            if pick_src is not None:
                rr = next(r for r in selected if r.id == pick_src)
                show_sources(rr)

# -------------------------------------------------------------------
# TAB 1: Constraint Finder
# -------------------------------------------------------------------
with tabs[1]:
    st.subheader("Constraint Finder")
    st.caption("Deterministic constraints over extracted fields. Great for spec questions.")

    q = st.text_input(
        'Try: "reverbs with expression under 125mm" or "midi stereo out 9v" or "top jacks true bypass"',
        key="constraint_finder_input",
    )

    explain = st.checkbox("Explain eliminations", value=False, key="constraint_finder_explain")
    llm_cfg = ollama_controls("constraint_finder_llm")

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
                st.dataframe(eliminated_rows, use_container_width=True)

        st.write(f"Matches: {len(results)}")

        if results:
            st.dataframe([record_row(r) for r in results], use_container_width=True)
            pick = st.selectbox("Pick a result", options=[r.id for r in results], key="constraint_pick")
            if pick is not None:
                rr = next(r for r in results if r.id == pick)
                show_sources(rr)
        else:
            st.info("No matches. Try relaxing one constraint.")

        if llm_cfg["use_llm"]:
            st.markdown("### LLM narration (grounded)")
            with st.spinner("Asking Ollama..."):
                ans = ollama_narrate(
                    question=q,
                    matches=results,
                    model=str(llm_cfg["model"]),
                    base_url=str(llm_cfg["base_url"]),
                )
                ans_clean = strip_snippets_used_section(ans)
                st.write(ans_clean)

# -------------------------------------------------------------------
# TAB 2: Vibe Search (Embeddings)
# -------------------------------------------------------------------
with tabs[2]:
    st.subheader("Vibe Search (Embeddings)")
    st.caption("Free-text ‘sound/vibe’ similarity over the entire pedal notes. Specs remain deterministic via constraints.")

    index_path = st.text_input("Embeddings index path", value="out/embeddings.npz", key="vibe_index_path")
    embed_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2", key="vibe_emb_model")

    merged = st.text_input('Merge mode: "ambient wash reverb but must be midi stereo out 9v"', key="vibe_merge")
    use_merge = st.checkbox("Use merge mode", value=False, key="vibe_merge_enable")

    sound_q = ""
    constraints = {}
    if use_merge and merged.strip():
        sound_q, constraints = split_merge_query(merged)
        st.caption(f"Parsed constraints: {constraints}")
        st.write(f"Sound query extracted: **{sound_q or '(empty)'}**")

    colx, coly = st.columns([2, 1])
    with colx:
        sound_q_input = st.text_input("Sound query", value=sound_q, key="vibe_sound_query")
    with coly:
        top_k = st.slider("Top K", 3, 20, 8, 1, key="vibe_topk")

    apply_specs_first = st.checkbox("Apply constraints first", value=bool(constraints), key="vibe_apply_specs_first")
    spec_query = st.text_input('Spec constraints string (optional)', value=(merged if apply_specs_first else ""), key="vibe_spec_query")

    if st.button("Run vibe search", key="vibe_run"):
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

            results = sound_search(
                query=sound_q_input.strip(),
                ids=ids,
                embs=embs,
                model_name=embed_model,
                top_k=int(top_k),
                allowed_ids=allowed_ids,
            )

            if not results:
                st.info("No results.")
            else:
                by_id = {r.id: r for r in records}
                rows: List[Dict[str, object]] = []
                for pid, score in results:
                    r = by_id.get(pid)
                    if not r:
                        continue
                    row = record_row(r)
                    row["score"] = round(score, 4)
                    rows.append(row)

                st.dataframe(rows, use_container_width=True)

                row_ids = [str(r["id"]) for r in rows]
                pick = st.selectbox("Inspect sources", options=row_ids, key="vibe_pick")
                if pick is not None:
                    show_sources(by_id[pick])

# -------------------------------------------------------------------
# TAB 3: Board Builder
# -------------------------------------------------------------------
with tabs[3]:
    st.subheader("Board Builder")
    st.caption("Pick pedals → see power budget + MIDI list + red flags.")

    options = {r.id: f"{r.name} ({r.category})" for r in records}
    selected_ids = st.multiselect("Select pedals", list(options.keys()), format_func=lambda x: options[x], key="board_select")
    selected = [r for r in records if r.id in selected_ids]

    col1, col2, col3 = st.columns(3)
    total_ma = sum((r.power.current_ma or 0) for r in selected)
    unknown_current = [r for r in selected if r.power.current_ma is None]

    with col1:
        st.metric("Total current draw (mA)", total_ma)
        if unknown_current:
            st.warning("Missing current draw for: " + ", ".join(r.id for r in unknown_current))

    with col2:
        midi_pedals = [r for r in selected if r.control.midi]
        st.metric("MIDI-capable pedals", len(midi_pedals))
        for r in midi_pedals:
            st.write(f"- {r.name} — {r.control.midi_type} ({r.control.trs_midi_type})")

    with col3:
        stereo_out = [r for r in selected if r.io.stereo_out]
        st.metric("Stereo-out pedals", len(stereo_out))

    if selected:
        st.dataframe([record_row(r) for r in selected], use_container_width=True)
        pick = st.selectbox("Inspect sources", options=[r.id for r in selected], key="board_pick")
        if pick is not None:
            rr = next(r for r in selected if r.id == pick)
            show_sources(rr)

# -------------------------------------------------------------------
# TAB 4: Browse Data
# -------------------------------------------------------------------
with tabs[4]:
    st.subheader("All extracted records")
    st.dataframe([record_row(r) for r in records], use_container_width=True)

    pick = st.selectbox("Inspect sources", options=[r.id for r in records], key="browse_pick")
    if pick is not None:
        rr = next(r for r in records if r.id == pick)
        show_sources(rr)