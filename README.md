# Pedalboard Extractor Demo (RAG-friendly)

This repo is a small, musician-friendly demo of a powerful idea:

**Extractors turn messy gear text into structured facts, so your “assistant” can filter by reality before any LLM tries to talk.**

It’s intentionally lightweight:
- **Deterministic extraction** (regex + keyword rules)
- **Constraint routing** (simple “question → filters” parsing)
- Optional **LLM narration** (local Ollama) used only as a *presentation layer*, not a source of truth

---

## What this demo does

### 1) Messy notes → structured records
You write (or collect) messy text notes about pedals—think a mix of:
- “manual-ish” specs,
- product blurb,
- your own comments (“MIDI via TRS Type B”, “stereo out only”, etc.)

The extractor turns each note into a JSON record with fields like:
- power: `voltage_v`, `current_ma`, `polarity`
- I/O: mono/stereo in/out, `top_jacks`
- control: MIDI, MIDI type, TRS Type A/B, expression, tap tempo
- size: width/depth (when present)
- bypass type

**Crucially:** it also captures `sources` snippets per field so answers can be cited and audited.

### 2) Natural language questions → deterministic filtering
A mini router parses constraint-style questions like:
- “midi stereo out 9v”
- “reverbs with expression under 125mm”
- “midi trs type b”

…and converts them into structured constraints, then filters the extracted records.

### 3) Optional: LLM narration (Ollama)
Once the shortlist is deterministic, you can optionally ask a local Ollama model (e.g. `llama3.2:3b`) to **narrate** the results.

The LLM is **not** allowed to invent facts:
- it only sees the already-matched pedals + the snippet citations
- it’s instructed to say “unknown” when a field is missing
- it must cite snippet keys like `(power.current_ma)`

---

## Quickstart

### Create/activate a venv

**Windows (PowerShell)**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Extract records
```bash
python -m src.extract --data_dir data/pedals --out_dir out
```

This generates:
- `out/pedals.jsonl` — one JSON record per pedal note

### Run the CLI question mode
```bash
python -m src.query_demo --records out/pedals.jsonl --question "midi stereo out 9v"
python -m src.query_demo --records out/pedals.jsonl --question "reverbs with expression under 125mm"
python -m src.query_demo --records out/pedals.jsonl --question "midi trs type b"
```

---

## Streamlit UI

Run:
```bash
python -m streamlit run src/app.py
```

Tabs:
- **Search (Question Mode)**: ask a constraint question → see matches + sources + optional Ollama narration
- **Board Builder**: select pedals → see total current draw + MIDI list + inspect sources
- **Prompt Comparison**: show “raw context” vs “structured context” (useful for explaining why extractors make LLM prompting cleaner)
- **Browse Data**: explore all extracted records + inspect sources

---

## Ollama (local LLM narration)

### Install + pull a model
Install Ollama, then:
```bash
ollama pull llama3.2:3b
```

### CLI narration
```bash
python -m src.query_demo --records out/pedals.jsonl --question "midi stereo out 9v" --ollama --model llama3.2:3b
```

### Streamlit narration
In **Search (Question Mode)**, expand **LLM narration (Ollama)** and enable it.

> Tip: the demo is designed so the deterministic shortlist is always visible even if the LLM fails.

---

## Repo structure

```
data/
  pedals/              # messy pedal note .txt files
out/
  pedals.jsonl         # generated structured records (one per pedal)
src/
  app.py               # Streamlit UI (search, board builder, prompt comparison)
  extract.py           # extractor: notes -> structured JSONL + sources
  llm_answer.py        # Ollama narration (presentation layer, grounded by snippets)
  query_demo.py        # CLI demo + question->constraints router + filters
  schema.py            # Pydantic schema for PedalRecord + typed enums
  utils.py             # helpers (regex utilities, unit parsing, category guessing)
```

---

## What each `src/` file does

### `src/schema.py`
Defines the typed record model (`PedalRecord`) using Pydantic:
- category, power, I/O, control, size
- musician-friendly fields like `top_jacks`, `bypass`, `trs_midi_type`
- `sources`: a mapping of `field_path -> snippet` for provenance

### `src/utils.py`
Utility helpers:
- whitespace normalization
- safe “find first regex match”
- snippet extraction for citations
- unit parsing (mm/cm/in to mm)
- lightweight category inference

### `src/extract.py`
The extractor pipeline:
- loads each `data/pedals/*.txt`
- extracts structured fields with deterministic rules
- stores extracted `sources` snippets for key fields
- writes `out/pedals.jsonl`

### `src/query_demo.py`
CLI demo:
- loads `out/pedals.jsonl`
- supports `--question` mode that routes a natural language request to constraints
- applies deterministic filters to return a shortlist
- optional `--ollama` mode for narrated answers via `llm_answer.py`

### `src/rag_index.py`
Embeddings index builder:
- reads the **entire pedal note text** from `data/pedals/*.txt`
- builds sentence-transformer embeddings (default: `all-MiniLM-L6-v2`)
- writes a normalized vector index to `out/embeddings.npz` (`ids` + `embs`)

Run:
```bash
python -m src.rag_index --data_dir data/pedals --out_dir out
```

### `src/rag_search.py`
Embeddings search helper (cosine similarity):
- loads `out/embeddings.npz`
- embeds a free-text “sound/vibe” query
- returns the top-k matching pedal IDs + similarity scores

Run:
```bash
python -m src.rag_search --index out/embeddings.npz --query "huge ambient wash, modulated trails" --top_k 5
```

### `src/llm_answer.py`
Ollama integration:
- calls `http://127.0.0.1:11434/api/chat`
- provides the model only the matched records + snippet citations
- enforces grounded formatting and retries if the model contradicts match counts

### `src/app.py`
Streamlit UI:
- Question Mode: type a question → see deterministic results + sources + optional Ollama narration
- Board Builder: pick pedals → power budget + MIDI chain list + inspect sources
- Prompt Comparison: shows why “structured context” prompts are cleaner than raw text blobs

---

## Why this matters (the “extractor layer”)
If you only do RAG over unstructured chunks, the LLM has to infer specs from paragraphs.
That’s fine for open-ended Q&A, but it gets shaky for constraint questions like:
- “Will this run on 9V 100mA?”
- “Which ones have MIDI + stereo out?”
- “Find reverbs with expression under 125mm wide”

Extractors solve that by:
- making “spec” questions **queryable**
- making responses **auditable** (via snippets)
- letting the LLM do what it’s good at: **explaining and summarizing**, not guessing.

---
