# AGENTS.md

## Purpose of this repo
This repo supports a live presentation demo about extraction, retrieval, and grounded AI behavior.

The core teaching point is:

**Prompt engineering starts earlier than the prompt.**

The demo is designed to show two related ideas:

1. **Extraction for ingestion**  
   Messy source notes become structured records.

2. **Extraction for prompt design**  
   Those structured records become better task-specific context for the model.

Reliability and clarity matter more than ambitious refactors.

---

## Architecture overview

### Raw inputs
- `data/pedals/*.txt`
- Human-readable pedal notes with inconsistent phrasing, mixed facts, and implied meaning.

### Extraction / record layer
- `src/extract.py`
- `src/schema.py`
- `src/utils.py`

This layer:
- parses raw note text
- normalizes explicit facts
- builds typed `PedalRecord` objects
- attaches source snippets for provenance
- writes `out/pedals.jsonl`

### Retrieval / filtering layer
- `src/query_demo.py`
- `src/rag_index.py`
- `src/rag_search.py`

This layer:
- parses deterministic constraints from user prompts
- evaluates/filter records against those constraints
- builds and queries embeddings for fuzzy “vibe” retrieval

### Prompt / synthesis layer
- `src/llm_answer.py`
- `src/app.py`

This layer:
- assembles raw vs structured context
- compares naive vs grounded LLM behavior
- exposes inspectability and explainability in the Streamlit UI

---

## Most important demo path
The live presentation depends primarily on this flow:

1. show raw source files in `data/pedals`
2. show focused portions of `src/extract.py`
3. run extraction to produce `out/pedals.jsonl`
4. open the Streamlit app
5. use **Demo: One-Prompt (Comparison)**
6. show:
   - parsed constraints
   - sound query used for embeddings
   - auto-selected pedals
   - raw vs structured context
   - naive vs grounded outputs
   - source snippets

Preserve this flow unless explicitly asked to redesign it.

---

## Working principles
- Prefer **small, presentation-safe changes**
- Preserve the distinction between:
  - extraction for ingestion
  - extraction for prompt design
- Prefer **readable, explicit code** over clever abstractions
- Treat the Streamlit app as a **live demo artifact**, not just an internal tool
- Preserve **inspectability** and **provenance**
- Do not remove or weaken the raw vs structured comparison
- Do not collapse deterministic logic, embeddings retrieval, and LLM synthesis into one opaque step

---

## File-specific guidance

### `src/app.py`
This is the highest-risk file for presentation stability.

Priorities:
- keep the **Demo: One-Prompt (Comparison)** tab stable
- avoid breaking existing tab behavior
- prefer minimal UI changes
- preserve clear labeling of:
  - raw context
  - structured context
  - naive output
  - grounded output
- keep Streamlit widget keys stable unless necessary
- handle optional/nullable UI values safely

### `src/extract.py`
Keep extraction logic conservative and explainable.

Priorities:
- explicit normalization
- stable typed output
- readable parsing helpers
- provenance via source snippets

Do not replace deterministic extraction with vague model-based logic unless explicitly requested.

### `src/query_demo.py`
This file handles deterministic constraints and explainability.

Priorities:
- preserve clear constraint parsing
- preserve explainability
- prefer correctness and readability over broader but brittle parsing

### `src/rag_index.py` and `src/rag_search.py`
These support the embeddings layer.

Priorities:
- keep embeddings support simple and reliable
- do not introduce unnecessary complexity
- keep the distinction clear:
  - extraction handles facts
  - embeddings handle semantic similarity

### `src/llm_answer.py`
This is the grounded narration layer.

Priorities:
- preserve grounding behavior
- preserve controlled use of extracted facts and snippet-backed evidence
- do not make the grounded path more speculative

---

## Preferred change style
When making changes:
- keep diffs narrow
- avoid renaming files or major functions unless necessary
- avoid introducing new dependencies unless requested
- avoid broad stylistic rewrites
- avoid changing presentation copy unless asked

When fixing bugs:
- prefer the smallest safe fix
- preserve behavior unless the bug requires behavior change
- note any user-visible demo impact

---

## Validation expectations
Before considering a task done, check the following when relevant:

- `src/app.py` still loads cleanly
- the **Demo: One-Prompt (Comparison)** path still works
- `out/pedals.jsonl` is still a valid input to the app
- embeddings-based retrieval still works when `out/embeddings.npz` exists
- nullable Streamlit widget values are handled safely
- no change weakens the raw vs structured comparison

If a change could affect the presentation flow, call that out explicitly.

---

## Presentation constraints
This repo is actively used in a live interview/demo setting.

Optimize for:
- clarity
- reliability
- legibility on screen
- easy explanation in front of an audience

Avoid:
- fragile refactors right before a demo
- hidden behavior changes
- over-automation that makes the system harder to explain
- anything that weakens the “same prompt, better system context” narrative

---

## Good task framing for this repo
Preferred requests look like:

- “Fix this type error without changing demo behavior.”
- “Refactor this block for readability, but preserve the live flow.”
- “Add guards for nullable Streamlit selections.”
- “Make this comparison clearer on screen.”
- “Improve explainability without changing the overall architecture.”

Less useful requests are broad prompts like:
- “Improve the app”
- “Refactor everything”
- “Make this more advanced”

---

## Short repo summary
Use this mental model when working here:

- **Extractors for facts**
- **Embeddings for vibes**
- **LLMs for synthesis**

And always preserve the main teaching chain:

**better extraction → better records → better prompt context → better downstream behavior**