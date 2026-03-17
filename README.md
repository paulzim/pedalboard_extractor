# Pedalboard Extractor Demo (RAG-friendly)

This repo demonstrates **extractors**: code that turns messy gear text into structured JSON you can reliably filter on.

## Quickstart

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

python -m src.extract --data_dir data/pedals --out_dir out
python -m src.query_demo --records out/pedals.jsonl