from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def load_docs(data_dir: Path) -> List[Tuple[str, str]]:
    """
    Returns list of (id, text) where id is filename stem.
    """
    docs: List[Tuple[str, str]] = []
    for p in sorted(data_dir.glob("*.txt")):
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        docs.append((p.stem, text))
    if not docs:
        raise SystemExit(f"No .txt files found in {data_dir}")
    return docs


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/pedals", help="Directory of pedal note .txt files")
    ap.add_argument("--out_dir", default="out", help="Output directory")
    ap.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    ap.add_argument("--max_chars", type=int, default=6000, help="Limit doc text length to embed")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs(data_dir)
    ids = [doc_id for doc_id, _ in docs]
    texts = [txt[: args.max_chars] for _, txt in docs]

    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"Embedding {len(texts)} docs...")
    embs = model.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True)
    embs = embs.astype(np.float32)
    embs_norm = normalize_rows(embs)

    out_path = out_dir / "embeddings.npz"
    np.savez_compressed(out_path, ids=np.array(ids), embs=embs_norm)

    print(f"Wrote {out_path} ({embs_norm.shape[0]} vectors, dim={embs_norm.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())