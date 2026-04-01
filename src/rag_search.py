from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n


def load_index(path: Path) -> Tuple[List[str], np.ndarray]:
    data = np.load(path, allow_pickle=False)
    ids = data["ids"].tolist()
    embs = data["embs"]
    return ids, embs


def search(
    query: str,
    ids: List[str],
    embs: np.ndarray,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    model = SentenceTransformer(model_name)
    q = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
    q = normalize_vec(q)

    # cosine similarity because vectors are normalized
    sims = embs @ q
    idx = np.argsort(-sims)[:top_k]
    return [(ids[i], float(sims[i])) for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="out/embeddings.npz")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    ids, embs = load_index(Path(args.index))
    results = search(args.query, ids, embs, model_name=args.model, top_k=args.top_k)

    print(f'\n=== Sound search: "{args.query}" ===')
    for pid, score in results:
        print(f"- {pid:20s}  score={score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())