#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
search_documents.py
CLI: Semantic search over 'document_chunks' using Gemini embeddings + pgvector.

Example:
  python search_documents.py -q "reset procedure for device" -k 5
"""

import os
import argparse
from typing import List, Tuple

from dotenv import load_dotenv
import google.generativeai as genai
import psycopg2


def configure() -> str:
    """Load .env, configure Gemini, and return the Postgres URL."""
    load_dotenv()
    api = os.getenv("GEMINI_API_KEY")
    if not api:
        raise RuntimeError("GEMINI_API_KEY missing")
    genai.configure(api_key=api)

    url = os.getenv("POSTGRES_URL")
    if not url:
        raise RuntimeError("POSTGRES_URL missing")
    return url


def _extract_embedding_values(resp):
    """Return a flat list[float] from various possible google-generativeai response shapes."""
    # dict with "embedding" -> {"values": [...]}
    if isinstance(resp, dict) and "embedding" in resp:
        emb = resp["embedding"]
        if isinstance(emb, dict) and "values" in emb:
            return emb["values"]
        if isinstance(emb, list) and emb and isinstance(emb[0], float):
            return emb

    # dict with "embeddings" -> [{"values": [...]}, ...]
    if isinstance(resp, dict) and "embeddings" in resp:
        embs = resp["embeddings"]
        if isinstance(embs, list) and embs:
            first = embs[0]
            if isinstance(first, dict) and "values" in first:
                return first["values"]
            if isinstance(first, list) and first and isinstance(first[0], float):
                return first

    # object with .embedding
    if hasattr(resp, "embedding"):
        emb = getattr(resp, "embedding")
        if isinstance(emb, dict) and "values" in emb:
            return emb["values"]
        if isinstance(emb, list) and emb and isinstance(emb[0], float):
            return emb

    # plain list[float]
    if isinstance(resp, list) and resp and isinstance(resp[0], float):
        return resp

    raise RuntimeError(f"Unexpected embedding response format: {type(resp)} -> {resp}")


def embed_query(q: str, model: str = "text-embedding-004") -> List[float]:
    """Embed the query string for retrieval using a robust extractor."""
    resp = genai.embed_content(model=model, content=q, task_type="retrieval_query")
    return _extract_embedding_values(resp)



def search(url: str, qvec: List[float], k: int = 5) -> List[Tuple[float, str, str]]:
    """Return [(similarity, chunk_text, filename), ...] ranked by cosine similarity."""
    with psycopg2.connect(url) as conn:
        with conn.cursor() as cur:
            # cosine distance operator <=> (lower = closer); 1 - distance gives similarity
            cur.execute(
                """
                SELECT
                    1 - (embedding <=> %s::vector) AS similarity,
                    chunk_text,
                    filename
                FROM public.document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (qvec, qvec, k),
            )
            return cur.fetchall()


def main() -> None:
    """CLI entry point: embed the query, run vector search, print top-k results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", required=True, help="Search text")
    ap.add_argument("-k", type=int, default=5, help="Number of results to return")
    ap.add_argument("--model", default="text-embedding-004", help="Embedding model name")
    args = ap.parse_args()

    url = configure()
    qvec = embed_query(args.query, model=args.model)
    results = search(url, qvec, k=args.k)

    print(f"\nTop {args.k} results for: {args.query}\n" + "-" * 60)
    for i, (sim, text, fname) in enumerate(results, 1):
        preview = (text or "").replace("\n", " ")
        if len(preview) > 280:
            preview = preview[:277] + "..."
        print(f"{i}. [{sim:.4f}] {fname}\n   {preview}\n")


if __name__ == "__main__":
    main()
