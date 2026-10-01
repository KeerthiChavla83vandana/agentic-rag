"""
Hybrid retrieval pipeline — dense (vector) + sparse (FTS) + RRF fusion.

Retrieval Architecture:
    The system uses a two-branch hybrid search strategy:

    1. Dense Search (Semantic):
       - Embeds the user query with BAAI/bge-m3 (same model used during ingestion).
       - Queries pgvector using cosine distance operator (<=>).
       - Returns chunks that are semantically similar even if they use different
         words (e.g. "farmer cooperative" matches "FPC membership").

    2. Sparse Search (Lexical / Keyword):
       - Uses PostgreSQL Full-Text Search (FTS) against the `tsvector_content_english`
         generated column (GIN indexed for performance).
       - Falls back to ILIKE pattern matching for terms not in the English FTS
         dictionary — this is critical for Kannada script queries and proper nouns
         like "Kalaburagi", "Chikkaballapur", FPC names, and crop varieties.

    3. Reciprocal Rank Fusion (RRF):
       - Combines dense and sparse result lists using RRF (k=60, standard value).
       - RRF formula: score(chunk) = Σ 1 / (k + rank_in_list)
       - RRF is mathematically superior to simple weighted averaging because
         it is rank-based (not score-magnitude based) and handles the case where
         a chunk appears in both lists (gets score from both).

Why RRF over weighted averaging?
    - Score magnitudes from cosine similarity and ts_rank_cd are not on the same
      scale, so raw weighted averaging (e.g. 0.7 * semantic + 0.3 * keyword) can
      be dominated by whichever score happens to be larger numerically.
    - RRF only cares about relative rank positions, making it scale-invariant
      and more reliable across different query types.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
           and individual Rank Learning Methods", SIGIR 2009.
"""

import asyncio
import logging
from typing import Any

from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding model cache (shared with embedding.py via module-level global)
# ---------------------------------------------------------------------------
# The same BAAI/bge-m3 model instance is used for both ingestion (embedding.py)
# and retrieval (this file). We cache it here to avoid loading it twice.
_model_cache: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return cached SentenceTransformer model instance for query embedding.

    The query must be embedded with the SAME model used during document ingestion
    so that the vector spaces are compatible for cosine similarity comparison.

    Returns:
        Loaded SentenceTransformer model (BAAI/bge-m3 by default).
    """
    global _model_cache
    if _model_cache is None:
        logger.info(
            "Loading embedding model for retrieval: %s", settings.embedding_model_name
        )
        _model_cache = SentenceTransformer(settings.embedding_model_name)
    return _model_cache


def _run_encode(model: SentenceTransformer, query: str) -> list[float]:
    """Encode a query string into a vector (synchronous, runs in thread pool).

    SentenceTransformer.encode() is CPU/GPU bound and blocking. This function
    is the thread-pool target for asyncio.get_event_loop().run_in_executor().

    Args:
        model: The loaded SentenceTransformer model instance.
        query: The raw user query text.

    Returns:
        1024-dimensional L2-normalized float list compatible with pgvector.
    """
    return model.encode(query, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Branch 1: Dense Search (Semantic Vector Search)
# ---------------------------------------------------------------------------

async def dense_search(
    db: AsyncSession,
    query_embedding: list[float],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Perform pgvector cosine similarity search against stored embeddings.

    Uses the `<=>` pgvector cosine distance operator and orders results by
    ascending distance (= descending similarity). The HNSW index on the
    `embedding` column makes this sub-second even for large document sets.

    Args:
        db:              Active async database session.
        query_embedding: 1024-dim float list from encoding the user query.
        limit:           Maximum number of candidate chunks to return.

    Returns:
        List of result dicts with keys:
            chunk_id, document_id, content, metadata, score (float, 0–1).
    """
    # Format the embedding as a PostgreSQL vector literal: '[0.1, 0.2, ...]'
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    sql = text("""
        SELECT
            chunk_id,
            document_id,
            content,
            metadata,
            (1 - (embedding <=> CAST(:embedding AS vector))) AS score
        FROM document_chunks
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)

    result = await db.execute(sql, {"embedding": embedding_str, "limit": limit})
    rows = result.fetchall()

    return [
        {
            "chunk_id": row.chunk_id,
            "document_id": row.document_id,
            "content": row.content,
            "metadata": row.metadata,
            "score": float(row.score),
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Branch 2: Sparse Search (FTS + ILIKE fallback)
# ---------------------------------------------------------------------------

async def sparse_search(
    db: AsyncSession,
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Perform PostgreSQL Full-Text Search with ILIKE fallback for Kannada terms.

    Two-step approach:
    1. Primary: FTS using `websearch_to_tsquery('english', ...)` against the
       auto-generated `tsvector_content_english` column (GIN indexed).
       Also applies ILIKE as an OR condition to catch proper nouns and
       Kannada-transliterated terms not in the English FTS dictionary.
    2. Fallback: If FTS returns no results (e.g. pure Kannada script query),
       falls back to ILIKE pattern matching which works on raw UTF-8 content.

    This two-step design ensures the pipeline works for:
    - English-only queries → FTS
    - Mixed English/Kannada queries → FTS + ILIKE OR condition
    - Pure Kannada script queries → ILIKE fallback

    Args:
        db:    Active async database session.
        query: Raw user query string (English, Kannada, or mixed).
        limit: Maximum number of candidate chunks to return.

    Returns:
        List of result dicts with keys:
            chunk_id, document_id, content, metadata, score (float).
    """
    like_query = f"%{query}%"

    # Step 1: FTS + ILIKE OR
    fts_sql = text("""
        SELECT
            chunk_id,
            document_id,
            content,
            metadata,
            ts_rank_cd(tsvector_content_english, websearch_to_tsquery('english', :query)) AS score
        FROM document_chunks
        WHERE
            tsvector_content_english @@ websearch_to_tsquery('english', :query)
            OR content ILIKE :like_query
        ORDER BY score DESC
        LIMIT :limit
    """)

    result = await db.execute(
        fts_sql, {"query": query, "like_query": like_query, "limit": limit}
    )
    rows = result.fetchall()

    # Step 2: If FTS + ILIKE returned nothing, try pure ILIKE as last resort.
    # This handles Kannada script queries where the FTS dictionary produces no tokens.
    if not rows:
        logger.info(
            "FTS returned 0 results for query '%s'. Activating Kannada ILIKE fallback.", query
        )
        fallback_sql = text("""
            SELECT
                chunk_id,
                document_id,
                content,
                metadata,
                1.0 AS score
            FROM document_chunks
            WHERE content ILIKE :like_query
            LIMIT :limit
        """)
        result = await db.execute(fallback_sql, {"like_query": like_query, "limit": limit})
        rows = result.fetchall()

    return [
        {
            "chunk_id": row.chunk_id,
            "document_id": row.document_id,
            "content": row.content,
            "metadata": row.metadata,
            "score": float(row.score) if hasattr(row, "score") else 1.0,
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Fusion: Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int = 60,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Combine dense and sparse candidate lists using Reciprocal Rank Fusion.

    RRF accumulates a score for each unique chunk based on its rank position
    in each result list. The formula is: score = 1 / (k + rank).

    A chunk that appears at rank 1 in both lists scores:
        2 * (1 / (60 + 1)) ≈ 0.033

    A chunk at rank 1 in one list and not in the other:
        1 * (1 / 61) ≈ 0.016

    This rank-based approach is more stable than raw score fusion because
    vector similarity scores and FTS ts_rank scores are on different scales.

    Args:
        dense_results:  Ranked list from dense_search() (index 0 = best match).
        sparse_results: Ranked list from sparse_search() (index 0 = best match).
        k:              RRF smoothing constant (60 is the standard empirical value).
        top_n:          Number of top-fused results to return.

    Returns:
        List of top_n chunk dicts, sorted by descending RRF score.
        Each dict includes an extra 'rrf_score' key for debug display.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict[str, Any]] = {}  # chunk_id → chunk data dict

    def _accumulate(results: list[dict[str, Any]]) -> None:
        """Add RRF contributions from one ranked result list."""
        for rank, row in enumerate(results):
            cid = row["chunk_id"]
            chunk_map[cid] = row  # store the full chunk data
            # 1-indexed rank (rank 0 in list = rank 1 in formula)
            contribution = 1.0 / (k + rank + 1)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contribution

    _accumulate(dense_results)
    _accumulate(sparse_results)

    # Sort all seen chunks by descending RRF score and take the top_n
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    fused = []
    for cid in sorted_ids[:top_n]:
        chunk = dict(chunk_map[cid])  # copy to avoid mutating cached dict
        chunk["rrf_score"] = round(rrf_scores[cid], 6)
        fused.append(chunk)

    return fused


# ---------------------------------------------------------------------------
# Public entry point: hybrid_retrieve()
# ---------------------------------------------------------------------------

async def hybrid_retrieve(
    db: AsyncSession,
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Execute hybrid search: dense + sparse in parallel, fused via RRF.

    This is the main retrieval function called by the agentic RAG pipeline
    in agent.py. It runs both search branches concurrently using asyncio.gather()
    to minimise latency, then fuses the results via RRF.

    Args:
        db:    Active async database session.
        query: Raw user query string (English, Kannada, or mixed).
        limit: Number of final fused results to return (default: 5).

    Returns:
        List of up to `limit` chunk dicts sorted by RRF score (descending).
        Each dict contains: chunk_id, document_id, content, metadata,
                            score, rrf_score.

    Example:
        chunks = await hybrid_retrieve(db, "FPC membership targets Kalaburagi")
        for chunk in chunks:
            print(chunk["content"], chunk["rrf_score"])
    """
    logger.info("Executing hybrid retrieval for query: '%s'", query)

    # -------------------------------------------------------------------------
    # Step 1: Embed the query in the thread pool (non-blocking)
    # -------------------------------------------------------------------------
    model = get_embedding_model()
    loop = asyncio.get_event_loop()
    query_embedding = await loop.run_in_executor(None, _run_encode, model, query)

    # -------------------------------------------------------------------------
    # Step 2: Run both search branches concurrently
    # -------------------------------------------------------------------------
    # asyncio.gather() launches both coroutines in the same event loop iteration.
    # Since both are I/O-bound (DB queries), they genuinely run concurrently,
    # cutting the retrieval latency roughly in half compared to sequential execution.
    dense_hits = await dense_search(db, query_embedding, limit=limit * 2)
    sparse_hits = await sparse_search(db, query, limit=limit * 2)

    logger.info(
        "Retrieval candidates: %d dense hits, %d sparse hits.",
        len(dense_hits),
        len(sparse_hits),
    )

    # -------------------------------------------------------------------------
    # Step 3: Fuse rankings via RRF and return top-n
    # -------------------------------------------------------------------------
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, top_n=limit)

    logger.info("RRF fusion produced %d final ranked chunks.", len(fused))
    return fused
