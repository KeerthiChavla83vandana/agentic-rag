"""
WMF FPO Intelligence Platform — FastAPI Application Entry Point.

This module defines the FastAPI app instance, startup lifecycle logic,
and all API route handlers for the WMF QA RAG pipeline.

API Route Groups:
    Health Checks:
        GET  /          → Returns service name, version, embedding model info.

    Data Processing:
        POST /wmf_qa/upload  → Upload + parse + embed a document.

    Pipeline Execution:
        POST /wmf_qa/query   → Full end-to-end Agentic CRAG pipeline.

    Debug Endpoints (isolated pipeline stage testing):
        POST /wmf_qa/route   → Test the adaptive query router only.
        POST /wmf_qa/search  → Test hybrid retrieval only.
        POST /wmf_qa/grade   → Test chunk relevance grader only.
        POST /wmf_qa/rewrite → Test query reformulator only.
        POST /wmf_qa/ingest  → Test Docling parsing only (returns markdown).

Architecture Notes:
    - All route handlers are `async def` — they yield control to the event loop
      while waiting for DB queries, LLM calls, and file I/O.
    - The database session is provided via Depends(get_db) which yields an
      AsyncSession from the async SQLAlchemy session factory.
    - Database tables and indexes are created at startup via the `lifespan`
      context manager (the modern FastAPI alternative to @app.on_event("startup")).
"""

import logging
import os
import shutil
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
)

from app.core.config import settings
from app.core.database import AsyncSessionLocal, Base, engine, get_db
from app.pipeline.agent import execute_agentic_rag
from app.pipeline.embedding import embed_and_store_document
from app.pipeline.grader import grade_chunk, rewrite_query
from app.pipeline.ingestion import ingest_document
from app.pipeline.retrieval import hybrid_retrieve
from app.pipeline.router import route_query


# =============================================================================
# Application Lifespan — Database Initialization
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager — runs startup and shutdown logic.

    Startup tasks (before `yield`):
        1. Enable pgvector extension in PostgreSQL.
        2. Create all ORM tables (documents, document_chunks) if they don't exist.
        3. Create HNSW vector index (for fast cosine similarity search).
        4. Create GIN index on the tsvector column (for fast FTS keyword search).
        5. Create index on document_id FK (for fast chunk lookups by document).

    Why create indexes explicitly here?
        SQLAlchemy's `create_all()` creates tables based on ORM model definitions
        but does NOT automatically create custom index types like HNSW or GIN.
        We issue raw SQL for those after table creation.

    Shutdown tasks (after `yield`):
        - Dispose the connection pool to cleanly release all DB connections.
    """
    print("=== WMF QA Backend Starting Up ===")

    async with engine.begin() as conn:
        # Enable pgvector — required before any VECTOR(n) column operations
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        print("✓ pgvector extension enabled.")

        # Create all tables declared in ORM models (idempotent — IF NOT EXISTS)
        await conn.run_sync(Base.metadata.create_all)
        print("✓ ORM tables initialized.")

        # HNSW index for approximate nearest-neighbor vector search.
        # vector_cosine_ops = cosine distance metric, matching our retrieval query.
        # IF NOT EXISTS prevents errors on container restart.
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
            ON document_chunks
            USING hnsw (embedding vector_cosine_ops);
        """))
        print("✓ HNSW vector index created/verified.")

        # GIN index on the generated tsvector column for fast FTS.
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS document_chunks_fts_idx
            ON document_chunks
            USING gin (tsvector_content_english);
        """))
        print("✓ GIN full-text search index created/verified.")

        # B-tree index on document_id FK for fast chunk lookups per document.
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS document_chunks_doc_id_idx
            ON document_chunks (document_id);
        """))
        print("✓ document_id FK index created/verified.")

    print("=== WMF QA Backend Ready ===")
    yield

    # Shutdown: cleanly close all DB connections in the pool.
    await engine.dispose()
    print("=== WMF QA Backend Shutting Down ===")


# =============================================================================
# FastAPI Application Instance
# =============================================================================

app = FastAPI(
    title="WMF FPO Intelligence Platform — QA Module",
    description=(
        "Standalone Advanced RAG pipeline with Adaptive Routing, "
        "Corrective Retrieval Loop, and Citation-Aware Synthesis. "
        "Built for the Vrutti FPO Initiative (Walmart Foundation)."
    ),
    version=settings.backend_version,
    lifespan=lifespan,
)


# =============================================================================
# Pydantic Request Schemas
# =============================================================================

class QueryRequest(BaseModel):
    """Request body for the full CRAG pipeline and router/rewrite debug endpoints."""
    query: str
    """Natural language question (English, Kannada, or mixed)."""
    persona: str = "Untailored"
    """Target user role: 'Program Lead', 'Junior Associate', 'Field Enumerator', 'Untailored'."""


class SearchRequest(BaseModel):
    """Request body for the isolated hybrid search debug endpoint."""
    query: str
    """Query string to embed and search."""
    limit: int = 5
    """Number of chunks to return (default: 5)."""


class GradeRequest(BaseModel):
    """Request body for the isolated chunk grader debug endpoint."""
    query: str
    """The query to grade the chunk against."""
    chunk_content: str
    """The raw text content of the retrieved chunk."""


# =============================================================================
# Health Check
# =============================================================================

@app.get("/", tags=["Health"])
async def read_root() -> dict[str, Any]:
    """Service health check and identity endpoint.

    Returns:
        JSON with backend name, version, status, and active embedding model.
        Used by the Streamlit frontend to verify backend connectivity on startup.
    """
    return {
        "backend": settings.backend_name,
        "version": settings.backend_version,
        "status": "online",
        "embedding_model": settings.embedding_model_name,
    }


# =============================================================================
# Debug Endpoints — Test individual pipeline stages in isolation
# =============================================================================

@app.post("/wmf_qa/route", tags=["Debug Endpoints"])
async def api_route_query(request: QueryRequest) -> dict[str, Any]:
    """Debug the adaptive query router in isolation.

    Classifies the query as 'vector_db' or 'general_knowledge'
    without running retrieval or synthesis.

    Returns:
        JSON with the original query and the routing decision.
    """
    route = await route_query(request.query)
    return {"query": request.query, "route": route}


@app.post("/wmf_qa/search", tags=["Debug Endpoints"])
async def api_search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Debug the hybrid retrieval pipeline in isolation.

    Runs dense + sparse search and RRF fusion without grading or synthesis.
    Useful for validating that documents are indexed correctly.

    Returns:
        JSON with the query and the top-N fused chunk results.
    """
    try:
        results = await hybrid_retrieve(db, request.query, limit=request.limit)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/wmf_qa/grade", tags=["Debug Endpoints"])
async def api_grade_chunk(request: GradeRequest) -> dict[str, Any]:
    """Debug the LLM-based chunk relevance grader in isolation.

    Grades a single (query, chunk) pair without running retrieval.

    Returns:
        JSON with the query, chunk preview, and grade ('YES' or 'NO').
    """
    grade = await grade_chunk(request.query, request.chunk_content)
    return {"query": request.query, "grade": grade}


@app.post("/wmf_qa/rewrite", tags=["Debug Endpoints"])
async def api_rewrite_query(request: QueryRequest) -> dict[str, Any]:
    """Debug the LLM-based query reformulator in isolation.

    Reformulates a query to maximize retrieval recall without running retrieval.
    Useful for checking that Gemini correctly expands domain-specific queries.

    Returns:
        JSON with the original and reformulated query strings.
    """
    new_query = await rewrite_query(request.query)
    return {"original_query": request.query, "reformulated_query": new_query}


@app.post("/wmf_qa/ingest", tags=["Debug Endpoints"])
async def api_ingest_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Debug Docling document parsing in isolation (no embedding, no storage).

    Useful for verifying that a specific PDF/DOCX/PPTX file can be parsed
    correctly by Docling before committing to full indexing.

    Returns:
        JSON with the filename and the full Markdown export of the parsed document.
    """
    upload_dir = "data"
    os.makedirs(upload_dir, exist_ok=True)
    temp_path = os.path.join(upload_dir, file.filename)

    try:
        # Save file temporarily for Docling (Docling needs a file path, not bytes)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        docling_document = await ingest_document(temp_path)
        markdown = docling_document.export_to_markdown()

        return {"filename": file.filename, "markdown": markdown}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up temp file regardless of success or failure
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =============================================================================
# Main Pipeline Endpoints
# =============================================================================

@app.post("/wmf_qa/query", tags=["Pipeline Execution"])
async def api_query_agent(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Execute the full Corrective Agentic RAG (CRAG) pipeline.

    Orchestrates: routing → retrieval → grading → optional reformulation
                  → synthesis → citation verification.

    Args:
        request: JSON body with `query` and optional `persona`.
        db:      Async DB session injected by FastAPI dependency.

    Returns:
        Structured JSON containing:
            - response: Final cited answer text.
            - route, search_query, corrective_action_taken: Pipeline trace.
            - retrieved_chunks: All chunks with relevance grades (for Streamlit debug).
            - citations_verified: List of verified chunk IDs cited in the response.
    """
    try:
        result = await execute_agentic_rag(db, request.query, request.persona)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/wmf_qa/upload", tags=["Data Processing"])
async def upload_and_process_file(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    folder_name: str | None = Form(None),
    subfolder_name: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Upload a document, parse it with Docling, embed chunks, and store in PostgreSQL.

    Full pipeline:
        1. Save uploaded file to disk (Docling requires a file path).
        2. Parse with Docling DocumentConverter → DoclingDocument.
        3. Chunk with HybridChunker + contextualize() → semantic chunks.
        4. Apply RecursiveCharacterTextSplitter if any chunk > 1500 tokens (Option B).
        5. Embed each chunk with BAAI/bge-m3 → 1024-dim vectors.
        6. Store Document + DocumentChunk rows in PostgreSQL.

    Args:
        file:          The uploaded file (PDF, DOCX, or PPTX).
        document_id:   Unique string identifier for this document (caller-defined).
        folder_name:   Optional high-level grouping tag.
        subfolder_name: Optional secondary grouping tag.
        db:            Async DB session from dependency injection.

    Returns:
        JSON with document_id, filename, chunk count, and status="success".

    Raises:
        HTTPException 500: If Docling parsing or embedding fails.
    """
    upload_dir = "data"
    os.makedirs(upload_dir, exist_ok=True)
    temp_path = os.path.join(upload_dir, file.filename)

    try:
        # Step 1: Write the uploaded file to disk
        # (Docling's converter requires a file path, not raw bytes)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 2: Parse with Docling (runs in thread pool — non-blocking)
        docling_document = await ingest_document(temp_path)

        # Step 3–6: Chunk → embed → store (all in embedding.py)
        chunks_count = await embed_and_store_document(
            db=db,
            document_id=document_id,
            source_file=file.filename,
            file_path=temp_path,
            docling_document=docling_document,
            folder_name=folder_name,
            subfolder_name=subfolder_name,
        )

        return {
            "document_id": document_id,
            "filename": file.filename,
            "chunks_extracted": chunks_count,
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File processing failed: {e}",
        )

    finally:
        # Always clean up temp file to avoid disk accumulation
        if os.path.exists(temp_path):
            os.remove(temp_path)
