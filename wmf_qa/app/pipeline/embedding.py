"""
Document chunking, embedding, and storage pipeline.

Pipeline flow for a single document:
    DoclingDocument
        │
        ▼
    [1] HybridChunker (Docling)
        - Understands document structure: headings, tables, paragraphs, lists.
        - Produces semantically coherent chunks respecting the layout hierarchy.
        │
        ▼
    [2] contextualize(chunk)
        - Prepends the document hierarchy path (e.g. "Section 2 > 2.1 Targets")
          to each chunk's text before embedding.
        - This enriches the vector with document-level structural context,
          dramatically improving semantic search quality.
        │
        ▼
    [3] Token-length safety check (Option B chunking)
        - If a contextualized chunk exceeds MAX_CHUNK_TOKENS tokens
          (e.g. large OCR-output blocks or long tables), it is further split
          using RecursiveCharacterTextSplitter with overlap.
        - This is a safety net that future-proofs the pipeline for OCR output,
          which can produce very large undivided text blocks.
        │
        ▼
    [4] BAAI/bge-m3 SentenceTransformer embedding
        - Produces 1024-dimensional L2-normalized vectors.
        - The model runs in a thread pool executor (run_in_executor) so
          the asyncio event loop is not blocked during GPU/CPU inference.
        │
        ▼
    [5] Store DocumentChunk rows in PostgreSQL via AsyncSession.
        - Chunk ID: SHA-256 hash of "{document_id}_ch_{index}".
        - Stores: content, embedding, structured metadata_bucket.

Chunking Strategy Rationale (Option B — selected):
    Primary:  Docling HybridChunker preserves document structure (best for PDFs/DOCX).
    Fallback: RecursiveCharacterTextSplitter handles oversized chunks safely.
    Overlap:  200 tokens of overlap between sub-chunks prevents context cutoffs.
    Why not RecursiveCharacterTextSplitter alone?
        It ignores document structure (headings, tables) and would split across
        section boundaries, degrading retrieval quality for structured FPO reports.
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from docling.chunking import HybridChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Maximum number of whitespace-delimited tokens a contextualized chunk may have
# before triggering the RecursiveCharacterTextSplitter safety net.
# 350 words ≈ ~450 sub-word tokens, keeping it safely under the 512-token limit of the embedding model.
MAX_CHUNK_TOKENS: int = 350

# Sub-chunk overlap in tokens when RecursiveCharacterTextSplitter is applied.
# 50-token overlap prevents context from being cut off at chunk boundaries.
CHUNK_OVERLAP: int = 50

# ---------------------------------------------------------------------------
# Embedding model cache
# ---------------------------------------------------------------------------
# SentenceTransformer is expensive to load (~2.5 GB for BAAI/bge-m3).
# We cache the instance at module level so it is loaded exactly once per
# process lifetime (pre-downloaded in Docker build layer).
_model_cache: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return a cached SentenceTransformer model instance.

    The model name is read from settings.embedding_model_name
    (default: 'BAAI/bge-m3').

    BAAI/bge-m3 was chosen because:
    - 1024-dimensional vectors for rich semantic representation.
    - Native multilingual support including English and Kannada.
    - Strong performance on both dense retrieval and re-ranking benchmarks.
    - Produced by BAAI (Beijing Academy of AI), widely used in production RAG systems.

    Returns:
        A loaded SentenceTransformer model instance (cached after first call).
    """
    global _model_cache
    if _model_cache is None:
        logger.info(
            "Loading SentenceTransformer model: %s (first-time only — model is cached after this).",
            settings.embedding_model_name,
        )
        _model_cache = SentenceTransformer(settings.embedding_model_name)
    return _model_cache


# ---------------------------------------------------------------------------
# Internal helper: RecursiveCharacterTextSplitter safety net
# ---------------------------------------------------------------------------

def _split_oversized_chunk(text: str) -> list[str]:
    """Split a single oversized text block into smaller sub-chunks with overlap.

    This is the Option B safety net. It is only invoked when a chunk from
    Docling's HybridChunker exceeds MAX_CHUNK_TOKENS tokens (word count estimate).

    We use a word-count heuristic (len(text.split())) rather than a full
    tokenizer for speed. For a rough equivalence: 1 word ≈ 1.3 sub-word tokens.

    Args:
        text: The oversized chunk text to split further.

    Returns:
        A list of sub-chunk strings, each under MAX_CHUNK_TOKENS word-tokens.
        Always returns at least one element (the original text if splitting fails).
    """
    splitter = RecursiveCharacterTextSplitter(
        # chunk_size in characters; ~1500 words ≈ 7500–9000 characters for English.
        chunk_size=MAX_CHUNK_TOKENS * 5,  # rough char estimate: avg 5 chars per word
        chunk_overlap=CHUNK_OVERLAP * 5,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    sub_chunks = splitter.split_text(text)
    # Filter out trivially short fragments (less than 20 characters)
    return [s for s in sub_chunks if len(s.strip()) >= 20] or [text]


def _run_embed(model: SentenceTransformer, text: str) -> list[float]:
    """Synchronously generate an embedding vector for a single text (thread-pool target).

    This function is CPU/GPU bound and blocking. It is always called via
    asyncio's run_in_executor() so the event loop is not blocked.

    Args:
        model: The loaded SentenceTransformer model.
        text:  The text string to encode.

    Returns:
        A list of 1024 floats representing the L2-normalized embedding vector.
    """
    # normalize_embeddings=True ensures vectors have unit length,
    # making cosine similarity equivalent to dot product — required for pgvector.
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


# ---------------------------------------------------------------------------
# Public async function: embed_and_store_document()
# ---------------------------------------------------------------------------

async def embed_and_store_document(
    db: AsyncSession,
    document_id: str,
    source_file: str,
    file_path: str,
    docling_document: Any,
    folder_name: str | None = None,
    subfolder_name: str | None = None,
    tagging_metadata: dict[str, Any] | None = None,
) -> int:
    """Chunk a DoclingDocument, compute embeddings, and persist to PostgreSQL.

    This is the main entry point called from the /wmf_qa/upload route handler.
    It orchestrates the full pipeline: chunk → validate → embed → store.

    Args:
        db:               Active AsyncSession from the get_db() dependency.
        document_id:      Caller-defined unique document identifier string.
        source_file:      Original filename (e.g. 'fpo_report_2025.pdf').
        file_path:        Disk path where the uploaded file was saved.
        docling_document: Parsed DoclingDocument from ingestion.ingest_document().
        folder_name:      Optional grouping label (e.g. 'Vrutti_FPO_RAG').
        subfolder_name:   Optional secondary grouping label.
        tagging_metadata: Optional caller-supplied dict stored in metadata_bucket.user.

    Returns:
        Number of chunk rows successfully stored in the database.

    Raises:
        Exception: Any DB or encoding error. The document status is set to
                   'failed' before re-raising so the caller can detect it.

    Pipeline steps:
        1. Upsert the Document row (create if new, reset status if re-indexing).
        2. Run HybridChunker on the DoclingDocument.
        3. For each chunk, apply contextualize() then check token length.
        4. If oversized, split further via RecursiveCharacterTextSplitter.
        5. Generate embedding in thread pool (avoids blocking event loop).
        6. Build the structured metadata_bucket JSON.
        7. Batch-insert all DocumentChunk rows and commit.
    """
    logger.info("Starting embedding pipeline for document_id='%s'", document_id)

    # -------------------------------------------------------------------------
    # Step 1: Upsert the parent Document record
    # -------------------------------------------------------------------------
    result = await db.execute(
        select(Document).where(Document.document_id == document_id)
    )
    doc = result.scalar_one_or_none()

    if not doc:
        # First time ingesting this document — create a new record.
        doc = Document(
            document_id=document_id,
            source_file=source_file,
            file_path=file_path,
            folder_name=folder_name,
            subfolder_name=subfolder_name,
            tagging_metadata=tagging_metadata or {},
            embedding_status="pending",
        )
        db.add(doc)
    else:
        # Re-indexing an existing document — reset status to allow re-run.
        doc.embedding_status = "pending"

    await db.commit()
    await db.refresh(doc)

    try:
        # -------------------------------------------------------------------------
        # Step 2: Chunk document with Docling HybridChunker
        # -------------------------------------------------------------------------
        # HybridChunker understands the document's internal structure (headings,
        # tables, paragraph boundaries) and produces layout-aware semantic chunks.
        chunker = HybridChunker()
        raw_chunks = list(chunker.chunk(docling_document))

        if not raw_chunks:
            logger.warning("HybridChunker produced 0 chunks for document '%s'. Marking complete.", document_id)
            doc.embedding_status = "completed"
            doc.embedded_at = datetime.now(tz=timezone.utc)
            await db.commit()
            return 0

        logger.info(
            "HybridChunker produced %d raw chunks for document '%s'.",
            len(raw_chunks),
            document_id,
        )

        # -------------------------------------------------------------------------
        # Step 3: Expand chunks (contextualize → optional sub-split)
        # -------------------------------------------------------------------------
        # final_chunks is a list of (text, page_no) tuples ready for embedding.
        final_chunks: list[tuple[str, int]] = []

        for raw_chunk in raw_chunks:
            # contextualize() prepends the document hierarchy path to the chunk text.
            # e.g. "Section 3 > 3.2 Budget Allocations\n\nActual content here..."
            # This gives the embedding model structural context beyond raw text.
            contextualized_text = chunker.contextualize(raw_chunk)

            # Extract page number from Docling provenance metadata (best-effort).
            page_no = 1
            if getattr(raw_chunk, "meta", None) and getattr(raw_chunk.meta, "doc_items", None):
                for item in raw_chunk.meta.doc_items:
                    if getattr(item, "prov", None):
                        for prov in item.prov:
                            if getattr(prov, "page_no", None) is not None:
                                page_no = prov.page_no
                                break
                    if page_no != 1:
                        break

            # -----------------------------------------------------------------------
            # Option B: Token-length safety net
            # -----------------------------------------------------------------------
            # Word count heuristic to detect oversized chunks.
            word_count = len(contextualized_text.split())

            if word_count > MAX_CHUNK_TOKENS:
                # The chunk exceeds the safe size limit (common for large tables or
                # OCR-output blocks). Split it into overlapping sub-chunks.
                logger.debug(
                    "Chunk exceeds %d words (%d words). Applying RecursiveCharacterTextSplitter.",
                    MAX_CHUNK_TOKENS,
                    word_count,
                )
                sub_texts = _split_oversized_chunk(contextualized_text)
                for sub_text in sub_texts:
                    final_chunks.append((sub_text, page_no))
            else:
                # Chunk is within acceptable size — use it as-is.
                final_chunks.append((contextualized_text, page_no))

        logger.info(
            "After contextualization and safety splitting: %d final chunks to embed.",
            len(final_chunks),
        )

        # -------------------------------------------------------------------------
        # Step 4: Clear old chunks for this document (support re-indexing)
        # -------------------------------------------------------------------------
        # Delete any previously stored chunks so re-uploading the same document
        # produces a clean, consistent state rather than duplicates.
        await db.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )

        # -------------------------------------------------------------------------
        # Step 5: Embed and store each chunk
        # -------------------------------------------------------------------------
        model = get_embedding_model()
        loop = asyncio.get_event_loop()
        chunks_saved = 0

        for idx, (text, page_no) in enumerate(final_chunks):
            # ------------------------------------------------------------------
            # Embed in thread pool to avoid blocking the asyncio event loop.
            # run_in_executor submits _run_embed() to the default ThreadPoolExecutor.
            # ------------------------------------------------------------------
            embedding = await loop.run_in_executor(None, _run_embed, model, text)

            # Stable, deterministic chunk ID using SHA-256 hash.
            # Using idx in the seed ensures uniqueness within a document.
            raw_id = f"{document_id}_ch_{idx}"
            chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

            # ------------------------------------------------------------------
            # Build structured metadata bucket
            # ------------------------------------------------------------------
            # The bucket is split into three namespaced sub-dicts to keep
            # system metadata, document metadata, and user metadata separate.
            metadata_bucket: dict[str, Any] = {
                "system": {
                    "chunk_id": chunk_id,
                    "chunk_number": idx + 1,          # 1-indexed for human readability
                    "total_chunks": len(final_chunks), # useful for frontend display
                    "source_file_name": source_file,
                    "page_number": page_no,
                    "word_count": len(text.split()),   # approximate token count
                },
                "document": {
                    "document_title": source_file,
                    "folder_name": folder_name,
                    "subfolder_name": subfolder_name,
                },
                "user": tagging_metadata or {},
            }

            db_chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                content=text,
                embedding=embedding,
                metadata_bucket=metadata_bucket,
            )
            db.add(db_chunk)
            chunks_saved += 1

        # -------------------------------------------------------------------------
        # Step 6: Finalize — update document status and commit all chunks at once
        # -------------------------------------------------------------------------
        doc.embedding_status = "completed"
        doc.embedded_at = datetime.now(tz=timezone.utc)
        await db.commit()

        logger.info(
            "Successfully embedded and stored %d chunks for document '%s'.",
            chunks_saved,
            document_id,
        )
        return chunks_saved

    except Exception as e:
        logger.error(
            "Embedding pipeline failed for document '%s': %s", document_id, str(e)
        )
        # Mark document as failed so the caller and frontend can detect the error.
        doc.embedding_status = "failed"
        await db.commit()
        raise
