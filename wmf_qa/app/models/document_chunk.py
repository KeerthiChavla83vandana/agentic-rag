"""
SQLAlchemy ORM model for the `document_chunks` table.

Each row represents one semantic chunk of a parsed document.
Chunks carry a 1024-dimensional vector embedding (BAAI/bge-m3),
the raw contextualized text, a generated tsvector column for
Full-Text Search (FTS), and a structured metadata bucket.

Table schema is also maintained in: sql/ddl/document_chunks.sql
"""

from __future__ import annotations

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class DocumentChunk(Base):
    """SQLAlchemy model for a single document chunk with vector embedding.

    A chunk is produced by Docling's HybridChunker (with contextual enrichment
    via `contextualize()`) and optionally further split by RecursiveCharacterTextSplitter
    if the raw chunk exceeds 1500 tokens.

    Attributes:
        chunk_id:                   SHA-256 derived unique 16-char hex identifier.
        document_id:                FK reference to the parent Document.
        content:                    The contextualized text that was embedded.
        embedding:                  1024-dimensional float vector (BAAI/bge-m3).
        tsvector_content_english:   PostgreSQL generated tsvector column for FTS.
                                    Auto-populated by the DB from `content` using
                                    the 'english' dictionary. Supports GIN indexing.
        metadata_bucket:            Structured JSON with three nested keys:
                                      - system:   chunk_id, chunk_number, source_file, page_number
                                      - document: document_title, folder_name
                                      - user:     custom tagging dict from the caller
        document:                   Many-to-one relationship back to the parent Document.
    """

    __tablename__ = "document_chunks"

    # ---------------------------------------------------------------------------
    # Primary Key
    # ---------------------------------------------------------------------------
    chunk_id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        comment="SHA-256 derived 16-char hex ID uniquely identifying this chunk.",
    )

    # ---------------------------------------------------------------------------
    # Foreign Key — links to parent document
    # ---------------------------------------------------------------------------
    document_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="References the parent document. Cascades deletes to clean up chunks.",
    )

    # ---------------------------------------------------------------------------
    # Content
    # ---------------------------------------------------------------------------
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Contextualized chunk text that was passed to the embedding model.",
    )

    # ---------------------------------------------------------------------------
    # Vector Embedding
    # ---------------------------------------------------------------------------
    embedding: Mapped[list[float]] = mapped_column(
        Vector(1024),  # 1024 dimensions — matches BAAI/bge-m3 output size
        nullable=False,
        comment="1024-dim L2-normalized embedding vector from BAAI/bge-m3.",
    )

    # ---------------------------------------------------------------------------
    # Full-Text Search — generated column (PostgreSQL computes this automatically)
    # ---------------------------------------------------------------------------
    tsvector_content_english: Mapped[Any] = mapped_column(
        TSVECTOR,
        # GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        # PostgreSQL auto-populates this column — we never write to it manually.
        # The Computed() tells SQLAlchemy this is a server-side computed column.
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=False,
        comment=(
            "PostgreSQL tsvector auto-generated from `content` using the English "
            "full-text dictionary. Used for sparse keyword search and Kannada "
            "ILIKE fallback in hybrid_retrieve()."
        ),
    )

    # ---------------------------------------------------------------------------
    # Metadata bucket — structured JSON
    # ---------------------------------------------------------------------------
    metadata_bucket: Mapped[dict[str, Any]] = mapped_column(
        "metadata",  # actual DB column name is 'metadata'
        JSONB,
        default=dict,
        nullable=False,
        comment=(
            "Structured JSON with three nested keys: "
            "'system' (chunk_id, chunk_number, source_file, page_number), "
            "'document' (document_title, folder_name), "
            "'user' (custom caller-supplied tagging dict)."
        ),
    )

    # ---------------------------------------------------------------------------
    # Relationship back to parent Document
    # ---------------------------------------------------------------------------
    # Uses a forward-reference string to match the back_populates in Document.chunks.
    document: Mapped["Document"] = relationship(  # noqa: F821
        "Document",
        back_populates="chunks",
    )
