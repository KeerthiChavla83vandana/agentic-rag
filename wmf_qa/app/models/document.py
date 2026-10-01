"""
SQLAlchemy ORM model for the `documents` table.

Each row in this table represents a single ingested source document
(PDF, DOCX, PPTX). It tracks the embedding lifecycle status and stores
optional tagging metadata for folder-based grouping.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

# TYPE_CHECKING guard avoids a circular import at runtime.
# DocumentChunk imports Document (via ForeignKey), and Document now
# imports DocumentChunk only for type hints — this pattern breaks the cycle.
if TYPE_CHECKING:
    from app.models.document_chunk import DocumentChunk


class Document(Base):
    """SQLAlchemy model representing an uploaded and indexed source document.

    Attributes:
        document_id:       Caller-defined unique string identifier (e.g. filename slug).
        source_file:       Original filename of the uploaded file.
        file_path:         Storage path where the file was saved during ingestion.
        folder_name:       Optional high-level folder tag for grouping documents.
        subfolder_name:    Optional secondary grouping tag.
        uploaded_at:       UTC timestamp of when the document was first created.
        tagging_metadata:  Arbitrary JSON dictionary for custom domain tagging.
        embedding_status:  Lifecycle state: "pending" | "completed" | "failed".
        embedded_at:       UTC timestamp of when embedding completed successfully.
        chunks:            One-to-many relationship to DocumentChunk records.
    """

    __tablename__ = "documents"

    # ---------------------------------------------------------------------------
    # Primary Key
    # ---------------------------------------------------------------------------
    document_id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        comment="Caller-supplied unique document identifier (e.g. filename slug).",
    )

    # ---------------------------------------------------------------------------
    # File Identity
    # ---------------------------------------------------------------------------
    source_file: Mapped[str] = mapped_column(
        String,
        nullable=False,
        comment="Original filename of the uploaded source file.",
    )

    file_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
        comment="Absolute storage path where the file was saved on disk.",
    )

    # ---------------------------------------------------------------------------
    # Grouping Tags (optional)
    # ---------------------------------------------------------------------------
    folder_name: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        comment="High-level folder group label (e.g. 'Vrutti_FPO_RAG').",
    )

    subfolder_name: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        comment="Secondary grouping label within folder_name.",
    )

    # ---------------------------------------------------------------------------
    # Timestamps
    # ---------------------------------------------------------------------------
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="UTC timestamp when the document record was first created.",
    )

    embedded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="UTC timestamp of when embedding completed successfully. NULL if not yet done.",
    )

    # ---------------------------------------------------------------------------
    # Metadata & Status
    # ---------------------------------------------------------------------------
    tagging_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Arbitrary JSON metadata dict for custom domain-level tagging.",
    )

    embedding_status: Mapped[str] = mapped_column(
        String,
        default="pending",
        nullable=False,
        comment="Embedding lifecycle status: 'pending', 'completed', or 'failed'.",
    )

    # ---------------------------------------------------------------------------
    # Relationships
    # ---------------------------------------------------------------------------
    # Forward reference string avoids circular import at runtime.
    # At runtime Python resolves "DocumentChunk" using the mapper registry.
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_id",
        lazy="selectin",  # use selectin loading — avoids N+1 queries in async context
    )
