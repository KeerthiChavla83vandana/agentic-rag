"""
Document ingestion pipeline — parses source files into DoclingDocument objects.

This module is the entry point for the document processing pipeline.
It converts uploaded files (PDF, DOCX, PPTX) into Docling's structured
document representation, which downstream steps (embedding.py) then
chunk and vectorize.

Supported formats (via IBM Docling):
    - PDF  — with automatic text-layer detection (OCR-ready for future extension)
    - DOCX — Word documents
    - PPTX — PowerPoint presentations

Future OCR Extension:
    When OCR support is needed (e.g. scanned FPO field reports), extend this
    module to use DoclingPDFLoader from the AI Course reference project, which
    includes:
        - Automatic OCR detection (checks if PDF has a text layer)
        - ThreadedPdfPipelineOptions with AcceleratorOptions
        - Image extraction + vision LLM summarization (via NVIDIA NIM / Gemini)
    The HybridChunker downstream will handle OCR output natively without changes.
"""

import asyncio
import logging
from typing import Any

from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level Docling converter singleton
# ---------------------------------------------------------------------------
# DocumentConverter is expensive to initialize (loads internal models).
# A single instance is reused for all ingestion calls within the process lifetime.
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    """Return a cached DocumentConverter instance.

    Docling's DocumentConverter supports PDF, DOCX, PPTX out of the box,
    extracting text, tables, and headers for HybridChunker.
    """
    global _converter
    if _converter is None:
        logger.info("Initializing Docling DocumentConverter (one-time startup cost).")
        _converter = DocumentConverter()
    return _converter




def _run_docling_convert(file_path: str) -> Any:
    """Synchronous Docling conversion — runs in a thread pool executor.

    Docling's converter is CPU-bound and blocking. It should NOT be called
    directly from an async context as it would block the asyncio event loop
    and prevent other requests from being served concurrently.

    This function is always called via asyncio.get_event_loop().run_in_executor()
    from the public async `ingest_document()` function below.

    Args:
        file_path: Absolute or relative path to the source document.

    Returns:
        A raw ConversionResult.document (DoclingDocument) object.

    Raises:
        ValueError: If Docling fails to parse the file.
    """
    try:
        converter = _get_converter()
        result = converter.convert(file_path)
        logger.info("Docling successfully converted: %s", file_path)
        return result.document
    except Exception as e:
        logger.error("Docling conversion failed for %s: %s", file_path, str(e))
        raise ValueError(f"Docling ingestion failed for '{file_path}': {e}") from e


async def ingest_document(file_path: str) -> Any:
    """Parse a document file into a Docling DoclingDocument object (async-safe).

    This function offloads the blocking Docling conversion to the default
    thread pool executor so the asyncio event loop remains free to handle
    other requests during the (potentially multi-second) conversion.

    Supported formats:
        - .pdf  (with automatic text-layer detection; OCR extension planned)
        - .docx (Word documents)
        - .pptx (PowerPoint presentations)

    Args:
        file_path: Absolute or relative path to the source document file.

    Returns:
        A DoclingDocument object. This is passed directly to
        `embed_and_store_document()` in embedding.py for chunking.

    Raises:
        ValueError: If the file cannot be parsed by Docling.

    Example:
        docling_doc = await ingest_document("/workspace/data/report.pdf")
        chunks_count = await embed_and_store_document(db, "doc_001", ..., docling_doc)
    """
    logger.info("Starting async document ingestion for: %s", file_path)

    # run_in_executor dispatches the blocking _run_docling_convert() to a
    # thread pool so the event loop is not blocked during conversion.
    # None → uses the default ThreadPoolExecutor managed by asyncio.
    loop = asyncio.get_event_loop()
    docling_document = await loop.run_in_executor(None, _run_docling_convert, file_path)

    logger.info("Ingestion complete for: %s", file_path)
    return docling_document
