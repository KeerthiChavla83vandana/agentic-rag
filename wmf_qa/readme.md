# WMF QA Agentic RAG Pipeline

This document explains the architecture and flow of the WMF QA (Question Answering) Intelligence Platform built for the Vrutti FPO Initiative.

## 1. Pipeline Overview

The system uses an **Agentic Corrective RAG (CRAG)** pipeline. Unlike a simple RAG (Retrieve-then-Generate) pipeline, this architecture introduces "agentic" decision-making: it routes queries intelligently, evaluates the retrieved context for relevance, and has a self-correcting loop to rewrite queries if initial searches fail.

### Step-by-Step Flow

1. **Adaptive Routing**:
   - When a query comes in, the **Router** decides if it requires internal knowledge (`vector_db`) or if it is a general question (`general_knowledge`).
   - It checks for Kannada script and specific domain keywords first. If found, it routes to `vector_db`.
   - It then uses an LLM to classify the intent. 
   - If routed to `general_knowledge`, the pipeline skips retrieval and directly answers.

2. **Hybrid Retrieval**:
   - The query is embedded (using the `BAAI/bge-m3` model) to perform **Dense Vector Search** (Cosine Similarity).
   - A Full-Text Search (FTS) is executed concurrently using PostgreSQL `tsvector` (BM25-like keyword search).
   - The results from dense and sparse searches are combined using **Reciprocal Rank Fusion (RRF)** to extract the top-5 most relevant chunks.

3. **Corrective Grading (Parallel)**:
   - Each of the top-5 chunks is independently evaluated by an LLM Grader to determine if it is truly relevant ("YES" or "NO") to the query. 
   - This grading happens in parallel to minimize latency.

4. **Query Reformulation (Self-Correction Loop)**:
   - If fewer than 2 chunks are graded as relevant ("YES"), the system assumes the search failed.
   - The LLM rewrites the user's query into a more search-optimized form (expanding keywords, handling Kannada/English translations).
   - It then repeats the retrieval and grading process with the new reformulated query.

5. **Citation-Aware Synthesis**:
   - The strictly relevant chunks are formatted into a context block.
   - The LLM generates an answer, explicitly citing the source chunk IDs (e.g., `[chunk_id]`).
   - The prompt dynamically injects a **Persona** (Program Lead, Junior Associate, Field Enumerator, or Untailored) to tailor the tone, language, and detail level of the response.

6. **Citation Verification**:
   - The final response is parsed. The system strips out any hallucinated citations (chunk IDs that the LLM invented) to ensure high factual reliability.

## 2. Models Used

- **LLM / Generation**: 
  - `gemini-2.5-flash` (Google GenAI) or `meta/llama-3.2-11b-vision-instruct` (NVIDIA NIM). The backend seamlessly swaps between them depending on the active API keys.
- **Embeddings**: 
  - `BAAI/bge-m3` (a powerful multilingual model generating 1024-dimensional vectors).

## 3. Database & Retrieval Mechanics

- **Database**: PostgreSQL 14 running the `pgvector` extension.
- **Indexing**: 
  - **HNSW Index** for extremely fast approximate nearest-neighbor vector search.
  - **GIN Index** on a `tsvector` column for robust Full-Text Search.
- **Document Ingestion**: Uses **Docling** for extracting content from PDFs, DOCX, and PPTX files before chunking them into semantic segments.
