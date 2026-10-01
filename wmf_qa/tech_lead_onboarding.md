# WMF Intelligence Platform - Tech Lead Onboarding & Context

This document extracts the core context, objectives, and architectural flow from the WMF Intelligence Platform Technical Proposal. It is designed to give the new Technical Lead the necessary context to draft the Technical Design Document (TDD).

## 1. Project Context: Who & What
*   **Client / End-Users:** Nitish & Soumi from Catalyst Group (acting on behalf of **Vrutti**, supported by the Walmart Foundation).
*   **Our Team:** Accelyzei Technologies.
*   **The Project:** The "WMF Intelligence Platform." 
*   **The Goal:** Build a ground-up, production-grade **Agentic RAG** platform on AWS. The platform serves as a funder-facing evidence tool where users can ask natural language questions about FPO (Farmer Producer Organization) data, surveys, and field recordings, and receive highly reliable, properly cited answers.

## 2. Core Engineering Philosophy
1.  **Retrieval Quality > Agent Sprawl:** We are avoiding unnecessary multi-agent complexity. Instead, we are focusing heavily on advanced retrieval (hybrid search + cross-encoder reranking + contextual enrichment) to ensure the LLM gets the best possible context.
2.  **Deliberate Autonomy:** "Agentic" behavior is isolated to where it matters most:
    *   **Adaptive Routing:** Deciding if a query needs DB data, document data, or general knowledge.
    *   **Corrective Verifier Loop:** An agent that rejects ungrounded drafts and forces a re-retrieval.
3.  **Privacy by Design:** Strict PII redaction (using CPU-based tools like Presidio/AWS Comprehend) before any sensitive data hits reasoning LLMs.
4.  **Decoupled Architecture:** Strict REST API boundaries so the frontend (currently Streamlit) can be swapped to React without backend changes.

## 3. Technical Architecture & Tech Stack
The platform is deployed entirely on **AWS (Region-Pinned)** and is broken into four distinct layers:

### A. View Layer (Frontend)
*   **Phase 1:** Streamlit (hosted on ECS) for rapid prototyping and validation.
*   **Phase 2:** Full React/TypeScript application hosted via S3 + CloudFront.

### B. Backend Layer
*   **Stack:** Python, FastAPI, Pydantic, Server-Sent Events (SSE) for token streaming.
*   **Auth:** Google OAuth or Amazon Cognito.
*   **Compute:** Dockerized and running on AWS ECS Fargate.
*   **Role:** Acts as a thin, async API router that handles auth, RBAC, file storage orchestration, and triggers the Agentic RAG pipeline.

### C. Data & Storage Layer
*   **Primary DB:** Amazon RDS running **PostgreSQL 14+**.
*   **Search:** We are NOT using a separate vector DB. Both **Dense Vector Search** (via `pgvector`) and **Sparse Keyword Search** (via Postgres native Full-Text Search) live in the same PostgreSQL instance.
*   **File Storage:** Amazon S3 (tiered storage for raw audio/PDFs).
*   **Security:** Multi-tenant isolation and RBAC (Region/FPC-level access) is handled strictly via **PostgreSQL Row-Level Security (RLS)**.

### D. Agentic / RAG Layer (The Core Engine)
This is the intelligence pipeline:
1.  **Router:** (LLM: Gemini Flash-Lite/Claude Haiku) Determines the query intent.
2.  **Retrieval:** Executes a Hybrid Search (pgvector + FTS) and merges results using Reciprocal Rank Fusion (RRF).
3.  **Reranker:** (Cross-encoder like BGE/Cohere) Re-scores the retrieved chunks.
4.  **Synthesis:** (LLM: Claude Sonnet / Gemini 3.5 Flash) Drafts the response with citations.
5.  **Verifier (Corrective Loop):** (LLM: GPT-5.5 / Gemini 3.1 Pro) Checks if the draft is hallucinated. If yes, it rewrites the search query and loops back to retrieval.
6.  **Proactive Analysis (Async):** Celery + SQS + EventBridge run standing cron queries in the background (e.g., "Alert me if any FPC membership drops").

## 4. Ingestion & Voice Pipeline (Data Flow)
The ingestion pipeline handles raw data (Docs, Spreadsheets, Audio):
1.  **Audio Ingestion:** Field recordings go through an Indic ASR engine (e.g., Sarvam / IndicWhisper).
2.  **Translation:** Translated to English using Gemini on Vertex AI (region-pinned, zero-retention).
3.  **PII Redaction:** CPU-based NER (Regex + Presidio / AWS Comprehend / DLP) scrubs names, IDs, phones. *Crucial: We do this on CPU, no dedicated GPUs required.*
4.  **Document Parsing:** **Docling** is used for layout-aware PDF/Doc parsing.
5.  **Chunking:** Parent-Child semantic chunking strategy. Each chunk is contextually enriched (a short document summary is prepended to the chunk).
6.  **Embedding:** BAAI/bge-m3 or multilingual-e5 generates 768/1024-dim vectors.
7.  **Storage:** Written to PostgreSQL.

## 5. Observability & Delivery Phases
*   **Observability:** Amazon CloudWatch (Logs/Metrics) + **LangFuse** (Agent tracing and cost tracking).
*   **Delivery Phases:**
    *   **Phase 0:** Discovery & ASR Benchmarking.
    *   **Phase 1:** Prototype (Streamlit + Core RAG pipeline).
    *   **Phase 2:** Production (React, Full AWS Infra, RBAC).
    *   **Phase 3:** Hardening & UAT.
