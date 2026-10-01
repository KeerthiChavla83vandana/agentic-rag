"""
Agentic RAG Orchestrator — full end-to-end Corrective RAG (CRAG) pipeline.

This module implements the main intelligence loop for the WMF FPO platform.
It is called by the /wmf_qa/query API endpoint and coordinates all pipeline steps.

Full Pipeline (execute_agentic_rag):
    ┌──────────────────────────────────────────────────────────┐
    │  User Query (+ Persona)                                  │
    └──────────────────────┬───────────────────────────────────┘
                           │
                    Step 1: Adaptive Router
                    route_query() → "vector_db" | "general_knowledge"
                           │
               ┌───────────┴──────────────┐
               │ general_knowledge        │ vector_db
               │                         │
               ▼                         ▼
    generate_direct_response()    Step 2: Hybrid Retrieval
    (LLM answers from pretrained   hybrid_retrieve() → top-5 chunks (RRF fused)
     knowledge, tailored to persona)      │
               │                         │
               │                  Step 3: Corrective Grading (PARALLEL)
               │                  asyncio.gather(grade_chunk() × N chunks)
               │                         │
               │                  Are ≥ 2 chunks relevant?
               │                         │
               │              Yes ──────┐│ No
               │                       ││
               │                       ││   Step 3b: Query Reformulation
               │                       ││   rewrite_query() → new query
               │                       ││   hybrid_retrieve() → re-retrieved chunks
               │                       ││   grade_chunk() × N (second pass)
               │                       ││
               │                       ▼▼
               │                  Step 4: Context Formatting
               │                  Format relevant chunks with [chunk_id] citation tags
               │                         │
               │                         ▼
               │                  Step 5: Citation-Aware Synthesis
               │                  Gemini generates answer citing [chunk_id] tags
               │                         │
               │                         ▼
               │                  Step 6: Citation Verification
               │                  Strip hallucinated [chunk_id] tags from response
               │                         │
               └──────────────────────── ▼
                                  Return structured result dict

Key Design Choices:
    - Chunk grading is done in PARALLEL (asyncio.gather) — reduces grading
      latency from N×LLM_latency to ~1×LLM_latency for N chunks.
    - Citation verification prevents hallucinated source references from
      appearing in the final user response.
    - Persona system prompt is injected at synthesis time, not at routing time,
      so the same retrieved context is tailored differently per audience.
    - All LLM calls use Gemini 2.5 Flash for speed (low latency) and cost.
"""

import asyncio
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.pipeline.grader import grade_chunk, rewrite_query
from app.pipeline.prompts import CITED_SYNTHESIS_BASE_PROMPT, PERSONA_PROMPTS
from app.pipeline.retrieval import hybrid_retrieve
from app.pipeline.router import route_query

logger = logging.getLogger(__name__)


async def generate_direct_response(query: str, persona: str) -> str:
    """Answer a general knowledge query directly from LLM without document context.

    Used when the router classifies the query as 'general_knowledge' — no
    database lookup is needed. The response is tailored to the selected persona.

    Args:
        query:   The user's natural language question.
        persona: Target role (e.g. "Program Lead", "Field Enumerator").

    Returns:
        LLM-generated response string. Returns a safe fallback message
        if the API key is missing or the LLM call fails.
    """
    persona_instruction = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["Untailored"])

    system_instruction = f"""
You are the intelligence agent of the Vrutti FPO Intelligence Platform.
Answer the User Query directly. You do not need database context for this query.
Keep the response helpful, clear, and tailored to the requested persona.

{persona_instruction}
"""
    fallback_response = "I am sorry, I am currently unable to connect to the LLM backend."

    has_key = (
        (settings.llm_provider == "nvidia" and settings.nvidia_api_key)
        or (settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here")
    )
    if not has_key:
        return "LLM API Key not configured. Please supply a valid key to run LLM synthesis."

    try:
        from app.pipeline.llm import generate_llm_response

        return await generate_llm_response(
            prompt=query,
            system_instruction=system_instruction,
            json_mode=False,
        )

    except Exception as e:
        logger.error("Direct generation failed: %s", str(e))
        return fallback_response


def verify_response_citations(
    response: str,
    available_chunk_ids: list[str],
) -> tuple[str, list[str]]:
    """Inspect the LLM response and strip any hallucinated citation tags.

    The synthesis step instructs the LLM to cite chunk IDs in the format [chunk_id].
    Occasionally the LLM hallucinates chunk IDs that were not in the provided context.
    This function:
        1. Extracts all [tag] patterns from the response.
        2. Checks each against the list of chunk IDs actually retrieved.
        3. Removes hallucinated tags from the response text.
        4. Returns a clean response and the list of verified real citations.

    Args:
        response:             Raw LLM-generated response text with citation tags.
        available_chunk_ids:  List of chunk IDs that were actually in the context.

    Returns:
        Tuple of (cleaned_response: str, verified_citations: list[str]).
    """
    # Match patterns like [abc123def456] — alphanumeric + underscore + hyphen
    citation_pattern = re.compile(r"\[([a-zA-Z0-9_\-]+)\]")
    found_tags = citation_pattern.findall(response)

    verified_citations: list[str] = []
    cleaned_response = response

    for tag in found_tags:
        # Accept a citation if the tag is a substring of any real chunk ID
        # (partial prefix match handles truncated IDs in LLM output)
        matched = any(tag in cid or cid in tag for cid in available_chunk_ids)

        if matched:
            # Record the full chunk ID that matched this citation tag
            for cid in available_chunk_ids:
                if tag in cid or cid in tag:
                    if cid not in verified_citations:
                        verified_citations.append(cid)
                    break
        else:
            # Remove the hallucinated tag from the response to keep it clean
            logger.warning("Removing hallucinated citation tag: [%s]", tag)
            cleaned_response = cleaned_response.replace(f"[{tag}]", "")

    # Collapse any double spaces caused by tag removal
    cleaned_response = re.sub(r"\s+", " ", cleaned_response).strip()
    return cleaned_response, verified_citations


async def execute_agentic_rag(
    db: AsyncSession,
    query: str,
    persona: str = "Untailored",
) -> dict[str, Any]:
    """Execute the full Corrective Agentic RAG (CRAG) pipeline.

    This is the main entry point called from the /wmf_qa/query FastAPI route.
    It orchestrates all pipeline steps: routing → retrieval → grading →
    optional reformulation → synthesis → citation verification.

    Args:
        db:      Active AsyncSession injected by FastAPI's get_db() dependency.
        query:   Raw user query string (English, Kannada, or mixed).
        persona: Target user role that shapes the synthesis response style.
                 One of: "Program Lead", "Junior Associate",
                         "Field Enumerator", "Untailored".

    Returns:
        A structured dict with the following keys:
            route:                  "vector_db" | "general_knowledge"
            original_query:         The user's original query text.
            search_query:           The query used for final retrieval (may be reformulated).
            corrective_action_taken: True if query reformulation was triggered.
            retrieved_chunks:       All retrieved chunks with their relevance grades.
            relevant_chunks:        Only the chunks graded as relevant.
            response:               Final synthesized answer text.
            citations_verified:     List of verified chunk IDs cited in the response.
    """
    t_start = time.perf_counter()
    logger.info("================================================================================")
    logger.info("🚀 AGENTIC RAG REQUEST EXECUTED")
    logger.info("--------------------------------------------------------------------------------")
    logger.info("📥 Query: '%s'", query)
    logger.info("👤 Persona: '%s'", persona)

    # =========================================================================
    # Step 1: Adaptive Query Routing
    # =========================================================================
    t_route_start = time.perf_counter()
    route = await route_query(query)
    t_route = time.perf_counter() - t_route_start
    logger.info("🛣️ Step 1: Route -> '%s' (took %.2fs)", route, t_route)

    if route == "general_knowledge":
        t_synth_start = time.perf_counter()
        response_text = await generate_direct_response(query, persona)
        t_total = time.perf_counter() - t_start
        logger.info("✍️ Direct Synthesis Completed (took %.2fs)", time.perf_counter() - t_synth_start)
        logger.info("⏱️ Total Pipeline Latency: %.2fs", t_total)
        logger.info("================================================================================")
        return {
            "route": route,
            "original_query": query,
            "search_query": query,
            "corrective_action_taken": False,
            "retrieved_chunks": [],
            "relevant_chunks": [],
            "response": response_text,
            "citations_verified": [],
            "latency": {"total_s": round(t_total, 2), "route_s": round(t_route, 2)},
        }

    # =========================================================================
    # Step 2: Hybrid Retrieval (Round 1)
    # =========================================================================
    t_ret_start = time.perf_counter()
    retrieved = await hybrid_retrieve(db, query, limit=5)
    t_ret = time.perf_counter() - t_ret_start
    logger.info("🔎 Step 2: Hybrid Retrieval -> %d chunks retrieved (took %.2fs)", len(retrieved), t_ret)

    # =========================================================================
    # Step 3: Corrective Grading — grade all chunks IN PARALLEL
    # =========================================================================
    t_grade_start = time.perf_counter()
    corrective_action_taken = False
    search_query = query

    grade_results = await asyncio.gather(
        *[grade_chunk(query, chunk["content"]) for chunk in retrieved],
        return_exceptions=False,
    )

    graded_chunks: list[dict[str, Any]] = []
    relevant_chunks: list[dict[str, Any]] = []

    for idx, (chunk, grade) in enumerate(zip(retrieved, grade_results), start=1):
        chunk_with_grade = dict(chunk)
        chunk_with_grade["grade"] = grade
        graded_chunks.append(chunk_with_grade)
        if grade == "YES":
            relevant_chunks.append(chunk)
        logger.info(
            "   └─ Chunk #%d (ID: %s) | RRF Score: %.6f | Relevant: %s",
            idx,
            chunk["chunk_id"],
            chunk.get("rrf_score", 0.0),
            grade,
        )

    t_grade = time.perf_counter() - t_grade_start
    logger.info(
        "📊 Step 3: Relevance Grading -> %d/%d chunks relevant (took %.2fs)",
        len(relevant_chunks),
        len(retrieved),
        t_grade,
    )

    # =========================================================================
    # Step 3b: Corrective Loop — triggered if relevance is low
    # =========================================================================
    if len(relevant_chunks) < 2 and len(retrieved) > 0:
        logger.info(
            "⚠️ Low relevance threshold (%d/5). Triggering query reformulation.",
            len(relevant_chunks),
        )
        corrective_action_taken = True
        search_query = await rewrite_query(query)

        retrieved_retry = await hybrid_retrieve(db, search_query, limit=5)
        retry_grades = await asyncio.gather(
            *[grade_chunk(search_query, chunk["content"]) for chunk in retrieved_retry],
            return_exceptions=False,
        )

        graded_chunks = []
        relevant_chunks = []

        for chunk, grade in zip(retrieved_retry, retry_grades):
            chunk_with_grade = dict(chunk)
            chunk_with_grade["grade"] = grade
            graded_chunks.append(chunk_with_grade)
            if grade == "YES":
                relevant_chunks.append(chunk)

    # =========================================================================
    # Step 4: Fallback if still no relevant chunks after corrective loop
    # =========================================================================
    if not relevant_chunks:
        t_total = time.perf_counter() - t_start
        logger.warning("⚠️ No relevant chunks found. Returning partial fallback.")
        fallback_intro = (
            "I could not find any specific records in the Vrutti FPO program documents "
            "related to your query. Here is a general response based on typical FPO practices:\n\n"
        )
        fallback_body = await generate_direct_response(query, persona)
        logger.info("⏱️ Total Pipeline Latency: %.2fs", t_total)
        logger.info("================================================================================")
        return {
            "route": route,
            "original_query": query,
            "search_query": search_query,
            "corrective_action_taken": corrective_action_taken,
            "retrieved_chunks": graded_chunks,
            "relevant_chunks": [],
            "response": fallback_intro + fallback_body,
            "citations_verified": [],
            "latency": {"total_s": round(t_total, 2)},
        }

    # =========================================================================
    # Step 5: Citation-Aware Synthesis
    # =========================================================================
    t_synth_start = time.perf_counter()
    context_parts: list[str] = []
    chunk_ids: list[str] = []

    for chunk in relevant_chunks:
        cid = chunk["chunk_id"]
        chunk_ids.append(cid)
        source_file = (
            chunk.get("metadata", {})
            .get("system", {})
            .get("source_file_name", "document")
        )
        context_parts.append(
            f"--- START CHUNK [{cid}] (Source: {source_file}) ---\n"
            f"{chunk['content']}\n"
            f"--- END CHUNK ---"
        )

    context_text = "\n\n".join(context_parts)
    persona_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["Untailored"])
    system_instruction = (
        CITED_SYNTHESIS_BASE_PROMPT.format(context_text=context_text)
        + "\n"
        + persona_prompt
    )

    has_key = bool(
        (settings.llm_provider == "nvidia" and settings.nvidia_api_key)
        or (settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here")
    )
    if not has_key:
        t_total = time.perf_counter() - t_start
        return {
            "route": route,
            "original_query": query,
            "search_query": search_query,
            "corrective_action_taken": corrective_action_taken,
            "retrieved_chunks": graded_chunks,
            "relevant_chunks": relevant_chunks,
            "response": "LLM API Key not configured.",
            "citations_verified": [],
            "latency": {"total_s": round(t_total, 2)},
        }

    try:
        from app.pipeline.llm import generate_llm_response

        raw_response = await generate_llm_response(
            prompt=f"User Query: {query}",
            system_instruction=system_instruction,
            json_mode=False,
        )

    except Exception as e:
        t_total = time.perf_counter() - t_start
        logger.error("LLM synthesis exception (%s): %s", type(e).__name__, str(e), exc_info=True)
        err_msg = f"{type(e).__name__}: {str(e) if str(e) else 'API Request Timed Out'}"
        return {
            "route": route,
            "original_query": query,
            "search_query": search_query,
            "corrective_action_taken": corrective_action_taken,
            "retrieved_chunks": graded_chunks,
            "relevant_chunks": relevant_chunks,
            "response": f"An error occurred during response synthesis: {err_msg}",
            "citations_verified": [],
            "latency": {"total_s": round(t_total, 2)},
        }

    # =========================================================================
    # Step 6: Citation Verification & Telemetry Summary
    # =========================================================================
    cleaned_response, verified_cites = verify_response_citations(raw_response, chunk_ids)
    t_synth = time.perf_counter() - t_synth_start
    t_total = time.perf_counter() - t_start

    logger.info(
        "✍️ Step 5: LLM Synthesis -> Completed (took %.2fs) | Citations verified: %d/%d",
        t_synth,
        len(verified_cites),
        len(chunk_ids),
    )
    logger.info("⏱️ Total Pipeline Latency: %.2fs", t_total)
    logger.info("================================================================================")

    return {
        "route": route,
        "original_query": query,
        "search_query": search_query,
        "corrective_action_taken": corrective_action_taken,
        "retrieved_chunks": graded_chunks,
        "relevant_chunks": relevant_chunks,
        "response": cleaned_response,
        "citations_verified": verified_cites,
        "latency": {
            "total_s": round(t_total, 2),
            "route_s": round(t_route, 2),
            "retrieval_s": round(t_ret, 2),
            "grading_s": round(t_grade, 2),
            "synthesis_s": round(t_synth, 2),
        },
    }
