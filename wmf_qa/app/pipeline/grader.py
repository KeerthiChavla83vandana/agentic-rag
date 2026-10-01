"""
LLM-based chunk relevance grader and query reformulator.

These two functions are core components of the Corrective RAG (CRAG) loop
implemented in agent.py:

    Retrieved chunks → [grade_chunk()] → Relevant / Irrelevant
                                               │
                                      If < 2 relevant:
                                               │
                                               ▼
                                       [rewrite_query()]
                                               │
                                               ▼
                                       Re-retrieve with new query

Design Decisions:
    - Both functions use Gemini 2.5 Flash for speed and cost efficiency.
    - Structured JSON responses (`response_mime_type="application/json"`) make
      parsing deterministic and eliminate the need for regex extraction.
    - A keyword-based fallback is provided for both functions so the pipeline
      degrades gracefully when the GEMINI_API_KEY is missing or the LLM call fails.
    - Both functions are async (using the google-genai async client) so they can
      be called in parallel via asyncio.gather() in agent.py, reducing the 5-chunk
      sequential grading latency from ~5 LLM calls to ~1 parallel round trip.
"""

import json
import logging

from google import genai
from google.genai import types

from app.core.config import settings
from app.pipeline.prompts import GRADER_SYSTEM_PROMPT, REWRITER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _has_llm_key() -> bool:
    return bool(
        (settings.llm_provider == "nvidia" and settings.nvidia_api_key)
        or (settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here")
    )


async def grade_chunk(query: str, chunk_content: str) -> str:
    """Grade the relevance of a retrieved document chunk to the user query."""
    # Keyword-based fallback
    query_words = set(query.lower().split())
    chunk_words = set(chunk_content.lower().split())
    shared_words = query_words.intersection(chunk_words)

    if shared_words:
        fallback_grade = "YES"
    elif len(query_words) > 3:
        fallback_grade = "NO"
    else:
        fallback_grade = "YES"

    if not _has_llm_key():
        logger.warning("LLM API key not set. Using keyword fallback for grading.")
        return fallback_grade

    try:
        from app.pipeline.llm import generate_llm_response

        prompt = f"User Query: {query}\n\nDocument Chunk Content: {chunk_content}"
        response_text = await generate_llm_response(
            prompt=prompt,
            system_instruction=GRADER_SYSTEM_PROMPT,
            json_mode=True,
        )

        data = json.loads(response_text)
        grade = data.get("relevance", fallback_grade).upper().strip()

        if grade not in ("YES", "NO"):
            logger.warning("Unexpected grade value '%s' from LLM. Using fallback.", grade)
            return fallback_grade

        return grade

    except Exception as e:
        logger.error(
            "LLM grading failed: %s. Using keyword fallback grade: %s", str(e), fallback_grade
        )
        return fallback_grade


async def rewrite_query(query: str) -> str:
    """Reformulate a low-performing search query to improve retrieval recall."""
    logger.info("Reformulating query for improved recall. Original: '%s'", query)

    if not _has_llm_key():
        logger.warning("LLM API key not set. Skipping query reformulation.")
        return query

    try:
        from app.pipeline.llm import generate_llm_response

        response_text = await generate_llm_response(
            prompt=f"Original Query: {query}",
            system_instruction=REWRITER_SYSTEM_PROMPT,
            json_mode=True,
        )

        data = json.loads(response_text)
        new_query = data.get("reformulated_query", query)

        logger.info("Query reformulated: '%s' → '%s'", query, new_query)
        return new_query

    except Exception as e:
        logger.error("Query reformulation failed: %s. Using original query.", str(e))
        return query

