"""
Adaptive query router — classifies incoming queries as vector DB or general knowledge.

The router is the first step in the agentic RAG pipeline. It decides whether
to search the local document database or answer from the LLM's general knowledge.

Routing Logic:
    vector_db       → Query is about Vrutti FPO program data:
                      FPC performance, farmer meetings, surveys, transcripts,
                      Kannada content, budgets, field reports, Karnataka districts.

    general_knowledge → Query is a general question, translation request,
                        casual greeting, or coding task with no FPO relevance.

Two-Layer Design:
    1. LLM Classification (Primary):
       Sends the query to Gemini 2.5 Flash with a structured JSON prompt.
       The LLM understands nuanced domain intent better than keyword rules.

    2. Keyword + Kannada Regex Fallback (Backup):
       If the Gemini API key is missing or the call fails, a hardcoded
       keyword list and a Kannada Unicode range check are used as fallback.
       This ensures the pipeline keeps working even without an API key
       during local development or testing.
"""

import json
import logging
import re

from google import genai
from google.genai import types

from app.core.config import settings
from app.pipeline.prompts import ROUTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback keyword list for domain-specific routing without LLM
# ---------------------------------------------------------------------------
# Covers key FPO program terms, district names, crop types, and Kannada-related words.
# Extend this list as new FPO topics are added to the program scope.
FPO_KEYWORDS: list[str] = [
    # Program identifiers
    "farmer", "fpo", "fpc", "vrutti", "catalyst", "walmart", "livelihood",
    # Karnataka geography
    "kalaburagi", "chikkaballapur", "karnataka",
    # Program activities
    "survey", "transcript", "meeting", "recording", "budget", "member", "women",
    # Crops relevant to the FPO program districts
    "crop", "yield", "ragi", "tur", "red gram", "soybean", "cotton",
    # Document types
    "report", "minutes", "agenda", "enrollment",
]

# Kannada Unicode block: U+0C80 to U+0CFF
# Any query containing Kannada script characters is automatically routed to vector_db
# since it almost certainly refers to local FPO program content.
KANNADA_PATTERN = re.compile(r"[\u0c80-\u0cff]")


async def route_query(query: str) -> str:
    """Classify the user query as 'vector_db' or 'general_knowledge'.

    This is the first step executed in the agentic RAG pipeline (agent.py).
    The route determines whether the system retrieves from the local document
    store or generates a response directly from the LLM's pretrained knowledge.

    Routing Decision Flow:
        1. Check for Kannada script → vector_db (immediate short-circuit).
        2. Check keyword list → preliminary fallback_route.
        3. Call Gemini 2.5 Flash for LLM-based classification.
        4. Validate LLM response and return route.
        5. On any failure → use fallback_route from steps 1–2.

    Args:
        query: The raw user query string (English, Kannada, or mixed).

    Returns:
        "vector_db"        — retrieve from local indexed FPO documents.
        "general_knowledge" — answer using LLM general knowledge directly.

    Example:
        route = await route_query("What were the issues in Kalaburagi FPC?")
        # → "vector_db"
        route = await route_query("What is photosynthesis?")
        # → "general_knowledge"
    """
    logger.info("Routing query: '%s'", query)

    # -------------------------------------------------------------------------
    # Layer 1: Kannada script detection (immediate override to vector_db)
    # -------------------------------------------------------------------------
    contains_kannada = bool(KANNADA_PATTERN.search(query))

    # -------------------------------------------------------------------------
    # Layer 2: Domain keyword match for fallback route
    # -------------------------------------------------------------------------
    is_fpo_topic = contains_kannada or any(
        keyword in query.lower() for keyword in FPO_KEYWORDS
    )
    fallback_route = "vector_db" if is_fpo_topic else "general_knowledge"

    # -------------------------------------------------------------------------
    # Layer 3: LLM-based classification (primary)
    # -------------------------------------------------------------------------
    has_key = (
        (settings.llm_provider == "nvidia" and settings.nvidia_api_key)
        or (settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here")
    )
    if not has_key:
        logger.warning(
            "LLM API Key not configured. Using keyword/Kannada fallback route: '%s'.",
            fallback_route,
        )
        return fallback_route

    try:
        from app.pipeline.llm import generate_llm_response

        response_text = await generate_llm_response(
            prompt=f"User Query: {query}",
            system_instruction=ROUTER_SYSTEM_PROMPT,
            json_mode=True,
        )

        data = json.loads(response_text)
        route = data.get("route", fallback_route)

        # Validate: only accept the two known route values
        if route not in ("vector_db", "general_knowledge"):
            logger.warning(
                "LLM returned unknown route '%s'. Using fallback: '%s'.", route, fallback_route
            )
            route = fallback_route

        logger.info("LLM routed query to: '%s'", route)
        return route

    except Exception as e:
        logger.error(
            "LLM routing failed: %s. Using fallback route: '%s'.", str(e), fallback_route
        )
        return fallback_route
