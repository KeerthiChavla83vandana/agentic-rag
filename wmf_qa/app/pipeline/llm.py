"""
Unified LLM Client supporting both NVIDIA NIM API and Google Gemini API.

When settings.llm_provider == "nvidia", calls NVIDIA NIM API (OpenAI chat completions compatible).
When settings.llm_provider == "gemini", calls Google Gemini API.
"""

import json
import logging

import httpx
from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)


async def generate_llm_response(
    prompt: str,
    system_instruction: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.2,
) -> str:
    """Generate a response from the configured LLM provider (NVIDIA NIM or Gemini).

    Args:
        prompt: User prompt content.
        system_instruction: Optional system instruction prompt.
        json_mode: If True, requests JSON output structure.
        temperature: Model sampling temperature (default 0.2).

    Returns:
        String text output from the model.
    """
    if settings.llm_provider == "nvidia" and settings.nvidia_api_key:
        return await _generate_nvidia(
            prompt, system_instruction, json_mode, temperature
        )
    elif settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here":
        return await _generate_gemini(
            prompt, system_instruction, json_mode, temperature
        )
    else:
        raise ValueError("No valid LLM API key configured (neither NVIDIA nor Gemini).")


async def _generate_nvidia(
    prompt: str,
    system_instruction: str | None,
    json_mode: bool,
    temperature: float,
) -> str:
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_instruction:
        sys_content = system_instruction
        if json_mode:
            sys_content += "\nCRITICAL: Respond ONLY with a valid JSON object. Do not include markdown code fences or conversational text."
        messages.append({"role": "system", "content": sys_content})
    elif json_mode:
        messages.append({"role": "system", "content": "Respond ONLY with a valid JSON object."})

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": settings.nvidia_model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(url, headers=headers, json=payload)
            if res.status_code != 200:
                logger.error(
                    "NVIDIA NIM API HTTP %d Error: %s", res.status_code, res.text
                )
                res.raise_for_status()
            data = res.json()
            content = data["choices"][0]["message"]["content"].strip()

            if json_mode:
                if content.startswith("```"):
                    lines = content.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    content = "\n".join(lines).strip()
            return content
    except Exception as exc:
        logger.error(
            "NVIDIA NIM API Exception (%s): %s", type(exc).__name__, str(exc), exc_info=True
        )
        raise


async def _generate_gemini(
    prompt: str,
    system_instruction: str | None,
    json_mode: bool,
    temperature: float,
) -> str:
    client = genai.Client(api_key=settings.gemini_api_key)
    config_args = {}
    if system_instruction:
        config_args["system_instruction"] = system_instruction
    if json_mode:
        config_args["response_mime_type"] = "application/json"

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(**config_args),
    )
    return response.text.strip()
