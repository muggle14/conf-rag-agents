"""
Provider-agnostic LLM wrapper for JSON responses.
"""

import json
import os
from typing import Any, Dict

from ..config.env import LLM_MODEL, LLM_PROVIDER


def call_json(prompt: str, system: str = "", max_tokens: int = 1024) -> Dict[str, Any]:
    """
    Returns parsed JSON object produced by the model.
    Primary implementation uses Azure OpenAI with GPT-4o.

    Args:
        prompt: User prompt
        system: System prompt
        max_tokens: Maximum tokens for response

    Returns:
        Parsed JSON object from model response
    """
    provider = LLM_PROVIDER

    if provider == "azure_openai" or provider == "openai":
        # Azure OpenAI implementation with GPT-4o
        from openai import AzureOpenAI

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        # Use GPT-4o deployment
        deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

        if not endpoint or not api_key:
            raise ValueError("Azure OpenAI credentials not configured")

        client = AzureOpenAI(
            api_key=api_key, azure_endpoint=endpoint, api_version="2024-02-01"
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
            response_format={"type": "json_object"},  # Request JSON response
        )

        text = response.choices[0].message.content

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try lenient parse
            start = text.find("{")
            end = text.rfind("}")

            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return {"raw": text, "parse_error": "Failed to parse JSON"}

            return {"raw": text, "parse_error": "No JSON found"}

    elif provider == "openai":
        # Standard OpenAI implementation (non-Azure)
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        client = OpenAI(api_key=api_key)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=LLM_MODEL or "gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text, "parse_error": "Failed to parse JSON"}

    else:
        raise NotImplementedError(
            f"LLM_PROVIDER={provider} not implemented. Use 'azure_openai' or 'openai'"
        )
