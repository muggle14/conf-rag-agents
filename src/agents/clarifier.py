# clarifier.py
"""
Simplified clarifier for generating concise clarification questions.
Works with confidence scoring to ask ONE clarifying question <= 20 words.
"""

import json
import os
from typing import List

from openai import AzureOpenAI

# System prompt for the clarifier
SYSTEM = 'You are ClarifierAgent. Ask exactly ONE clarifying question in <= 20 words. No extra text. Return JSON: {"question":"..."}'

USER_TMPL = """User query:
{query}

Top result titles:
{titles}

Output strict JSON with a single field "question".
"""


def ask_clarifying_question(query: str, top_titles: List[str]) -> str:
    """
    Generate a single clarifying question based on ambiguous query and top search results.

    Args:
        query: User's original query
        top_titles: List of top result titles for context

    Returns:
        A clarifying question (max 20 words)
    """
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=os.environ["AOAI_ENDPOINT"],
        api_key=os.environ["AOAI_KEY"],
        api_version="2025-01-01",
    )

    # Format the prompt
    prompt = USER_TMPL.format(
        query=query, titles="\n".join([f"- {t}" for t in top_titles[:4]])
    )

    try:
        # Call Azure OpenAI with JSON response format
        response = client.chat.completions.create(
            model=os.getenv("AOAI_CHAT_DEPLOY", "gpt-4o"),
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        # Parse JSON response
        result = json.loads(response.choices[0].message.content)
        question = result.get("question", "")

        # Fallback if no question in JSON
        if not question:
            # Try to extract from raw content
            question = response.choices[0].message.content.strip()

    except Exception:
        # Hard fallback if API fails
        question = "Which specific area are you asking about?"

    # Safety limit to 120 characters
    return question[:120]
