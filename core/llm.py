"""
Single place that builds the chat model used across the app.

Uses Google Gemini via langchain-google-genai. Configure in .env:
  GEMINI_API_KEY=...        (GOOGLE_API_KEY also works)
  LLM_MODEL=gemini-3-flash-preview  (optional)
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL_NAME = os.getenv("LLM_MODEL", "gemini-3-flash-preview")


def get_llm(temperature: float = 0) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Add GEMINI_API_KEY=... to the .env file "
            "in the project folder, then restart the app."
        )
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature,
        google_api_key=api_key,
        # v1 rejects system prompts ("Developer instruction is not enabled").
        api_version="v1beta",
        # Free-tier keys are rate limited (a few requests/minute); keep retrying.
        max_retries=6,
    )
