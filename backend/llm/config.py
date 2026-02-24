from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = os.getenv("GROQ_API_KEY", "")
    model: str = "llama-3.3-70b-versatile"
    timeout: float = 10.0
    max_tokens: int = 1024
    enabled: bool = True


DEFAULT_LLM_CONFIG = LLMConfig()
