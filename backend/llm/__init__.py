"""
LLM integration layer.

Responsibilities:
- Manage Groq API configuration and credentials.
- Build prompts from user preferences and candidate restaurants.
- Call Groq LLM to re-rank candidates and generate explanations.
- Graceful fallback when the LLM is unavailable or returns invalid output.
"""
