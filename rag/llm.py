import httpx

from config import get_settings


SYSTEM_PROMPT = """You are a careful fintech research assistant.
Answer only from the supplied RBNZ context.
If the context does not contain enough evidence, say that clearly.
Cite document titles and page numbers when useful."""


def build_prompt(query: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Answer:"""


def answer_with_llm(query: str, context: str) -> str:
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider in {"", "none", "off"}:
        return (
            "LLM answering is disabled. Retrieved context is available in the response; "
            "set LLM_PROVIDER to gemini, deepseek, or openrouter to generate an answer."
        )

    if provider == "gemini":
        return answer_with_gemini(query, context)
    if provider == "deepseek":
        return answer_with_openai_compatible(
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            base_url="https://api.deepseek.com/chat/completions",
            query=query,
            context=context,
        )
    if provider == "openrouter":
        return answer_with_openai_compatible(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            base_url="https://openrouter.ai/api/v1/chat/completions",
            query=query,
            context=context,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


def answer_with_gemini(query: str, context: str) -> str:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": build_prompt(query=query, context=context),
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 800,
        },
    }

    response = httpx.post(
        url,
        params={"key": settings.gemini_api_key},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def answer_with_openai_compatible(
    *,
    api_key: str | None,
    model: str,
    base_url: str,
    query: str,
    context: str,
) -> str:
    if not api_key:
        raise ValueError("API key is required for the selected LLM provider")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }
    response = httpx.post(
        base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
