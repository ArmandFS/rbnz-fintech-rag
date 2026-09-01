import time

import httpx

from config import get_settings


# Demands [N] citations tied to retrieval.py's numbered context blocks; earlier
# wording only asked for citations "when useful" and answers cited inconsistently.
SYSTEM_PROMPT = """You are a careful fintech research assistant.
Answer only from the supplied RBNZ context.
The context is a numbered list of chunks, each starting with a bracket number like [1], [2].
Cite sources inline using that bracket number directly after the claim it supports, e.g. "Inflation is expected to ease [2]."
Cite every non-trivial claim. Do not invent citations for chunk numbers that are not in the context.
If the context does not contain enough evidence to answer, say that clearly instead of guessing."""


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

    response = post_with_retries(
        url,
        headers={"x-goog-api-key": settings.gemini_api_key},
        json=payload,
        timeout=60,
        error_message="Gemini generation request failed",
    )
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
    response = post_with_retries(
        base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=60,
        error_message="OpenAI-compatible generation request failed",
    )
    data = response.json()
    return data["choices"][0]["message"]["content"]


def post_with_retries(
    url: str,
    *,
    headers: dict[str, str],
    json: dict,
    timeout: int,
    error_message: str,
) -> httpx.Response:
    settings = get_settings()
    retryable_statuses = {429, 500, 502, 503, 504}
    last_response: httpx.Response | None = None

    for attempt in range(settings.llm_max_retries + 1):
        try:
            response = httpx.post(url, headers=headers, json=json, timeout=timeout)
        except httpx.HTTPError as exc:
            if attempt >= settings.llm_max_retries:
                raise RuntimeError(f"{error_message}: {exc}") from exc
            sleep_before_retry(attempt)
            continue

        if response.status_code not in retryable_statuses:
            raise_clean_http_error(response, error_message)
            return response

        last_response = response
        if attempt >= settings.llm_max_retries:
            break
        sleep_before_retry(attempt)

    if last_response is not None:
        raise_clean_http_error(last_response, error_message)

    raise RuntimeError(f"{error_message}: request failed before receiving a response")


def sleep_before_retry(attempt: int) -> None:
    settings = get_settings()
    delay = settings.llm_retry_base_delay * (2**attempt)
    time.sleep(delay)


def raise_clean_http_error(response: httpx.Response, message: str) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text[:500]
        raise RuntimeError(
            f"{message}: HTTP {response.status_code}. Response body: {detail}"
        ) from exc
