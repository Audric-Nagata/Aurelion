import httpx
from config import openrouter_api_key, default_llm_model


# ponytail: function over class — stateless, no lifetime to manage
def llm_chat(
    messages: list[dict],
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    api_key = openrouter_api_key
    model = model or default_llm_model
    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            raise RuntimeError(
                f"OpenRouter API error: {body['error'].get('message') or body['error']}"
            )
        msg = body["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning") or ""
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"LLM API returned {e.response.status_code}: "
            f"{e.response.text[:500]}"
        ) from e
    except httpx.RequestError as e:
        raise RuntimeError(f"LLM API request failed: {e}") from e
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"Unexpected LLM API response format: {e}") from e
