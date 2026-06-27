import httpx
from config import settings


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.default_llm_model
        self.base_url = "https://openrouter.ai/api/v1"
        self.http_client = httpx.Client(timeout=120.0)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        try:
            response = self.http_client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            msg = response.json()["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning") or ""
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"LLM API returned {e.response.status_code}: "
                f"{e.response.text[:500]}"
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(
                f"LLM API request failed: {e}"
            ) from e
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(
                f"Unexpected LLM API response format: {e}"
            ) from e


