from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openrouter_api_key: str
    database_url: str
    default_llm_model: str = "openrouter/owl-alpha"
    embedding_model: str = "BAAI/bge-m3"
    research_llm_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    quant_engineer_llm_model: str = "cohere/north-mini-code:free"
    risk_llm_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    critic_llm_model: str = "qwen/qwen3-next-80b-a3b-instruct:free"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
