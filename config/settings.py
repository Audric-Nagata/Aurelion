from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openrouter_api_key: str 
    database_url: str 
    default_llm_model: str = "openrouter/owl-alpha"
    embedding_model: str = "BAAI/bge-m3"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
