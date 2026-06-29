import os

# ponytail: os.getenv over pydantic BaseSettings — 4 fields don't need a framework

openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/aurelion")
default_llm_model = os.getenv("DEFAULT_LLM_MODEL", "openrouter/owl-alpha")
embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
research_llm_model = os.getenv("RESEARCH_LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
quant_engineer_llm_model = os.getenv("QUANT_ENGINEER_LLM_MODEL", "openai/gpt-oss-120b:free")
risk_llm_model = os.getenv("RISK_LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
critic_llm_model = os.getenv("CRITIC_LLM_MODEL", "qwen/qwen3-next-80b-a3b-instruct:free")

AGENT_MODELS = {
    "research":       research_llm_model,
    "quant_engineer": quant_engineer_llm_model,
    "risk":           risk_llm_model,
    "critic":         critic_llm_model,
}
