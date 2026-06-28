from .settings import settings

AGENT_MODELS = {
    "research":       settings.research_llm_model,
    "quant_engineer": settings.quant_engineer_llm_model,
    "risk":           settings.risk_llm_model,
    "critic":         settings.critic_llm_model,
}
