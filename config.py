"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralised configuration for the e-commerce customer service agent."""

    # --- LLM ---
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = "sk-placeholder"
    llm_model: str = "gpt-4o"

    # --- External RAG ---
    rag_base_url: str = "http://localhost:8080"
    rag_api_key: str = ""

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Database ---
    db_path: str = "./data/agent.db"

    # --- Session ---
    session_ttl_minutes: int = 30

    # --- Handoff ---
    max_return_attempts: int = 2

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
