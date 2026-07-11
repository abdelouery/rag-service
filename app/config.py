from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-4-8"

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"

    data_dir: str = "data"
    chunk_tokens: int = 600
    chunk_overlap: int = 80

    log_level: str = "INFO"
