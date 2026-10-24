"""Application settings, loaded from environment / .env.

Scaffolding only: these are the knobs the future pipeline will read.
Nothing here talks to a database or an LLM yet.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Postgres (docker-compose service `db`); credentials here should match
    # POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB in .env.
    database_url: str = "postgresql+psycopg://comparer:comparer@localhost:5432/comparer"

    # LLM provider key. Optional: the answering pipeline is not built yet,
    # so the app runs fine without it.
    llm_api_key: str | None = None

    # Embedding model identifier used when the retrieval layer is built.
    embedding_model: str = "text-embedding-3-small"

    # Hard monthly spend ceiling for API calls, in USD.
    monthly_cost_cap_usd: float = 50.0

    # Where the source corpus lives (relative to repo root by default).
    corpus_dir: Path = REPO_ROOT / "corpus"


settings = Settings()
