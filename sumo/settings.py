import os

from dotenv import load_dotenv

load_dotenv(override=True)


class Settings:
    """Configuration of app settings."""

    AGENT_STEPS_LIMIT: int = 10
    CHUNK_TOKENS_LIMIT: int = 1000

    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
    OPENAI_CHAT_MODEL: str = "gpt-3.5-turbo"

    LLM_DEFAULT_TEMPERATURE: float = 0.0


config = Settings()
