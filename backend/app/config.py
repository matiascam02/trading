"""
Application configuration management.
Handles environment variables, LLM provider settings (Azure OpenAI, OpenAI, Gemini),
and database configuration.
"""

import os
from pathlib import Path
from typing import Optional

# Load environment variables
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Find .env file (project root)
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback: load from current directory
    load_dotenv(".env")


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LLM Stock Backtesting Dashboard"

    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # Database settings (future use)
    DATABASE_URL: Optional[str] = None

    # External API settings
    YFINANCE_TIMEOUT: int = 30

    # Backtesting settings
    DEFAULT_INITIAL_CAPITAL: float = 10000.0
    DEFAULT_COMMISSION: float = 0.001

    # OpenAI (Direct) settings
    # These settings enable calling OpenAI models directly, e.g. gpt-4o, gpt-4o-mini, gpt-5 (future)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    # Optional explicit selection: "openai" | "azure" | "gemini"
    LLM_PROVIDER: Optional[str] = None

    # Azure OpenAI settings
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4.1"

    # Google Gemini settings
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Additional .env settings
    REDIS_URL: Optional[str] = None
    BACKEND_URL: Optional[str] = None
    FRONTEND_URL: Optional[str] = None
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = "../../.env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
