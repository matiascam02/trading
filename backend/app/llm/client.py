"""
Unified LLM client for all modules
"""

import os
from typing import Literal, Optional, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from app.config import settings


def _detect_available_provider() -> Literal["azure", "openai", "gemini"]:
    """
    Auto-detect which LLM provider to use based on available API keys in .env

    Returns:
        "gemini" if only GOOGLE_API_KEY is available and uncommented
        "openai" if OPENAI_API_KEY is available and uncommented
        "azure" if AZURE_OPENAI_API_KEY is available and uncommented
        Prioritizes Azure, then OpenAI, then Gemini when multiple are available
        Defaults to "azure" for backward compatibility
    """
    # Read directly from environment variables (not from settings)
    # This allows users to comment out keys in .env to disable providers
    google_api_key = os.getenv("GOOGLE_API_KEY")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Check if keys are properly set and not placeholder values
    # Azure is considered available only if both API key AND endpoint are valid
    azure_available = (
        azure_api_key
        and azure_api_key.strip()
        and azure_api_key.lower() not in ["your_azure_api_key_here", "none", "null"]
        and azure_endpoint
        and azure_endpoint.strip()
        and azure_endpoint.lower() not in ["your_azure_openai_endpoint", "none", "null"]
    )

    openai_available = (
        openai_api_key
        and openai_api_key.strip()
        and openai_api_key.lower() not in ["your_openai_api_key_here", "none", "null"]
    )

    gemini_available = (
        google_api_key
        and google_api_key.strip()
        and google_api_key not in ["your_google_api_key_here", "", "none", "null"]
    )

    # Priority logic:
    # 1) If OpenAI key exists, prefer OpenAI (most common non-Azure setup)
    # 2) Else if full Azure config exists (key + endpoint), use Azure
    # 3) Else Gemini if available
    if openai_available:
        return "openai"
    elif azure_available:
        return "azure"
    elif gemini_available:
        return "gemini"
    else:
        # Default to azure for backward compatibility
        return "azure"


def get_llm_client(
    provider: Optional[Literal["azure", "openai", "gemini"]] = None,
    temperature: float = 0.1,
    max_tokens: int = 4000,
    **kwargs,
) -> Union[AzureChatOpenAI, ChatOpenAI, ChatGoogleGenerativeAI]:
    """
    Get unified LLM client with automatic provider detection

    Args:
        provider: LLM provider - "azure" for Azure OpenAI, "openai" for OpenAI, or "gemini" for Google Gemini
                 If None, auto-detects based on available API keys in .env
        temperature: Temperature parameter
        max_tokens: Maximum number of tokens
        **kwargs: Other provider-specific parameters

    Returns:
        LLM client instance (AzureChatOpenAI, ChatOpenAI, or ChatGoogleGenerativeAI)
    """
    # Allow explicit override via settings.LLM_PROVIDER
    if provider is None and settings and getattr(settings, "LLM_PROVIDER", None):
        explicit = str(settings.LLM_PROVIDER).strip().lower()
        if explicit in {"azure", "openai", "gemini"}:
            provider = explicit  # type: ignore[assignment]

    # Auto-detect provider if not explicitly provided
    if provider is None:
        provider = _detect_available_provider()

    # Debug provider selection
    try:
        print(f"[LLM] Selected provider: {provider}")
    except Exception:
        pass

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            **kwargs,
        )
    elif provider == "openai":
        # Ensure any Azure env vars don't force Azure mode in OpenAI SDK
        os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
        os.environ.pop("AZURE_OPENAI_API_KEY", None)
        os.environ.pop("AZURE_OPENAI_AD_TOKEN", None)
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=settings.OPENAI_API_KEY,
            **kwargs,
        )
    else:  # azure
        return AzureChatOpenAI(
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            api_key=settings.AZURE_OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


class LLMClientConfig:
    """LLM Client Configuration Class"""

    def __init__(
        self,
        provider: Optional[Literal["azure", "openai", "gemini"]] = None,
        # Azure OpenAI specific
        deployment_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        # OpenAI specific
        openai_api_key: Optional[str] = None,
        # Gemini specific
        google_api_key: Optional[str] = None,
        model: str = None,
        # Common parameters
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ):
        # Auto-detect provider if not specified
        self.provider = (
            provider if provider is not None else _detect_available_provider()
        )

        # Azure OpenAI configuration
        self.deployment_name = deployment_name or settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.endpoint = endpoint or settings.AZURE_OPENAI_ENDPOINT
        self.api_version = api_version or settings.AZURE_OPENAI_API_VERSION
        self.api_key = api_key or settings.AZURE_OPENAI_API_KEY

        # OpenAI configuration
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY

        # Gemini configuration
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        # Model configuration:
        # - For Gemini/OpenAI we use model name
        # - Azure uses deployment name instead
        if self.provider == "gemini":
            self.model = model or settings.GEMINI_MODEL
        else:
            self.model = model or settings.OPENAI_MODEL

        # Common parameters
        self.temperature = temperature
        self.max_tokens = max_tokens

    def create_client(self) -> Union[AzureChatOpenAI, ChatOpenAI, ChatGoogleGenerativeAI]:
        """Create client instance based on provider"""
        if self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                google_api_key=self.google_api_key,
            )
        elif self.provider == "openai":
            return ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.openai_api_key,
            )
        else:  # Azure OpenAI
            return AzureChatOpenAI(
                azure_deployment=self.deployment_name,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )


# Default configuration instance
default_config = LLMClientConfig()


def get_configured_client(
    config: Optional[LLMClientConfig] = None,
) -> Union[AzureChatOpenAI, ChatOpenAI, ChatGoogleGenerativeAI]:
    """
    Get client using configuration

    Args:
        config: LLM client configuration

    Returns:
        Configured LLM client (Azure OpenAI, OpenAI, or Google Gemini)
    """
    if config is None:
        config = default_config
    return config.create_client()
