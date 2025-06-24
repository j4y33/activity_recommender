"""
Configuration and environment management for the activity recommendation system.

This module centralizes:
- API key management and validation
- Instructor client initialization
- Provider configurations
- Environment setup with proper error handling
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path

import instructor
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


class APIConfig(BaseModel):
    """Configuration for external APIs."""
    
    openai_api_key: str = Field(..., min_length=1)
    openweather_api_key: str = Field(..., min_length=1)
    firecrawl_api_key: str = Field(..., min_length=1)
    
    class Config:
        extra = "ignore"


class ModelConfig(BaseModel):
    """Configuration for LLM models and providers."""
    
    default_model: str = Field(default="openai/gpt-4o-mini")
    intent_model: str = Field(default="openai/gpt-4o-mini")
    scraper_model: str = Field(default="openai/gpt-4o-mini")
    answer_model: str = Field(default="openai/gpt-4o-mini")
    analysis_model: str = Field(default="openai/gpt-4o-mini")
    
    # Model parameters
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    max_retries: int = Field(default=3, gt=0)


class SystemConfig(BaseModel):
    """General system configuration."""
    
    max_search_results: int = Field(default=5, gt=0, le=10)
    max_scrape_retries: int = Field(default=2, ge=0, le=5)
    max_conversation_turns: int = Field(default=5, gt=0, le=10)
    scrape_timeout: int = Field(default=20, gt=0, le=60)
    
    # Content limits
    max_content_chars: int = Field(default=8000, gt=0)
    max_sub_pages: int = Field(default=3, gt=0, le=5)


class Config:
    """Main configuration class that manages all settings and clients."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration with environment variables.
        
        Args:
            env_file: Path to .env file. If None, searches for .env in current and parent dirs.
        """
        self._load_environment(env_file)
        self._api_config = self._load_api_config()
        self._model_config = ModelConfig()
        self._system_config = SystemConfig()
        self._instructor_client = None
        
    def _load_environment(self, env_file: Optional[str] = None) -> None:
        """Load environment variables from .env file."""
        global _env_loaded
        
        # Only load once to prevent duplicates
        if _env_loaded and not env_file:
            return
            
        if env_file:
            load_dotenv(env_file)
            print(f"✅ Loaded environment from: {env_file}")
        else:
            # Search for .env in current directory and parent directories
            current_dir = Path.cwd()
            for path in [current_dir, *current_dir.parents]:
                env_path = path / ".env"
                if env_path.exists():
                    load_dotenv(env_path)
                    if not _env_loaded:  # Only print on first load
                        print(f"✅ Loaded environment from: {env_path}")
                    break
            else:
                if not _env_loaded:  # Only print on first attempt
                    print("⚠️ No .env file found, using system environment variables")
        
        # Ensure OPENAI_API_KEY is available for Agents SDK
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            
        _env_loaded = True
    
    def _load_api_config(self) -> APIConfig:
        """Load and validate API configuration."""
        try:
            return APIConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                openweather_api_key=os.getenv("OPENWEATHER_API_KEY", ""),
                firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY", "")
            )
        except ValidationError as e:
            missing_keys = []
            for error in e.errors():
                if error["type"] == "value_error.missing":
                    missing_keys.append(error["loc"][0])
            
            raise ValueError(
                f"Missing required API keys: {', '.join(missing_keys)}. "
                "Please add them to your .env file or environment variables."
            ) from e
    
    @property
    def api(self) -> APIConfig:
        """Get API configuration."""
        return self._api_config
    
    @property
    def models(self) -> ModelConfig:
        """Get model configuration."""
        return self._model_config
    
    @property
    def system(self) -> SystemConfig:
        """Get system configuration."""
        return self._system_config
    
    def get_instructor_client(self, provider: Optional[str] = None) -> instructor.Instructor:
        """
        Get Instructor client with proper configuration.
        
        Args:
            provider: Model provider (e.g., "openai/gpt-4o-mini"). Uses default if None.
            
        Returns:
            Configured Instructor client
        """
        if self._instructor_client is None:
            self._instructor_client = self._create_instructor_client(provider)
        return self._instructor_client
    
    def _create_instructor_client(self, provider: Optional[str] = None) -> instructor.Instructor:
        """Create and configure Instructor client."""
        try:
            # Create standard Instructor client with OpenAI
            openai_client = AsyncOpenAI(api_key=self._api_config.openai_api_key)
            client = instructor.from_openai(openai_client)
            
            # Only print on first initialization
            if self._instructor_client is None:
                print(f"✅ Instructor client initialized successfully")
            return client
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Instructor client: {e}") from e
    
    def get_openai_client(self) -> AsyncOpenAI:
        """Get async OpenAI client for direct API calls when needed."""
        return AsyncOpenAI(api_key=self._api_config.openai_api_key)
    
    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that all required services are properly configured.
        
        Returns:
            Dictionary with validation results for each service
        """
        results = {}
        
        # Check API keys
        results["openai_key"] = bool(self._api_config.openai_api_key)
        results["openweather_key"] = bool(self._api_config.openweather_api_key)
        results["firecrawl_key"] = bool(self._api_config.firecrawl_api_key)
        
        # Test Instructor client
        try:
            self.get_instructor_client()
            results["instructor_client"] = True
        except Exception:
            results["instructor_client"] = False
        
        return results
    
    def print_status(self) -> None:
        """Print configuration status for debugging."""
        print("🔧 CONFIGURATION STATUS")
        print("=" * 40)
        
        validation = self.validate_setup()
        
        print("📚 API Keys:")
        for key, status in validation.items():
            if "_key" in key:
                service = key.replace("_key", "").title()
                icon = "✅" if status else "❌"
                print(f"   {icon} {service}")
        
        print("\n🤖 Instructor:")
        icon = "✅" if validation["instructor_client"] else "❌"
        print(f"   {icon} Client initialized")
        
        print(f"\n⚙️ Models:")
        print(f"   📝 Default: {self._model_config.default_model}")
        print(f"   🎯 Temperature: {self._model_config.temperature}")
        print(f"   🔄 Max retries: {self._model_config.max_retries}")
        
        print(f"\n🔧 System:")
        print(f"   🔍 Max search results: {self._system_config.max_search_results}")
        print(f"   ⏱️ Scrape timeout: {self._system_config.scrape_timeout}s")
        print(f"   💬 Max conversation turns: {self._system_config.max_conversation_turns}")


# Global configuration instance
_config: Optional[Config] = None
_env_loaded: bool = False  # Track if environment has been loaded to prevent duplicates


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Global Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def initialize_config(env_file: Optional[str] = None) -> Config:
    """
    Initialize or reinitialize the global configuration.
    
    Args:
        env_file: Path to .env file
        
    Returns:
        Configured Config instance
    """
    global _config
    _config = Config(env_file)
    return _config


# Convenience functions for common operations
def get_instructor_client(provider: Optional[str] = None) -> instructor.Instructor:
    """Get Instructor client from global config."""
    return get_config().get_instructor_client(provider)


def get_openai_client() -> AsyncOpenAI:
    """Get OpenAI client from global config."""
    return get_config().get_openai_client()


def get_api_key(service: str) -> str:
    """
    Get API key for a specific service.
    
    Args:
        service: Service name ("openai", "openweather", "firecrawl")
        
    Returns:
        API key for the service
    """
    config = get_config()
    service_map = {
        "openai": config.api.openai_api_key,
        "openweather": config.api.openweather_api_key,
        "firecrawl": config.api.firecrawl_api_key
    }
    
    if service not in service_map:
        raise ValueError(f"Unknown service: {service}. Available: {list(service_map.keys())}")
    
    return service_map[service] 