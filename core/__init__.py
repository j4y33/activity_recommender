"""
Activity Recommendation Agent System

A modular, conversation-aware activity recommendation system built with:
- Multi-agent architecture using OpenAI Agents SDK
- Instructor for reliable structured outputs
- Intelligent web scraping and content analysis
- Conversational memory and context tracking

Main modules:
- config: Configuration and environment management
- models: Core Pydantic data models
- memory: Conversation context and memory management
- tools: Function tools for agents
- agents: Agent definitions with Instructor
- services: External API integrations
- pipeline: Main orchestration workflows
- evaluation: Testing and validation
"""

from .config import (
    Config,
    get_config,
    initialize_config,
    get_instructor_client,
    get_openai_client,
    get_api_key,
)

__version__ = "0.1.0"
__author__ = "Activity Agent Team"

# Expose main configuration interface
__all__ = [
    "Config",
    "get_config", 
    "initialize_config",
    "get_instructor_client",
    "get_openai_client",
    "get_api_key",
] 