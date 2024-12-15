from typing import List, Final

OPENAI_MODELS: Final[List[str]] = [
    "gpt4",
    "gpt4-legacy",
    "gpt4-0125",
    "gpt3-0125",
    "gpt4-turbo",
    "gpt4o",
    "gpt-4o-mini",
    "gpt4omini",
    "o1",
    "o1-mini",
]

ANTHROPIC_MODELS = [
    "claude-2",
    "claude-opus",
    "claude-sonnet",
    "claude-haiku",
    "claude-3-5-sonnet",
]

SUPPORTED_MINER_MODELS = OPENAI_MODELS + ANTHROPIC_MODELS

MODEL_NAME_TO_ENVAR_NAME = (
    {model: "OPENAI_API_KEY" for model in OPENAI_MODELS} |
    {model: "ANTHROPIC_API_KEY" for model in ANTHROPIC_MODELS}
)
