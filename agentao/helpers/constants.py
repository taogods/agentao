from typing import Final

SENTINEL_FLOAT_FAILURE_VALUE: Final[float] = -1.
SENTINEL_INT_FAILURE_VALUE: Final[int] = -1
SENTINEL_STRING_FAILURE_VALUE: Final[str] = "N/A"

PRICING_DATA_PER_MILLION_TOKENS = {
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-2024-11-20": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-2024-08-06": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-audio-preview": {
        "text": {
            "input": 2.50,
            "output": 10.00,
        },
        "audio": {
            "input": 100.00,
            "output": 200.00,
        }
    },
    "gpt-4o-audio-preview-2024-10-01": {
        "text": {
            "input": 2.50,
            "output": 10.00,
        },
        "audio": {
            "input": 100.00,
            "output": 200.00,
        }
    },
    "gpt-4o-2024-05-13": {
        "input": 5.00,
        "output": 15.00,
    },
    "gpt-4o-mini": {
        "input": 0.150,
        "output": 0.600,
    },
    "gpt4omini": {
        "input": 0.150,
        "output": 0.600,
    },
    "gpt-4o-mini-2024-07-18": {
        "input": 0.150,
        "output": 0.600,
    },
    "o1-preview": {
        "input": 15.00,
        "output": 60.00,
    },
    "o1-preview-2024-09-12": {
        "input": 15.00,
        "output": 60.00,
    },
    "o1-mini": {
        "input": 3.00,
        "output": 12.00,
    },
    "o1-mini-2024-09-12": {
        "input": 3.00,
        "output": 12.00,
    },
    "claude-3.5-sonnet": {
        "input": 3.00,
        "output": 15.00,
    },
    "claude-3.5-haiku": {
        "input": 1.00,
        "output": 5.00,
    },
    "claude-3-opus": {
        "input": 15.00,
        "output": 75.00
    },
    "claude-3-haiku": {
        "input": 0.25,
        "output": 1.25,
    },
    "claude-3-sonnet": {
        "input": 3.00,
        "output": 15.00,
    }
}
