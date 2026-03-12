"""Processors module exports with lazy loading.

Lazy imports prevent circular dependencies between routing and processor modules.
"""

from __future__ import annotations

import importlib

__all__ = [
    "MultiThinkingAgentFactory",
    "MultiThinkingProcessingResult",
    "MultiThinkingSequentialProcessor",
    "ThinkingDirection",
    "create_multi_thinking_step_output",
    "create_thinking_agent",
    "get_all_thinking_directions",
    "get_thinking_timing",
    "multi_thinking_core",
    "multi_thinking_processor",
]


def __getattr__(name: str):
    if name == "multi_thinking_core":
        return importlib.import_module(".multi_thinking_core", __name__)

    if name in {
        "MultiThinkingAgentFactory",
        "ThinkingDirection",
        "create_thinking_agent",
        "get_all_thinking_directions",
        "get_thinking_timing",
    }:
        module = importlib.import_module(".multi_thinking_core", __name__)
        return getattr(module, name)

    if name in {
        "multi_thinking_processor",
        "MultiThinkingProcessingResult",
        "MultiThinkingSequentialProcessor",
        "create_multi_thinking_step_output",
    }:
        module = importlib.import_module(".multi_thinking_processor", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
