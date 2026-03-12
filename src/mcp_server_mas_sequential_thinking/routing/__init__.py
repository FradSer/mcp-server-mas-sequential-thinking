"""Routing module for MCP Sequential Thinking Server."""

from __future__ import annotations

import importlib

__all__ = [
    "AIComplexityAnalyzer",
    "ComplexityLevel",
    "MultiThinkingIntelligentRouter",
    "MultiThinkingWorkflowResult",
    "MultiThinkingWorkflowRouter",
    "ProcessingStrategy",
    "create_multi_thinking_router",
]


def __getattr__(name: str):
    if name == "AIComplexityAnalyzer":
        module = importlib.import_module(".ai_complexity_analyzer", __name__)
        return getattr(module, name)
    if name in {"ComplexityLevel", "ProcessingStrategy"}:
        module = importlib.import_module(".complexity_types", __name__)
        return getattr(module, name)
    if name in {"MultiThinkingIntelligentRouter", "create_multi_thinking_router"}:
        module = importlib.import_module(".multi_thinking_router", __name__)
        return getattr(module, name)
    if name in {"MultiThinkingWorkflowResult", "MultiThinkingWorkflowRouter"}:
        module = importlib.import_module(".agno_workflow_router", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
