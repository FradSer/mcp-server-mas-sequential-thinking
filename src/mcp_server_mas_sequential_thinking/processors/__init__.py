"""Processors module for MCP Sequential Thinking Server.

This module contains processing logic including six hats core functionality
and six hats processing implementation.
"""

from .six_hats_core import (
    HatColor,
    SixHatsAgentFactory,
    create_hat_agent,
    get_all_hat_colors,
    get_hat_timing,
)
from .six_hats_processor import (
    SixHatsProcessingResult,
    SixHatsSequentialProcessor,
    create_six_hats_step_output,
)

__all__ = [
    # From six_hats_core
    "HatColor",
    "SixHatsAgentFactory",
    # From six_hats_processor
    "SixHatsProcessingResult",
    "SixHatsSequentialProcessor",
    "create_hat_agent",
    "create_six_hats_step_output",
    "get_all_hat_colors",
    "get_hat_timing",
]
