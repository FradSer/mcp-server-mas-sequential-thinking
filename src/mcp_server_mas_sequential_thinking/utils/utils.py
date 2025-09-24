"""Application utilities - completely rewritten for new architecture."""

from mcp_server_mas_sequential_thinking.infrastructure.logging_config import (
    get_logger,
    setup_logging,
)

# Re-export for convenience - no backward compatibility
__all__ = ["get_logger", "setup_logging"]
