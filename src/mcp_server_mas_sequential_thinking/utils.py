"""Application utilities - completely rewritten for new architecture."""

from .logging_config import setup_logging, get_logger

# Re-export for convenience - no backward compatibility
__all__ = ["setup_logging", "get_logger"]