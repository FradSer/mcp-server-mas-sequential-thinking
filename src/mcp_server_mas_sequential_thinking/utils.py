"""Simplified logging setup for the application."""

import logging
import logging.handlers
import os
import re
import sys
from pathlib import Path


class SensitiveDataFilter(logging.Filter):
    """High-performance filter to redact sensitive information from log messages."""
    
    # Pre-compiled patterns with optimized regex for better performance
    _SENSITIVE_PATTERNS = [
        (re.compile(r'(API_KEY|TOKEN|SECRET|PASSWORD)["\s]*[:=]["\s]*[^\s"\'<>&]{8,}', re.IGNORECASE), r'\1=***REDACTED***'),
        (re.compile(r'(Bearer\s+)[A-Za-z0-9\-._~+/]+=*', re.IGNORECASE), r'\1***REDACTED***'),
        (re.compile(r'(ghp_|github_pat_|gho_|ghu_)[A-Za-z0-9]{36,}', re.IGNORECASE), r'***REDACTED_GITHUB_TOKEN***'),
        (re.compile(r'sk-[A-Za-z0-9]{48}'), r'***REDACTED_OPENAI_KEY***'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter sensitive data from log record with optimized processing."""
        # Process message if present
        if hasattr(record, 'msg') and record.msg:
            record.msg = self._redact_sensitive_data(str(record.msg))
            
        # Process arguments efficiently
        if hasattr(record, 'args') and record.args:
            record.args = tuple(
                self._redact_sensitive_data(str(arg)) for arg in record.args
            )
            
        return True
    
    def _redact_sensitive_data(self, text: str) -> str:
        """Redact sensitive data from text with early exit optimization."""
        # Early exit for empty or very short strings
        if not text or len(text) < 8:
            return text
            
        # Apply patterns with short-circuit optimization
        for pattern, replacement in self._SENSITIVE_PATTERNS:
            if pattern.search(text):  # Only substitute if pattern found
                text = pattern.sub(replacement, text)
        
        return text


def setup_logging() -> logging.Logger:
    """Set up logging with simplified configuration and security filtering."""
    # Create logs directory with secure permissions
    log_dir = Path.home() / ".sequential_thinking" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    # Configure logger
    logger = logging.getLogger("sequential_thinking")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Simple formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Create singleton sensitive data filter for reuse
    sensitive_filter = SensitiveDataFilter()
    
    # Configure handlers with shared filter and formatter
    handlers = [
        logging.handlers.RotatingFileHandler(
            log_dir / "sequential_thinking.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
        ),
        logging.StreamHandler(sys.stderr),
    ]
    
    # Configure all handlers efficiently
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(sensitive_filter)
        logger.addHandler(handler)

    return logger
