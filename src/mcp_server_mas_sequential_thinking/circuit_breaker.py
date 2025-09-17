"""Circuit breaker pattern for preventing infinite loops and cascading failures (hotfix)."""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 3  # Number of failures before opening
    timeout_seconds: int = 60   # Time to wait before trying half-open
    success_threshold: int = 2  # Successes needed to close from half-open


@dataclass
class CircuitBreaker:
    """Simple circuit breaker for preventing cascading failures."""

    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        self._update_state()
        return self._state

    def _update_state(self) -> None:
        """Update circuit state based on current conditions."""
        if self._state == CircuitState.OPEN:
            if (self._last_failure_time and
                time.time() - self._last_failure_time >= self.config.timeout_seconds):
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")

    def can_proceed(self) -> bool:
        """Check if operation can proceed."""
        current_state = self.state
        if current_state == CircuitState.OPEN:
            return False
        return True

    def record_success(self) -> None:
        """Record a successful operation."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info(f"Circuit breaker '{self.name}' CLOSED - service recovered")
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' OPENED after {self._failure_count} failures"
                )
        elif self._state == CircuitState.HALF_OPEN:
            # Failed during testing, go back to open
            self._state = CircuitState.OPEN
            self._success_count = 0
            logger.warning(f"Circuit breaker '{self.name}' back to OPEN - test failed")

    def get_status(self) -> Dict[str, any]:
        """Get circuit breaker status for monitoring."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
        }


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                config=config or CircuitBreakerConfig()
            )
        return self._breakers[name]

    def get_status_all(self) -> Dict[str, Dict[str, any]]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_status() for name, breaker in self._breakers.items()}


# Global circuit breaker manager
_circuit_manager = CircuitBreakerManager()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get circuit breaker by name."""
    return _circuit_manager.get_breaker(name, config)


def get_all_circuit_status() -> Dict[str, Dict[str, any]]:
    """Get status of all circuit breakers."""
    return _circuit_manager.get_status_all()