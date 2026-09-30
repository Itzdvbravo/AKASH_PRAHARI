"""Structured logging configuration for TerraEyes backend."""
import logging
import sys
from typing import Any, Dict

try:
    import structlog

    def setup_logging(log_level: str = "INFO") -> None:
        level = getattr(logging, log_level.upper(), logging.INFO)
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=level,
        )

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(colors=True)
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def get_logger(name: str):
        return structlog.get_logger(name)

except ImportError:
    def setup_logging(log_level: str = "INFO") -> None:
        level = getattr(logging, log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

    def get_logger(name: str):
        return logging.getLogger(name)
