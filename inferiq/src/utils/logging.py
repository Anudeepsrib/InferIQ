"""Structured JSON logging configuration."""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger


def add_timestamp(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO format timestamp."""
    from datetime import datetime
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def add_log_level(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add log level to event."""
    event_dict["level"] = method_name.upper()
    return event_dict


def add_service_info(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add service identification."""
    event_dict["service"] = "inferiq"
    return event_dict


def configure_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: str | Path | None = None,
) -> structlog.BoundLogger:
    """Configure structured logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: "json" or "console"
        log_file: Optional file path for logging
        
    Returns:
        Configured structlog logger
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ExtraAdder(),
    ]
    
    if format_type == "json":
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(),
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level),
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(Path(log_file))
        file_handler.setLevel(getattr(logging, level))
        logging.getLogger().addHandler(file_handler)
    
    return structlog.get_logger()


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a configured structlog logger."""
    return structlog.get_logger(name)
