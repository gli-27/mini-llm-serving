"""Structured logging configuration using structlog.

Provides JSON-formatted logs with correlation IDs, timestamps, and
consistent key-value pairs for machine-parseable observability.
"""

import logging
import sys

import structlog


def setup_logging(log_level: str = "info", json_format: bool = True) -> None:
    """Configure structlog for the application.

    Sets up structlog with JSON rendering (production) or console rendering
    (development), shared processors for timestamps, log level, and caller
    info, and bridges stdlib logging so third-party libraries (uvicorn,
    transformers) also emit structured logs.

    Args:
        log_level: Logging level string (debug, info, warning, error, critical).
        json_format: If True, render logs as JSON. If False, use colored console output.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Shared processors applied to every log event
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to use structlog formatting — this ensures
    # third-party libraries (uvicorn, transformers, torch) also produce
    # structured output through the same pipeline.
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Quiet noisy loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog bound logger for the given module name.

    Args:
        name: The logger name, typically ``__name__`` of the calling module.

    Returns:
        A structlog BoundLogger instance with structured output.
    """
    return structlog.get_logger(name)
