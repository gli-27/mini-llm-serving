"""Tests for structured logging configuration."""

import logging

from llm_serving.logging import get_logger, setup_logging


class TestSetupLogging:
    """Tests for setup_logging() configuration."""

    def test_setup_logging_configures_root_logger(self) -> None:
        """setup_logging() should configure the root logger with a handler."""
        setup_logging(log_level="info", json_format=True)

        root = logging.getLogger()
        assert len(root.handlers) >= 1
        assert root.level == logging.INFO

    def test_setup_logging_respects_log_level(self) -> None:
        """setup_logging() should set the correct log level."""
        setup_logging(log_level="debug", json_format=True)
        assert logging.getLogger().level == logging.DEBUG

        setup_logging(log_level="warning", json_format=True)
        assert logging.getLogger().level == logging.WARNING

    def test_setup_logging_invalid_level_defaults_to_info(self) -> None:
        """setup_logging() should default to INFO for invalid log levels."""
        setup_logging(log_level="nonexistent", json_format=True)
        assert logging.getLogger().level == logging.INFO

    def test_setup_logging_quiets_noisy_loggers(self) -> None:
        """setup_logging() should set transformers and torch loggers to WARNING."""
        setup_logging(log_level="debug", json_format=True)

        assert logging.getLogger("transformers").level == logging.WARNING
        assert logging.getLogger("torch").level == logging.WARNING


class TestGetLogger:
    """Tests for get_logger() factory."""

    def test_get_logger_returns_bound_logger(self) -> None:
        """get_logger() should return a structlog BoundLogger."""
        setup_logging(log_level="info", json_format=True)
        logger = get_logger("test.module")
        assert logger is not None

    def test_get_logger_with_module_name(self) -> None:
        """get_logger() should accept a module name string."""
        setup_logging(log_level="info", json_format=True)
        logger = get_logger(__name__)
        assert logger is not None
