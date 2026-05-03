"""Tests for application configuration."""

import os
from unittest.mock import patch

from llm_serving.config import Settings, get_settings


class TestSettings:
    """Tests for the Settings pydantic model."""

    def test_default_values(self) -> None:
        """Settings should have sensible defaults."""
        settings = Settings()
        assert settings.app_env == "development"
        assert settings.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert settings.device == "cpu"
        assert settings.max_new_tokens == 256
        assert settings.generation_timeout_s == 60.0
        assert settings.max_concurrent_requests == 1
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.log_level == "info"
        assert settings.redis_url == "redis://localhost:6379/0"
        assert settings.rate_limit_bucket_size == 10
        assert settings.rate_limit_refill_rate == 2.0
        assert settings.max_queue_depth == 100
        assert settings.circuit_breaker_failure_threshold == 5
        assert settings.circuit_breaker_recovery_timeout_s == 30.0
        assert settings.max_batch_size == 8
        assert settings.max_batch_wait_ms == 50
        assert settings.batching_enabled is True

    def test_override_via_constructor(self) -> None:
        """Settings should accept overrides via constructor kwargs."""
        settings = Settings(
            app_env="production",
            model_name="my-custom-model",
            device="cuda",
            max_new_tokens=512,
            max_concurrent_requests=4,
        )
        assert settings.app_env == "production"
        assert settings.model_name == "my-custom-model"
        assert settings.device == "cuda"
        assert settings.max_new_tokens == 512
        assert settings.max_concurrent_requests == 4

    def test_env_var_override(self) -> None:
        """Settings should read from LLM_ prefixed env vars."""
        with patch.dict(os.environ, {"LLM_MODEL_NAME": "env-model", "LLM_PORT": "9000"}):
            settings = Settings()
            assert settings.model_name == "env-model"
            assert settings.port == 9000

    def test_env_prefix(self) -> None:
        """Settings should only read LLM_ prefixed env vars, not bare names."""
        with patch.dict(os.environ, {"MODEL_NAME": "wrong-model"}, clear=False):
            settings = Settings()
            # Should NOT pick up MODEL_NAME without the LLM_ prefix
            assert settings.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_model_dump(self) -> None:
        """Settings should be serializable to a dict."""
        settings = Settings(model_name="test-model")
        data = settings.model_dump()
        assert isinstance(data, dict)
        assert data["model_name"] == "test-model"
        assert "app_env" in data

    def test_get_settings_returns_settings_instance(self) -> None:
        """get_settings() should return a Settings instance."""
        # Clear cache to get a fresh instance
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)
        get_settings.cache_clear()

    def test_get_settings_is_cached(self) -> None:
        """get_settings() should return the same instance on repeated calls."""
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        get_settings.cache_clear()
