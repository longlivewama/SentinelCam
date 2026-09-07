import pytest

from app.main import on_startup


def test_startup_refuses_default_jwt_secret_in_production(monkeypatch):
    import app.config as config_module

    monkeypatch.setattr(config_module.settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(config_module.settings, "JWT_SECRET_KEY", "change-me-in-production")

    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        on_startup()


def test_startup_allows_custom_secret_in_production(monkeypatch):
    import app.config as config_module

    monkeypatch.setattr(config_module.settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(config_module.settings, "JWT_SECRET_KEY", "a-real-random-secret")

    on_startup()  # should not raise


def test_startup_allows_default_secret_outside_production(monkeypatch):
    import app.config as config_module

    monkeypatch.setattr(config_module.settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(config_module.settings, "JWT_SECRET_KEY", "change-me-in-production")

    on_startup()  # should not raise
