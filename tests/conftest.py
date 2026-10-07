import pytest


@pytest.fixture(autouse=True)
def _no_automatic_warm_up(monkeypatch):
    """Most tests count connections: no pool warms up unless a test asks for it."""
    monkeypatch.setenv("MIRAI_WARM_CONNECTIONS", "0")
    monkeypatch.setenv("MIRAI_WARM_WEBSOCKETS", "0")
