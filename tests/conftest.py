import pytest

from pipecat_mirai import edge as edge_module


@pytest.fixture(autouse=True)
def _no_automatic_warm_up(monkeypatch):
    """Most tests count connections: no pool warms up unless a test asks for it."""
    monkeypatch.setenv("MIRAI_WARM_CONNECTIONS", "0")
    monkeypatch.setenv("MIRAI_WARM_WEBSOCKETS", "0")
    monkeypatch.setenv("MIRAI_WARM_STT_WEBSOCKETS", "0")


@pytest.fixture(autouse=True)
def _gateway_only_unless_asked(monkeypatch):
    """Most tests talk to one fake server: no edge unless a test (test_edge.py) asks for it."""
    monkeypatch.setenv("MIRAI_TTS_EDGE", "off")
    monkeypatch.setenv("MIRAI_STT_EDGE", "off")
    edge_module._reset()
    yield
    edge_module._reset()
