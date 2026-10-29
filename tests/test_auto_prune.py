"""Tests for best-effort dead-link cleanup."""

from __future__ import annotations

from autoviz.store.repository import auto_prune_dead_links


class _Store:
    def __init__(self, result):
        self.result = result

    def prune_dead_artifacts(self):
        return self.result


def test_auto_prune_returns_counts():
    result = auto_prune_dead_links(store=_Store({"runs": 2, "charts": 3}))

    assert result == {"runs": 2, "charts": 3}


def test_auto_prune_swallows_errors():
    class BrokenStore:
        def prune_dead_artifacts(self):
            raise RuntimeError("database down")

    result = auto_prune_dead_links(store=BrokenStore())

    assert result["runs"] == 0
    assert result["charts"] == 0
    assert "database down" in str(result["error"])


def test_auto_prune_without_database_is_noop(monkeypatch):
    from autoviz.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "")

    assert auto_prune_dead_links() == {"runs": 0, "charts": 0}
