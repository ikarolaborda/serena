"""Tests for serena.util.secrets_store."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from serena.util.secrets_store import SecretsStore, mask_secret


@pytest.fixture
def store(tmp_path: Path) -> SecretsStore:
    return SecretsStore(path=tmp_path / "secrets.sqlite3")


class TestMask:
    def test_short_values_collapse_to_stars(self) -> None:
        assert mask_secret("") == ""
        assert mask_secret(None) == ""
        assert mask_secret("short") == "****"
        assert mask_secret("a" * 12) == "****"  # threshold-exact

    def test_long_values_preserve_prefix_suffix(self) -> None:
        masked = mask_secret("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert masked.startswith("ABCD")
        assert masked.endswith("WXYZ")
        assert "****" in masked
        # Length should not equal original (no exact-length oracle).
        assert len(masked) < len("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class TestSecretsStore:
    def test_round_trip(self, store: SecretsStore) -> None:
        store.set_secret("r2", "R2_ACCOUNT_ID", "abc123")
        assert store.get_secret("r2", "R2_ACCOUNT_ID") == "abc123"

    def test_update_overwrites(self, store: SecretsStore) -> None:
        store.set_secret("r2", "R2_ACCOUNT_ID", "old")
        store.set_secret("r2", "R2_ACCOUNT_ID", "new")
        assert store.get_secret("r2", "R2_ACCOUNT_ID") == "new"

    def test_missing_returns_none(self, store: SecretsStore) -> None:
        assert store.get_secret("r2", "MISSING") is None

    def test_list_scope(self, store: SecretsStore) -> None:
        store.set_secret("r2", "A", "1")
        store.set_secret("r2", "B", "2")
        store.set_secret("other", "C", "3")
        assert store.list_scope("r2") == {"A": "1", "B": "2"}
        assert store.list_scope("other") == {"C": "3"}

    def test_list_scope_masked_never_leaks_raw(self, store: SecretsStore) -> None:
        secret = "supersecretvalue" * 4  # long enough to keep prefix/suffix
        store.set_secret("r2", "R2_SECRET_ACCESS_KEY", secret)
        masked = store.list_scope_masked("r2")["R2_SECRET_ACCESS_KEY"]
        assert "****" in masked
        # The middle of the secret must not appear in the masked rendering.
        assert secret[8:-8] not in masked

    def test_list_scopes(self, store: SecretsStore) -> None:
        store.set_secret("r2", "A", "1")
        store.set_secret("zzz", "X", "2")
        assert store.list_scopes() == ["r2", "zzz"]

    def test_delete_secret(self, store: SecretsStore) -> None:
        store.set_secret("r2", "A", "1")
        assert store.delete_secret("r2", "A") is True
        assert store.delete_secret("r2", "A") is False
        assert store.get_secret("r2", "A") is None

    def test_delete_scope(self, store: SecretsStore) -> None:
        store.set_secret("r2", "A", "1")
        store.set_secret("r2", "B", "2")
        assert store.delete_scope("r2") == 2
        assert store.list_scope("r2") == {}

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
    def test_db_created_with_mode_600(self, tmp_path: Path) -> None:
        path = tmp_path / "secrets.sqlite3"
        SecretsStore(path=path)
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    def test_empty_scope_or_key_rejected(self, store: SecretsStore) -> None:
        with pytest.raises(ValueError):
            store.set_secret("", "x", "y")
        with pytest.raises(ValueError):
            store.set_secret("r2", "", "y")
