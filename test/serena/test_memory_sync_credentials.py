"""Tests for R2 credential discovery: .env.r2 fallback/import + non-clobber projection."""

from __future__ import annotations

from pathlib import Path

from serena.memory_sync import R2_AUTH_KEYS, MemorySyncService
from serena.util.secrets_store import SecretsStore

_AUTH_ENV = (
    "QDRANT_URL=http://localhost:6333\n"
    "R2_ACCOUNT_ID=acc-123\n"
    "R2_ACCESS_KEY_ID=ak-123\n"
    "R2_SECRET_ACCESS_KEY=sk-123\n"
    "R2_ENDPOINT=https://acc-123.r2.cloudflarestorage.com"
)


def _service(tmp_path: Path, env_contents: str | None) -> MemorySyncService:
    qdrant = tmp_path / "qdrant"
    qdrant.mkdir()
    if env_contents is not None:
        (qdrant / ".env.r2").write_text(env_contents + "\n", encoding="utf-8")
    store = SecretsStore(path=tmp_path / "secrets.sqlite3")
    return MemorySyncService(secrets_store=store, qdrant_memory_dir=qdrant)


def test_import_from_env_backfills_store_and_presence(tmp_path: Path) -> None:
    svc = _service(tmp_path, _AUTH_ENV)  # __init__ imports from .env.r2
    assert svc.credentials_present() is True
    status = svc.credential_status()
    for k in R2_AUTH_KEYS:
        assert status[k]["present"] is True
        assert status[k]["source"] == "secrets"  # imported into the encrypted store


def test_bucket_absent_does_not_block_presence(tmp_path: Path) -> None:
    svc = _service(tmp_path, _AUTH_ENV)  # no R2_BUCKET in env
    status = svc.credential_status()
    assert status["R2_BUCKET"]["present"] is False
    assert status["R2_BUCKET"]["required"] is False
    assert svc.credentials_present() is True


def test_missing_auth_key_reports_not_present(tmp_path: Path) -> None:
    partial = "\n".join([ln for ln in _AUTH_ENV.splitlines() if not ln.startswith("R2_ENDPOINT")])
    svc = _service(tmp_path, partial)
    assert svc.credentials_present() is False
    assert svc.credential_status()["R2_ENDPOINT"]["present"] is False


def test_import_is_idempotent_and_non_overwriting(tmp_path: Path) -> None:
    svc = _service(tmp_path, _AUTH_ENV)
    # Tamper a stored value, re-import: the existing (non-empty) value must win.
    svc.secrets.set_secret("r2", "R2_ACCOUNT_ID", "kept-by-store")
    imported = svc.import_r2_credentials_from_env()
    assert "R2_ACCOUNT_ID" not in imported
    assert svc.secrets.get_secret("r2", "R2_ACCOUNT_ID") == "kept-by-store"


def test_write_env_r2_merge_preserves_existing_keys(tmp_path: Path) -> None:
    svc = _service(tmp_path, _AUTH_ENV)
    target = tmp_path / "out.env.r2"
    target.write_text("QDRANT_URL=http://keep\nR2_BUCKET=manual-bucket\n", encoding="utf-8")
    svc.write_env_r2_file(target_path=target)
    parsed = dict(
        line.split("=", 1) for line in target.read_text(encoding="utf-8").splitlines() if "=" in line and not line.startswith("#")
    )
    assert parsed["QDRANT_URL"] == "http://keep"  # env-only key preserved
    assert parsed["R2_BUCKET"] == "manual-bucket"  # manually-set bucket preserved
    for k in R2_AUTH_KEYS:
        assert parsed[k]  # auth keys written, non-empty


def test_write_env_r2_never_clobbers_with_empty(tmp_path: Path) -> None:
    svc = _service(tmp_path, _AUTH_ENV)
    # An empty stored value must not overwrite a non-empty existing one.
    svc.secrets.set_secret("r2", "R2_ENDPOINT", "")
    target = tmp_path / "out.env.r2"
    target.write_text("R2_ENDPOINT=https://existing.example\n", encoding="utf-8")
    svc.write_env_r2_file(target_path=target)
    parsed = dict(
        line.split("=", 1) for line in target.read_text(encoding="utf-8").splitlines() if "=" in line and not line.startswith("#")
    )
    assert parsed["R2_ENDPOINT"] == "https://existing.example"


def test_no_env_file_yields_no_credentials(tmp_path: Path) -> None:
    svc = _service(tmp_path, None)
    assert svc.credentials_present() is False
    assert svc.import_r2_credentials_from_env() == []
