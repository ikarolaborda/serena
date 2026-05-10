"""Tests for serena.util.secrets_crypto and the encryption layer of SecretsStore."""
from __future__ import annotations

import base64
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from serena.util.secrets_crypto import (
    AES_KEY_BYTES,
    CIPHER_AES_GCM_V1,
    CIPHER_PLAIN,
    DecryptionError,
    KEYFILE_MAGIC,
    KeyfileEncryption,
    NONCE_BYTES,
    PassphraseEncryption,
    PlainEncryption,
)
from serena.util.secrets_store import SecretsStore


# ---- KeyfileEncryption ------------------------------------------------------


class TestKeyfileEncryption:
    def test_round_trip(self, tmp_path: Path) -> None:
        enc = KeyfileEncryption.load_or_create(tmp_path / "k.key")
        ct = enc.encrypt("hello", scope="r2", key="R2_BUCKET")
        assert enc.decrypt(ct, scope="r2", key="R2_BUCKET") == "hello"

    def test_distinct_nonces_for_same_plaintext(self, tmp_path: Path) -> None:
        enc = KeyfileEncryption.load_or_create(tmp_path / "k.key")
        a = enc.encrypt("same", scope="r2", key="X")
        b = enc.encrypt("same", scope="r2", key="X")
        # AES-GCM nonces are random per encrypt; repeated encryption must
        # not produce the same ciphertext or we'd have a nonce-reuse bug.
        assert a != b

    def test_tampered_ciphertext_raises(self, tmp_path: Path) -> None:
        enc = KeyfileEncryption.load_or_create(tmp_path / "k.key")
        ct = enc.encrypt("hello", scope="r2", key="X")
        blob = bytearray(base64.b64decode(ct))
        blob[-1] ^= 0x01  # flip a bit in the tag
        tampered = base64.b64encode(bytes(blob)).decode("ascii")
        with pytest.raises(DecryptionError):
            enc.decrypt(tampered, scope="r2", key="X")

    def test_scope_or_key_swap_fails(self, tmp_path: Path) -> None:
        # AAD binds (scope, key) to the ciphertext: decrypting with the
        # wrong AAD must fail even if the ciphertext bytes are intact.
        enc = KeyfileEncryption.load_or_create(tmp_path / "k.key")
        ct = enc.encrypt("hello", scope="r2", key="R2_BUCKET")
        with pytest.raises(DecryptionError):
            enc.decrypt(ct, scope="r2", key="DIFFERENT_KEY")
        with pytest.raises(DecryptionError):
            enc.decrypt(ct, scope="not-r2", key="R2_BUCKET")

    def test_wrong_key_fails(self, tmp_path: Path) -> None:
        enc1 = KeyfileEncryption.load_or_create(tmp_path / "a.key")
        enc2 = KeyfileEncryption.load_or_create(tmp_path / "b.key")
        ct = enc1.encrypt("hello", scope="r2", key="X")
        with pytest.raises(DecryptionError):
            enc2.decrypt(ct, scope="r2", key="X")

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
    def test_keyfile_chmod_600(self, tmp_path: Path) -> None:
        path = tmp_path / "k.key"
        KeyfileEncryption.load_or_create(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_keyfile_persistence_across_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "k.key"
        enc1 = KeyfileEncryption.load_or_create(path)
        ct = enc1.encrypt("hello", scope="r2", key="X")
        enc2 = KeyfileEncryption.load_or_create(path)  # should reuse the same key
        assert enc2.decrypt(ct, scope="r2", key="X") == "hello"

    def test_keyfile_without_magic_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "k.key"
        path.write_text("not-our-key\nAAAAAA\n", encoding="utf-8")
        with pytest.raises(DecryptionError):
            KeyfileEncryption.load_or_create(path)

    def test_keyfile_with_magic_but_short_body_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "k.key"
        path.write_text(f"{KEYFILE_MAGIC}\n", encoding="utf-8")
        with pytest.raises(DecryptionError):
            KeyfileEncryption.load_or_create(path)


# ---- PassphraseEncryption ---------------------------------------------------


class TestPassphraseEncryption:
    def test_round_trip_with_same_salt(self) -> None:
        salt = PassphraseEncryption.new_salt()
        enc_a = PassphraseEncryption.derive("hunter2", salt)
        enc_b = PassphraseEncryption.derive("hunter2", salt)
        ct = enc_a.encrypt("hello", scope="r2", key="X")
        assert enc_b.decrypt(ct, scope="r2", key="X") == "hello"

    def test_wrong_passphrase_fails(self) -> None:
        salt = PassphraseEncryption.new_salt()
        enc_right = PassphraseEncryption.derive("hunter2", salt)
        enc_wrong = PassphraseEncryption.derive("hunter3", salt)
        ct = enc_right.encrypt("hello", scope="r2", key="X")
        with pytest.raises(DecryptionError):
            enc_wrong.decrypt(ct, scope="r2", key="X")

    def test_different_salts_diverge(self) -> None:
        enc_a = PassphraseEncryption.derive("hunter2", PassphraseEncryption.new_salt())
        enc_b = PassphraseEncryption.derive("hunter2", PassphraseEncryption.new_salt())
        ct = enc_a.encrypt("hello", scope="r2", key="X")
        with pytest.raises(DecryptionError):
            enc_b.decrypt(ct, scope="r2", key="X")


# ---- SecretsStore integration ----------------------------------------------


class TestSecretsStoreEncrypted:
    def test_round_trip_with_keyfile_encryption(self, tmp_path: Path) -> None:
        enc = KeyfileEncryption.load_or_create(tmp_path / "k.key")
        store = SecretsStore(path=tmp_path / "s.sqlite3", encryption=enc)
        store.set_secret("r2", "R2_BUCKET", "my-bucket")
        assert store.get_secret("r2", "R2_BUCKET") == "my-bucket"
        assert store.encryption_label == CIPHER_AES_GCM_V1

    def test_ciphertext_not_recoverable_from_raw_db(self, tmp_path: Path) -> None:
        enc = KeyfileEncryption.load_or_create(tmp_path / "k.key")
        db_path = tmp_path / "s.sqlite3"
        store = SecretsStore(path=db_path, encryption=enc)
        store.set_secret("r2", "R2_SECRET_ACCESS_KEY", "topsecret-plaintext-value")
        # Read the raw bytes column behind sqlite to assert the plaintext
        # never lands on disk.
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT value, cipher FROM secrets WHERE scope='r2' AND key='R2_SECRET_ACCESS_KEY'"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        stored_value, cipher = row
        assert cipher == CIPHER_AES_GCM_V1
        assert "topsecret-plaintext-value" not in stored_value
        assert "topsecret" not in stored_value

    def test_legacy_plain_rows_readable_after_upgrade(self, tmp_path: Path) -> None:
        # Simulate a pre-encryption DB: PlainEncryption write, then re-open
        # with encryption. Old row must still decode; new write upgrades.
        db_path = tmp_path / "s.sqlite3"
        plain = SecretsStore(path=db_path, encryption=PlainEncryption())
        plain.set_secret("r2", "OLD_KEY", "legacy-value")

        enc = KeyfileEncryption.load_or_create(tmp_path / "k.key")
        upgraded = SecretsStore(path=db_path, encryption=enc)
        assert upgraded.get_secret("r2", "OLD_KEY") == "legacy-value"

        upgraded.set_secret("r2", "OLD_KEY", "new-value")
        assert upgraded.get_secret("r2", "OLD_KEY") == "new-value"
        # The row's cipher should now be the encrypted one.
        conn = sqlite3.connect(db_path)
        try:
            (cipher,) = conn.execute(
                "SELECT cipher FROM secrets WHERE scope='r2' AND key='OLD_KEY'"
            ).fetchone()
        finally:
            conn.close()
        assert cipher == CIPHER_AES_GCM_V1

    def test_rotate_encryption_upgrades_legacy_rows(self, tmp_path: Path) -> None:
        db_path = tmp_path / "s.sqlite3"
        plain = SecretsStore(path=db_path, encryption=PlainEncryption())
        plain.set_secret("r2", "K1", "v1")
        plain.set_secret("r2", "K2", "v2")
        plain.set_secret("other", "X", "y")

        enc = KeyfileEncryption.load_or_create(tmp_path / "k.key")
        upgraded = SecretsStore(path=db_path, encryption=enc)
        n = upgraded.rotate_encryption()
        assert n == 3
        # Second call is a no-op.
        assert upgraded.rotate_encryption() == 0
        # Values still decode.
        assert upgraded.get_secret("r2", "K1") == "v1"
        assert upgraded.get_secret("r2", "K2") == "v2"
        assert upgraded.get_secret("other", "X") == "y"

    def test_wrong_key_after_keyfile_loss_surfaces_error(self, tmp_path: Path) -> None:
        # Write with one keyfile, lose it, try to open with another.
        # An operator with a fresh keyfile cannot impersonate the old one.
        db_path = tmp_path / "s.sqlite3"
        enc1 = KeyfileEncryption.load_or_create(tmp_path / "k1.key")
        store1 = SecretsStore(path=db_path, encryption=enc1)
        store1.set_secret("r2", "K", "v")

        enc2 = KeyfileEncryption.load_or_create(tmp_path / "k2.key")
        store2 = SecretsStore(path=db_path, encryption=enc2)
        with pytest.raises(DecryptionError):
            store2.get_secret("r2", "K")

    def test_default_factory_disabled_via_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SERENA_SECRETS_DISABLE_ENCRYPTION", "1")
        store = SecretsStore.default(path=tmp_path / "s.sqlite3")
        assert store.encryption_label == CIPHER_PLAIN

    def test_default_factory_passphrase_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SERENA_SECRETS_DISABLE_ENCRYPTION", raising=False)
        monkeypatch.setenv("SERENA_SECRETS_PASSPHRASE", "operator-secret")
        db_path = tmp_path / "s.sqlite3"
        store = SecretsStore.default(path=db_path)
        assert store.encryption_label == CIPHER_AES_GCM_V1
        store.set_secret("r2", "K", "v")
        # Reopen with the same passphrase: salt is read from store_meta, so
        # the derived key matches and the value is recoverable.
        store2 = SecretsStore.default(path=db_path)
        assert store2.get_secret("r2", "K") == "v"

    def test_default_factory_keyfile_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SERENA_SECRETS_DISABLE_ENCRYPTION", raising=False)
        monkeypatch.delenv("SERENA_SECRETS_PASSPHRASE", raising=False)
        db_path = tmp_path / "s.sqlite3"
        keyfile = db_path.with_name("secrets.key")
        store = SecretsStore.default(path=db_path)
        assert store.encryption_label == CIPHER_AES_GCM_V1
        assert keyfile.exists()
        # The keyfile is the master key file — not the database itself.
        assert keyfile != db_path

    def test_constants_have_expected_sizes(self) -> None:
        assert AES_KEY_BYTES == 32
        assert NONCE_BYTES == 12
