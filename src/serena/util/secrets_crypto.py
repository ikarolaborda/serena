"""
Encryption-at-rest for the SecretsStore.

AES-256-GCM (AEAD: authenticated encryption with associated data) via pyca
``cryptography``. Authenticated means a ciphertext that was modified — or
encrypted under a different key — fails verification with
``cryptography.exceptions.InvalidTag`` instead of decrypting to garbage.

Threat model — what this defends against:

* DB-only leakage (rsync backup, cloud-sync of ``~/.serena``, accidental
  upload of ``secrets.sqlite3``). The DB without the keyfile is inert.
* Tampered ciphertext (AEAD tag catches it).

What it does NOT defend against:

* An attacker with full filesystem read access to the user account
  (they get the keyfile too).
* In-process memory dump.
* Malicious code running as the same user.

Two key modes:

* ``KeyfileEncryption`` (default): random 32-byte key stored in
  ``~/.serena/secrets.key`` with ``chmod 600``. Auto-created on first
  encrypted write. Zero operator interaction.
* ``PassphraseEncryption`` (opt-in via ``SERENA_SECRETS_PASSPHRASE``): key
  derived from a passphrase using scrypt with a per-store random salt.
  Defeats "FS access but no shell-session env" attackers but requires
  the passphrase in every process that opens the store.

Storage layout per row (``cipher = 'aes-gcm-v1'``)::

    base64( nonce(12B) || ciphertext_with_tag )

``scope`` and ``key`` are included as Associated Authenticated Data so
that an attacker cannot swap an encrypted blob between (scope, key)
pairs without invalidating the tag.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import os
import secrets as _secrets
from pathlib import Path
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

log = logging.getLogger(__name__)

CIPHER_PLAIN = "plain"
CIPHER_AES_GCM_V1 = "aes-gcm-v1"

KEYFILE_MAGIC = "serena-secrets-key-v1"
"""First line of the keyfile so an operator inspecting it can tell what it is."""

NONCE_BYTES = 12
"""NIST-recommended GCM nonce length."""

AES_KEY_BYTES = 32
"""AES-256."""

SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_SALT_BYTES = 16
"""scrypt parameters; matches OWASP 2025 cheatsheet recommendation for interactive use."""


class DecryptionError(Exception):
    """Wraps cryptography.exceptions.InvalidTag and decode errors with a stable boundary."""


class Encryption(Protocol):
    """Strategy interface for at-rest encryption of secret values.

    Implementations MUST be self-contained: ``encrypt`` returns a string
    ``SecretsStore`` can drop into the ``value`` column verbatim, and
    ``decrypt`` reads it back. The ``cipher_id`` identifies the layout so
    rows written by different strategies can coexist in one DB.
    """

    cipher_id: str

    def encrypt(self, plaintext: str, *, scope: str, key: str) -> str: ...
    def decrypt(self, encoded: str, *, scope: str, key: str) -> str: ...


class PlainEncryption:
    """Identity strategy. Stores values as-is. Used as the fallback for legacy rows."""

    cipher_id = CIPHER_PLAIN

    def encrypt(self, plaintext: str, *, scope: str, key: str) -> str:
        return plaintext

    def decrypt(self, encoded: str, *, scope: str, key: str) -> str:
        return encoded


class _AesGcmEncryptionBase:
    """Common AES-GCM path; subclasses supply the master key."""

    cipher_id = CIPHER_AES_GCM_V1

    def __init__(self, key_bytes: bytes) -> None:
        if len(key_bytes) != AES_KEY_BYTES:
            raise ValueError(f"AES-256-GCM expects {AES_KEY_BYTES}-byte key, got {len(key_bytes)}")
        self._aes = AESGCM(key_bytes)

    @staticmethod
    def _aad(scope: str, key: str) -> bytes:
        # Binding scope+key into the AEAD prevents an attacker from swapping
        # an encrypted blob between rows without the tag failing.
        return f"{scope}\x00{key}".encode("utf-8")

    def encrypt(self, plaintext: str, *, scope: str, key: str) -> str:
        nonce = os.urandom(NONCE_BYTES)
        ct = self._aes.encrypt(nonce, plaintext.encode("utf-8"), self._aad(scope, key))
        return base64.b64encode(nonce + ct).decode("ascii")

    def decrypt(self, encoded: str, *, scope: str, key: str) -> str:
        try:
            blob = base64.b64decode(encoded.encode("ascii"))
        except (ValueError, TypeError) as e:
            raise DecryptionError("invalid base64 in encrypted secret") from e
        if len(blob) <= NONCE_BYTES:
            raise DecryptionError("encrypted blob shorter than nonce")
        nonce, ct = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
        try:
            pt = self._aes.decrypt(nonce, ct, self._aad(scope, key))
        except InvalidTag as e:
            raise DecryptionError("authentication tag did not verify (tamper / wrong key)") from e
        return pt.decode("utf-8")


class KeyfileEncryption(_AesGcmEncryptionBase):
    """AES-256-GCM with the master key loaded from a chmod-600 file."""

    def __init__(self, key_bytes: bytes, keyfile_path: Path) -> None:
        super().__init__(key_bytes)
        self._keyfile_path = keyfile_path

    @property
    def keyfile_path(self) -> Path:
        return self._keyfile_path

    @classmethod
    def load_or_create(cls, keyfile_path: str | os.PathLike[str]) -> "KeyfileEncryption":
        path = Path(keyfile_path)
        if path.exists():
            return cls._load(path)
        return cls._create(path)

    @classmethod
    def _load(cls, path: Path) -> "KeyfileEncryption":
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise DecryptionError(f"could not read keyfile at {path}: {e}") from e
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
        if not lines or lines[0] != KEYFILE_MAGIC:
            raise DecryptionError(
                f"keyfile {path} missing magic header — refusing to interpret arbitrary bytes as a key"
            )
        if len(lines) < 2:
            raise DecryptionError(f"keyfile {path} has no key body")
        try:
            key_bytes = base64.b64decode(lines[1].encode("ascii"), validate=True)
        except (ValueError, TypeError) as e:
            raise DecryptionError(f"keyfile {path} body is not valid base64") from e
        return cls(key_bytes, path)

    @classmethod
    def _create(cls, path: Path) -> "KeyfileEncryption":
        path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(path.parent, 0o700)
        key_bytes = AESGCM.generate_key(bit_length=AES_KEY_BYTES * 8)
        encoded = base64.b64encode(key_bytes).decode("ascii")
        body = f"{KEYFILE_MAGIC}\n{encoded}\n"
        # Create with mode 600 atomically: use os.open + O_EXCL so we never
        # widen the perms on an existing file we don't own.
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(body)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(path)
            raise
        log.info("created new SecretsStore master key at %s (chmod 600)", path)
        return cls(key_bytes, path)


class PassphraseEncryption(_AesGcmEncryptionBase):
    """AES-256-GCM with a key derived from an operator passphrase via scrypt.

    The salt is stored alongside the database so the same passphrase keeps
    producing the same data-encryption key across restarts.
    """

    @classmethod
    def derive(cls, passphrase: str, salt: bytes) -> "PassphraseEncryption":
        if len(salt) != SCRYPT_SALT_BYTES:
            raise ValueError(f"salt must be {SCRYPT_SALT_BYTES} bytes")
        kdf = Scrypt(salt=salt, length=AES_KEY_BYTES, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
        key_bytes = kdf.derive(passphrase.encode("utf-8"))
        return cls(key_bytes)

    @staticmethod
    def new_salt() -> bytes:
        return _secrets.token_bytes(SCRYPT_SALT_BYTES)


__all__ = [
    "CIPHER_PLAIN",
    "CIPHER_AES_GCM_V1",
    "DecryptionError",
    "Encryption",
    "PlainEncryption",
    "KeyfileEncryption",
    "PassphraseEncryption",
    "AES_KEY_BYTES",
    "NONCE_BYTES",
    "KEYFILE_MAGIC",
]
