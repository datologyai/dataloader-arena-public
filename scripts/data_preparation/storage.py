"""Interchangeable local/S3 object storage for output and temporary records."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .s3 import S3, digest, json_bytes


class Local:
    def __init__(self, directory, *, create=True):
        self.path = Path(directory).expanduser().resolve()
        self.uri = str(self.path)
        self.read_only = not create
        self.metadata = self.path / ".preparation-metadata"
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
            self.metadata.mkdir(exist_ok=True)

    def key(self, name):
        if name.startswith("/") or any(p in ("", ".", "..") for p in name.split("/")):
            raise ValueError("Invalid relative object name")
        target = self.path / name
        if not target.resolve().is_relative_to(self.path):
            raise ValueError("Object path escapes its local storage directory")
        return target

    def meta(self, name):
        return self.metadata / (digest(name.encode()) + ".json")

    @contextmanager
    def locked(self):
        with (self.metadata / "objects.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def head(self, name):
        target = self.key(name)
        if not target.exists():
            return None
        with target.open("rb") as data:
            sha = hashlib.file_digest(data, "sha256").hexdigest()
        meta = (
            json.loads(self.meta(name).read_text()) if self.meta(name).exists() else {}
        )
        return {
            "ContentLength": target.stat().st_size,
            "ETag": sha,
            "ChecksumSHA256": base64.b64encode(bytes.fromhex(sha)).decode(),
            "Metadata": meta,
        }

    def put(self, name, data):
        sha = digest(data)
        owner = getattr(self, "owner", None)
        metadata = {"sha256": sha, **({"preparation-owner": owner} if owner else {})}
        target = self.key(name)
        with self.locked():
            import shutil

            if shutil.disk_usage(self.path).free < len(data) + 256 * 1024**2:
                raise ValueError("Insufficient space on local object storage volume")
            head = self.head(name)
            old_meta = (
                json.loads(self.meta(name).read_text())
                if self.meta(name).exists()
                else None
            )
            if head is None:
                if old_meta is not None and old_meta != metadata:
                    raise ValueError(
                        f"Local object was claimed by another writer: {name}"
                    )
                # Persist ownership first, so a completed payload with a lost
                # receipt remains resumable and can be safely abandoned.
                self.atomic(self.meta(name), json_bytes(metadata))
                self.atomic(target, data)
                head = self.head(name)
            if (
                head["ETag"] != sha
                or head["ContentLength"] != len(data)
                or head["Metadata"] != metadata
            ):
                raise ValueError(
                    f"Local object differs or belongs to another preparation: {name}"
                )
        return dict(
            name=name,
            bytes=len(data),
            sha256=sha,
            etag=sha,
            version_id=None,
            owner=owner,
        )

    @staticmethod
    def atomic(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".upload-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def get(self, name, *, start=None, length=None, etag=None):
        path = self.key(name)
        # Record checksums validate each returned range in Job.fetch. The
        # metadata identity avoids rereading a whole spool for every range.
        if etag and json.loads(self.meta(name).read_text()).get("sha256") != etag:
            raise ValueError(f"Local range identity changed: {name}")
        with path.open("rb") as handle:
            if start is not None:
                handle.seek(start)
            data = handle.read(length if length is not None else 1024**3 + 1)
        if len(data) > 1024**3 or (length is not None and len(data) != length):
            raise ValueError(f"Truncated or oversized local object: {name}")
        return data

    def verify(self, receipt):
        head = self.head(receipt["name"])
        if (
            head is None
            or head["ETag"] != receipt["sha256"]
            or head["ContentLength"] != receipt["bytes"]
            or (
                receipt.get("owner")
                and head["Metadata"].get("preparation-owner") != receipt["owner"]
            )
        ):
            raise ValueError(f"Missing or changed local object: {receipt['name']}")

    def delete_owned(self, name, *, version_id=None):
        if version_id is not None:
            raise ValueError("Local objects do not have S3 versions")
        with self.locked():
            head = self.head(name)
            if head is None:
                return 0
            if (
                not getattr(self, "owner", None)
                or head["Metadata"].get("preparation-owner") != self.owner
            ):
                raise ValueError(
                    f"Refusing to delete an object owned by another preparation: {name}"
                )
            self.key(name).unlink()
            self.meta(name).unlink(missing_ok=True)
            return head["ContentLength"]


def open_store(destination, *, create=True):
    if str(destination).startswith("s3://"):
        return S3(str(destination))
    if "://" in str(destination):
        raise ValueError("Use a local directory or s3://bucket/prefix")
    return Local(destination, create=create)
