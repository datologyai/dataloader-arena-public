"""Checksummed, conditional S3 writes. No dataset-sized local staging."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def location(uri: str) -> tuple[str, str]:
    p = urlsplit(uri)
    if (
        p.scheme != "s3"
        or not p.netloc
        or not p.path.strip("/")
        or p.query
        or p.fragment
        or any(c in p.netloc for c in "@:")
        or any(c.isspace() for c in uri)
        or any(part in (".", "..", "") for part in p.path.strip("/").split("/"))
    ):
        raise ValueError("Use an s3://bucket/nonempty-prefix destination")
    return p.netloc, p.path.strip("/")


class S3:
    def __init__(self, uri: str, client=None):
        self.bucket, self.prefix = location(uri)
        self.uri = uri.rstrip("/")
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "s3",
                config=Config(
                    retries={"mode": "standard", "max_attempts": 12},
                    max_pool_connections=32,
                    connect_timeout=20,
                    read_timeout=180,
                ),
            )
        self.client = client

    def key(self, name):
        if name.startswith("/") or any(p in ("", ".", "..") for p in name.split("/")):
            raise ValueError("Invalid relative S3 object name")
        return f"{self.prefix}/{name}"

    def head(self, name):
        try:
            return self.client.head_object(
                Bucket=self.bucket, Key=self.key(name), ChecksumMode="ENABLED"
            )
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if str(code) in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def put(self, name: str, data: bytes) -> dict:
        """Never replace different bytes; deletion is not part of recovery."""
        sha = digest(data)
        checksum = base64.b64encode(bytes.fromhex(sha)).decode()
        existing = self.head(name)
        if existing is None:
            try:
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=self.key(name),
                    Body=data,
                    IfNoneMatch="*",
                    ChecksumSHA256=checksum,
                    Metadata={
                        "sha256": sha,
                        **(
                            {"preparation-owner": self.owner}
                            if getattr(self, "owner", None)
                            else {}
                        ),
                    },
                )
            except Exception as error:
                code = getattr(error, "response", {}).get("Error", {}).get("Code")
                if str(code) not in ("412", "PreconditionFailed"):
                    raise
            existing = self.head(name)
        if (
            existing is None
            or existing["ContentLength"] != len(data)
            or existing.get("ChecksumSHA256") != checksum
            or (
                getattr(self, "owner", None)
                and existing.get("Metadata", {}).get("preparation-owner") != self.owner
            )
        ):
            raise ValueError(
                f"S3 object differs or its server checksum is unavailable: {name}"
            )
        return {
            "name": name,
            "bytes": len(data),
            "sha256": sha,
            "etag": existing["ETag"],
            "version_id": existing.get("VersionId"),
            "owner": getattr(self, "owner", None),
        }

    def get(self, name, *, start=None, length=None, etag=None):
        import time

        from botocore.exceptions import BotoCoreError

        for attempt in range(5):
            try:
                return self._read(name, start=start, length=length, etag=etag)
            except (BotoCoreError, OSError):
                # Request retries do not cover failures halfway through Body.read().
                if attempt == 4:
                    raise
                time.sleep(min(8, 2**attempt))

    def _read(self, name, *, start=None, length=None, etag=None):
        kwargs = {"Bucket": self.bucket, "Key": self.key(name)}
        if start is not None:
            kwargs["Range"] = f"bytes={start}-{start + length - 1}"
        if etag:
            kwargs["IfMatch"] = etag
        response = self.client.get_object(**kwargs)
        with response["Body"] as body:
            data = body.read(1024**3 + 1)
        if len(data) > 1024**3:
            raise ValueError(
                f"S3 object exceeds the bounded preparation read size: {name}"
            )
        if length is not None and len(data) != length:
            raise ValueError(f"Truncated S3 range: {name}")
        return data

    def verify(self, receipt):
        head = self.head(receipt["name"])
        checksum = base64.b64encode(bytes.fromhex(receipt["sha256"])).decode()
        if (
            head is None
            or head["ContentLength"] != receipt["bytes"]
            or head.get("ChecksumSHA256") != checksum
            or (
                receipt.get("version_id")
                and head.get("VersionId") != receipt["version_id"]
            )
            or (
                receipt.get("owner")
                and head.get("Metadata", {}).get("preparation-owner")
                != receipt["owner"]
            )
        ):
            raise ValueError(f"Missing or changed S3 object: {receipt['name']}")

    def delete_owned(self, name, *, version_id=None):
        """Delete one exact object only after verifying this preparation's tag."""
        head = self.head(name)
        if head is None:
            return 0
        if version_id and head.get("VersionId") != version_id:
            raise ValueError(f"Refusing to delete a different S3 version: {name}")
        if (
            not getattr(self, "owner", None)
            or head.get("Metadata", {}).get("preparation-owner") != self.owner
        ):
            raise ValueError(
                f"Refusing to delete an object owned by another preparation: {name}"
            )
        kwargs = {"Bucket": self.bucket, "Key": self.key(name)}
        if head.get("VersionId"):
            kwargs["VersionId"] = head["VersionId"]
        self.client.delete_object(**kwargs)
        return head["ContentLength"]


class Scratch:
    """A private working directory, with space checks before bounded allocations."""

    def __init__(self, path, limit):
        self.path = Path(path).expanduser().resolve()
        self.limit = limit
        self.roots = [self.path]
        self.path.mkdir(parents=True, exist_ok=True)

    def check(self, additional=0, *, destination=None):
        import shutil

        files = {}
        for root in self.roots:
            for path in root.rglob("*"):
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    info = path.stat()
                except FileNotFoundError:
                    continue  # A concurrent cache download may replace its temporary file.
                files[(info.st_dev, info.st_ino)] = info.st_size
        used = sum(files.values())
        if used + additional > self.limit:
            raise ValueError(
                "Local scratch limit would be exceeded; preserve this directory for resume"
            )
        destination = Path(destination).resolve() if destination else self.path
        # Payload allocations belong on scratch; a small persistent journal
        # volume needs only its metadata margin for this operation.
        if shutil.disk_usage(destination).free < additional + 256 * 1024**2 or any(
            shutil.disk_usage(root).free < 256 * 1024**2 for root in self.roots
        ):
            raise ValueError(
                "Insufficient free disk space for the next preparation operation"
            )
