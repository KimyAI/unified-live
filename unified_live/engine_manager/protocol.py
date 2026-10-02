"""Private binary IPC: 4-byte big-endian JSON length, JSON, then raw array bytes."""
from __future__ import annotations

import json
import struct
from typing import BinaryIO

import numpy as np

MAX_META = 16_384
MAX_DATA = 64 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    parts: list[bytes] = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            raise EOFError("IPC stream closed")
        parts.append(chunk)
        size -= len(chunk)
    return b"".join(parts)


def _write_all(stream: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise BrokenPipeError("IPC write failed")
        view = view[written:]


def _validate(meta: dict, data: bytes) -> np.ndarray | None:
    if not isinstance(meta, dict) or type(meta.get("id")) is not int or meta["id"] < 0:
        raise ProtocolError("Invalid request ID")
    kind = meta.get("kind")
    if kind not in {"control", "video", "audio", "error"}:
        raise ProtocolError("Invalid message kind")
    if kind in {"control", "error"}:
        if data:
            raise ProtocolError("Unexpected data")
        return None
    dtype = "uint8" if kind == "video" else "float32"
    shape = meta.get("shape")
    if meta.get("dtype") != dtype or not isinstance(shape, list) or not all(type(x) is int and x > 0 for x in shape):
        raise ProtocolError("Invalid array format")
    if kind == "video" and (len(shape) != 3 or shape[2] != 3 or shape[0] > 8192 or shape[1] > 8192):
        raise ProtocolError("Invalid video shape")
    if kind == "audio" and (len(shape) not in (1, 2) or (len(shape) == 2 and shape[1] != 1) or shape[0] > 1_000_000):
        raise ProtocolError("Invalid audio shape")
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if expected != len(data):
        raise ProtocolError("Array length mismatch")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def write_message(stream: BinaryIO, meta: dict, array: np.ndarray | None = None) -> None:
    meta = dict(meta)
    if array is not None:
        if array.nbytes > MAX_DATA:
            raise ProtocolError("Data too large")
        meta.update(dtype=str(array.dtype), shape=list(array.shape))
    raw = b"" if array is None else np.ascontiguousarray(array).tobytes()
    meta["data_bytes"] = len(raw)
    if len(raw) > MAX_DATA:
        raise ProtocolError("Data too large")
    _validate(meta, raw)
    encoded = json.dumps(meta, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_META:
        raise ProtocolError("Metadata too large")
    _write_all(stream, struct.pack(">I", len(encoded)))
    _write_all(stream, encoded)
    if raw:
        _write_all(stream, raw)
    stream.flush()


def read_message(stream: BinaryIO) -> tuple[dict, np.ndarray | None]:
    meta_len = struct.unpack(">I", _read_exact(stream, 4))[0]
    if not 2 <= meta_len <= MAX_META:
        raise ProtocolError("Invalid metadata length")
    try:
        meta = json.loads(_read_exact(stream, meta_len))
    except (ValueError, UnicodeError) as exc:
        raise ProtocolError("Invalid metadata JSON") from exc
    if not isinstance(meta, dict) or type(meta.get("data_bytes")) is not int:
        raise ProtocolError("Invalid metadata")
    length = meta["data_bytes"]
    if not 0 <= length <= MAX_DATA:
        raise ProtocolError("Invalid data length")
    data = _read_exact(stream, length)
    return meta, _validate(meta, data)
