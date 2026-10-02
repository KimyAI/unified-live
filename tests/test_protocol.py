import io
import struct

import numpy as np
import pytest

from unified_live.engine_manager.protocol import ProtocolError, read_message, write_message


def test_binary_roundtrip_video_audio():
    for kind, array in (("video", np.arange(24, dtype=np.uint8).reshape(2, 4, 3)),
                        ("audio", np.linspace(-1, 1, 32, dtype=np.float32))):
        stream = io.BytesIO()
        write_message(stream, {"kind": kind, "id": 7}, array)
        stream.seek(0)
        meta, received = read_message(stream)
        assert meta["id"] == 7
        np.testing.assert_array_equal(received, array)


def test_protocol_rejects_oversized_header_and_wrong_dtype():
    with pytest.raises(ProtocolError):
        read_message(io.BytesIO(struct.pack(">I", 100_000)))
    with pytest.raises(ProtocolError):
        write_message(io.BytesIO(), {"kind": "video", "id": 1}, np.zeros((2, 2, 3), dtype=np.float32))
