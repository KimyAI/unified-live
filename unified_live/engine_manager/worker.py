"""Isolated worker entry point. stdout belongs exclusively to protocol."""
from __future__ import annotations

import importlib
import os
import sys
import time

from unified_live.engine_manager.protocol import read_message, write_message


def _construct(engine_id: str, module: str | None):
    if engine_id == "mock-face":
        from unified_live.engines.mock import MockFaceEngine
        return MockFaceEngine()
    if engine_id == "mock-voice":
        from unified_live.engines.mock import MockVoiceEngine
        return MockVoiceEngine()
    if not module:
        raise ValueError("External engine requires a configured bridge module")
    # Imported only in the selected backend's own interpreter, never in the core.
    bridge = importlib.import_module(module)
    return bridge.create_engine()


def main() -> int:
    # Preserve the private protocol descriptor, then redirect Python AND native
    # library stdout writes to stderr before importing any external backend.
    sys.stdout.flush()
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    engine_id = sys.argv[1]
    module = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    engine = None
    while True:
        try:
            meta, array = read_message(sys.stdin.buffer)
        except EOFError:
            return 0
        request_id = meta["id"]
        op = meta.get("op")
        try:
            if op == "init":
                engine = _construct(engine_id, module)
                engine.initialize(meta.get("options", {}))
                result = {"kind": "control", "id": request_id, "ok": True,
                          "capabilities": engine.get_capabilities()}
                write_message(protocol_output, result)
            elif op == "start":
                engine.start()
                write_message(protocol_output, {"kind": "control", "id": request_id, "ok": True})
            elif op == "stop":
                if engine:
                    engine.stop()
                write_message(protocol_output, {"kind": "control", "id": request_id, "ok": True})
                return 0
            elif op == "health":
                write_message(protocol_output, {"kind": "control", "id": request_id, "ok": True, "running": bool(engine and engine.running)})
            elif op == "reset":
                engine.reset()
                write_message(protocol_output, {"kind": "control", "id": request_id, "ok": True})
            elif op == "process":
                if engine is None or array is None:
                    raise RuntimeError("Engine not initialized")
                start = time.monotonic()
                output = engine.process_frame(array) if meta["kind"] == "video" else engine.process_audio(array)
                elapsed = (time.monotonic() - start) * 1000
                engine.last_latency_ms = elapsed
                delay = engine.get_algorithmic_delay_ms() if hasattr(engine, "get_algorithmic_delay_ms") else 0.0
                write_message(protocol_output, {"kind": meta["kind"], "id": request_id, "latency_ms": elapsed,
                              "algorithmic_delay_ms": delay, "warming_up": bool(getattr(engine, "is_warming_up", False)),
                              "processed_faces": getattr(engine, "processed_faces", None)}, output)
            else:
                raise ValueError("Unknown operation")
        except Exception as exc:
            # Diagnostic text is bounded and does not include media or options.
            print(f"worker {engine_id}: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr, flush=True)
            write_message(protocol_output, {"kind": "error", "id": request_id, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})


if __name__ == "__main__":
    raise SystemExit(main())
