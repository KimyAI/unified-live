from unified_live.engines import EngineRegistry


def test_planned_engines_are_not_claimed_as_validated():
    registry = EngineRegistry()
    for entry in registry.list():
        if entry["planned"]:
            assert not entry["installed"]
            assert not entry["validated"]
    configured = EngineRegistry({"reswapper": {"python": "/some/python", "module": "bridge"}}).get("reswapper")
    assert configured.configured and not configured.installed
    assert not configured.planned and not configured.validated
    assert registry.get("rvc").planned


def test_mock_capabilities_are_honest():
    registry = EngineRegistry()
    assert registry.get("mock-face").capabilities["face_swap"] is False
    assert registry.get("mock-voice").capabilities["voice_conversion"] is False
