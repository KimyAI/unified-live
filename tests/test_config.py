import pytest

from unified_live.settings import Settings, SettingsStore


def test_settings_atomic_update_and_reject_invalid(tmp_path):
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    store.update({"width": 1280, "voice_enabled": True, "backends": {"seed-vc": {"python": "/venv/python", "root": "/models", "options": {"rate": 48000}}}})
    assert SettingsStore(path).current.width == 1280
    before = path.read_bytes()
    with pytest.raises(ValueError):
        store.update({"fps": 0})
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        store.update({"voice_enabled": 1})
    with pytest.raises(ValueError):
        Settings.from_dict({"version": 2})


def test_settings_reject_unknown_and_nonfinite(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    with pytest.raises(ValueError):
        store.update({"surprise": True})
    with pytest.raises(ValueError):
        store.update({"audio_offset_ms": float("nan")})
