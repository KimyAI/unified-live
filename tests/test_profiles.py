import pytest

from unified_live.profiles import ProfileStore
from unified_live.settings import Settings


def test_profile_roundtrip_all_settings(tmp_path):
    store = ProfileStore(tmp_path / "profiles")
    settings = Settings.from_dict({**Settings().to_dict(), "face_enabled": True, "camera_index": 2,
                                   "voice_options": {"source": "/voice/reference.wav", "model": "/models/v.pth"},
                                   "backends": {"seed-vc": {"python": "/venv/python", "module": "bridge"}}})
    store.save("concert-1", settings)
    assert store.list() == ["concert-1"]
    assert store.load("concert-1") == settings


@pytest.mark.parametrize("name", ["../outside", "/tmp/foo", "a/b", "..", "name.json"])
def test_profile_name_rejects_traversal(tmp_path, name):
    store = ProfileStore(tmp_path / "profiles")
    with pytest.raises(ValueError):
        store.save(name, Settings())


def test_profile_load_rejects_symlink(tmp_path):
    store = ProfileStore(tmp_path / "profiles")
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (store.directory / "linked.json").symlink_to(outside)
    assert "linked" not in store.list()
    with pytest.raises(ValueError):
        store.load("linked")
def test_human_names_and_windows_reserved_names(tmp_path):
    from unified_live.profiles import ProfileStore
    from unified_live.settings import Settings
    import pytest

    profiles = ProfileStore(tmp_path)
    profiles.save("Character A", Settings())
    assert "Character A" in profiles.list()
    assert profiles.load("Character A").app_name == "Unified Live"
    for name in ("CON", "aux", "LPT1", "../../escape", "trailing "):
        with pytest.raises(ValueError):
            profiles.save(name, Settings())
