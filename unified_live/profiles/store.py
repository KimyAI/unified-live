from __future__ import annotations

import re
from pathlib import Path

from unified_live.settings import Settings
from unified_live.settings.store import atomic_json, read_json


class ProfileStore:
    """Complete settings snapshots. Names are identifiers, never filesystem paths."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        if not self._valid_name(name):
            raise ValueError("Profile name must use letters, digits, spaces, _ or - (no reserved filenames)")
        return self.directory / f"{name}.json"

    @staticmethod
    def _valid_name(name: str) -> bool:
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
        return (isinstance(name, str) and name == name.strip() and
                bool(re.fullmatch(r"\w[\w -]{0,63}", name)) and name.upper() not in reserved)

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json") if not p.is_symlink() and self._valid_name(p.stem))

    def save(self, name: str, settings: Settings) -> None:
        settings.validate()
        atomic_json(self._path(name), {"profile_version": 1, "name": name, "settings": settings.to_dict()})

    def load(self, name: str) -> Settings:
        path = self._path(name)
        if path.is_symlink():
            raise ValueError("Profile symlinks are not allowed")
        value = read_json(path)
        if value.get("profile_version") != 1 or value.get("name") != name:
            raise ValueError("Invalid profile")
        if not isinstance(value.get("settings"), dict):
            raise ValueError("Invalid profile settings")
        return Settings.from_dict(value["settings"])
