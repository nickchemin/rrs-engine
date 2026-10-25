import os
import shutil
import sys
from pathlib import Path

APP_DIR_NAME = "rrs-engine"


def data_dir() -> Path:
    override = os.environ.get("RRS_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_DIR_NAME


def ensure_data_dir() -> Path:
    path = data_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path() -> Path:
    return data_dir() / "data.db"


def boosters_path() -> Path:
    return data_dir() / "boosters.txt"


def migrate_legacy_files(legacy_root: Path) -> None:
    target = ensure_data_dir()
    for name in ("data.db", "boosters.txt"):
        source = legacy_root / name
        destination = target / name
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
