"""Local operational logs, never raw frames or samples."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(data_dir: Path):
    directory = data_dir / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    for name, filename in (
        ("unified_live", "core.log"),
        ("unified_live.video", "video-engine.log"),
        ("unified_live.audio", "voice-engine.log"),
        ("unified_live.training", "training.log"),
    ):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        for old in list(logger.handlers):
            logger.removeHandler(old)
            old.close()
        handler = RotatingFileHandler(directory / filename, maxBytes=2_000_000, backupCount=3)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
