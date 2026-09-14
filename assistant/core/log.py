"""Lightweight file+console logging so we can see the live pipeline."""
import logging
import os
import sys

from assistant import config

_LOG_PATH = os.path.join(config.DATA_DIR, "jarvis.log")
_configured = False


def setup():
    global _configured
    if _configured:
        return
    # The console is cp1252 on Windows; reconfigure so Telugu/Hindi/emoji in log
    # lines don't raise UnicodeEncodeError on every multilingual transcript.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        fh = logging.FileHandler(_LOG_PATH, mode="w", encoding="utf-8")
        handlers.append(fh)
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-10s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    _configured = True


def get(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(name)
