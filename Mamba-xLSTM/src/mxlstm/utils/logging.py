"""Centralized logger via loguru with rich formatting."""

from __future__ import annotations

import sys

from loguru import logger as _logger

_logger.remove()
_logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> <cyan>{name}</cyan> | {message}",
    level="INFO",
)

logger = _logger
