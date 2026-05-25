from __future__ import annotations

import logging

LOGGER_NAME = "fir_dsp"
logger = logging.getLogger(LOGGER_NAME)


def configure_logging(verbose: bool = False) -> None:
    level = logging.INFO if verbose else logging.WARNING
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    root.setLevel(level)
    logger.setLevel(level)

