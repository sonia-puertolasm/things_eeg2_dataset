import logging
import sys
from typing import ClassVar


class CustomFormatter(logging.Formatter):
    """
    Formatter that handles:
    1. 'Bare' logging for headers (no timestamp/level).
    2. Colorized output for standard logs depending on severity.
    3. Custom format with line number attached to the logger name.
    """

    default_fmt = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    bare_fmt = "%(message)s"

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: grey + default_fmt + reset,
        logging.INFO: grey + default_fmt + reset,
        logging.WARNING: yellow + default_fmt + reset,
        logging.ERROR: red + default_fmt + reset,
        logging.CRITICAL: bold_red + default_fmt + reset,
    }

    def __init__(self) -> None:
        super().__init__()
        self.bare_formatter = logging.Formatter(self.bare_fmt)

        date_format = "%H:%M:%S"
        self.formatters = {
            level: logging.Formatter(fmt, datefmt=date_format)
            for level, fmt in self.FORMATS.items()
        }

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "bare", False):
            return self.bare_formatter.format(record)

        formatter = self.formatters.get(record.levelno, self.formatters[logging.INFO])

        return formatter.format(record)


def setup_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())
    root_logger.addHandler(ch)
