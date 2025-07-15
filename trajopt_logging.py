#!/usr/bin/env python3

import logging
import copy


def get_logger(name: str, stdout_level: str = "info", use_color: bool = True) -> logging.Logger:
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  logger.handlers.clear()

  COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",  # Reset
  }

  class ConsoleFormatter(logging.Formatter):
    def __init__(self, *args, use_color: bool = True, **kwargs):
      super().__init__(*args, **kwargs)
      self.use_color = use_color

    def format(self, record):
      record_copy = copy.copy(record)
      original_levelname = record_copy.levelname
      record_copy.levelname = record_copy.levelname.lower()
      if self.use_color:
        color = COLORS.get(original_levelname, COLORS["RESET"])
        record_copy.levelname = f"{color}{record_copy.levelname}{COLORS['RESET']}"
      return super().format(record_copy)

  FORMAT = "[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s"
  DATEFMT = "%H:%M:%S"

  # Log to console
  console_handler = logging.StreamHandler()
  console_handler.setFormatter(ConsoleFormatter(FORMAT, datefmt=DATEFMT, use_color=use_color))
  console_handler.setLevel(getattr(logging, stdout_level.upper(), logging.DEBUG))
  logger.addHandler(console_handler)

  return logger


if __name__ == "__main__":
  logger = get_logger("test")
  logger.info("Hello, world!")
  logger.warning("Warning!")
  logger.error("Error!")
  logger.critical("Critical!")
  logger.debug("Debug!")
