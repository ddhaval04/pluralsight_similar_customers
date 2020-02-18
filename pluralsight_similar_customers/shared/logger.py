import logging
import os
import time

_loggers = {}
LOG_DIR = "./logs/"


def get_logger(name):
    global _loggers
    if _loggers.get(name):
        return _loggers[name]
    else:
        logger = logging.getLogger(name)
        formatter = logging.Formatter(
            "%(asctime)s [%(process)d] [%(levelname)s] %(name)-16s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        screen_handler = logging.StreamHandler()
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)
        _loggers[name] = logger

        return _loggers[name]
