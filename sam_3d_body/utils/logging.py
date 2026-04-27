# Copyright (c) Meta Platforms, Inc. and affiliates.
import functools
import logging
import os


def rank_zero_only(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if os.environ.get("RANK", "0") == "0":
            return fn(*args, **kwargs)

    return wrapped


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
