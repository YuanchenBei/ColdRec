import copy
import logging
import logging.config
import os
import sys
import time

import torch
from rich.text import Text
from rich.traceback import install as install_rich_tracebacks

# ==============================================================================
# CONFIGURATION
# ==============================================================================


# Note the special '()' key. It tells dictConfig to import and instantiate
# this class. The path must be importable from where the app runs.
# Assuming this file is in 'utils/logger.py', the path is correct.
class RemoveRichMarkupFilter(logging.Filter):
    """A logging filter to remove rich markup from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = Text.from_markup(record.getMessage()).plain
        record.args = ()
        return True


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "remove_markup": {
            "()": __name__ + ".RemoveRichMarkupFilter",
        }
    },
    "formatters": {
        "plain_text": {
            "format": "%(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "rich.logging.RichHandler",
            "level": "INFO",
            "rich_tracebacks": True,
            "markup": True,
            "show_path": False,
            "show_level": False,
            "show_time": False,
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "filename": "result.log",
            "mode": "a",
            "encoding": "utf-8",
            "formatter": "plain_text",
            "filters": ["remove_markup"],
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    },
}


# ==============================================================================
# SETUP LOGIC
# ==============================================================================

_logger_initialized = False


def setup_logging(
    dataset: str,
    model: str,
    no_log: bool = False,
    log_dir: str = "./log",
    console_log_level: int = logging.INFO,
    file_log_level: int = logging.INFO,
):
    """
    Configures the logging system using a dictionary configuration.
    This should be called once at the start of the application.

    Returns:
        The path to the log subdirectory if successful, otherwise None.
    """
    global _logger_initialized
    if _logger_initialized:
        logging.warning("Logger has already been initialized.")
        return None

    if no_log:
        logging.disable(logging.CRITICAL + 1)
        return None

    try:
        config = copy.deepcopy(LOGGING_CONFIG)

        timestamp = time.strftime("%Y-%m%d-%H%M", time.localtime())
        log_subdir = os.path.join(log_dir, dataset, model, timestamp)
        os.makedirs(log_subdir, exist_ok=True)
        log_filename = os.path.join(log_subdir, "result.log")

        config["handlers"]["file"]["filename"] = log_filename
        config["handlers"]["console"]["level"] = logging.getLevelName(console_log_level)
        config["handlers"]["file"]["level"] = logging.getLevelName(file_log_level)

        logging.config.dictConfig(config)

        # Actions that are not part of dictConfig (like global hooks)
        install_rich_tracebacks(show_locals=True)
        sys.excepthook = handle_exception

        _logger_initialized = True
        logging.info(f"Logger initialized. Log file at: {log_filename}")
        return log_subdir

    except Exception as e:
        print(f"Failed to initialize logger: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        return None


def handle_exception(exc_type, exc_value, exc_traceback):
    """A global exception hook to log unhandled exceptions before exiting."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    if _logger_initialized:
        logging.critical("Unhandled exception occurred:", exc_info=(exc_type, exc_value, exc_traceback))
    else:
        print("Unhandled exception occurred (logger not initialized):", file=sys.stderr)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


# ==============================================================================
# TENSORBOARD UTILITY FUNCTION
# ==============================================================================


def log_training_details_to_tensorboard(writer, step, data_dict):
    """Logs metrics to a TensorBoard SummaryWriter instance."""
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                writer.add_scalar(key, value.item(), step)
            else:
                writer.add_histogram(key, value, step)
        elif isinstance(value, (float, int)):
            writer.add_scalar(key, value, step)
