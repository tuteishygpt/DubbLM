"""Logging configuration for the Smart Dubbing system."""

import logging
import sys
import os
from datetime import datetime


_NOISY_PREFIXES: tuple[str, ...] = (
    "numba",
    "matplotlib",
    "PIL",
    "fsspec",
    "urllib3",
    "asyncio",
    "gradio",
    "gradio_client",
    "httpx",
    "httpcore",
    "hpack",
    "librosa",
    "resampy",
    "audioread",
    "soundfile",
    "filelock",
    "speechbrain",
)


class NoisyPrefixFilter(logging.Filter):
    """Drop DEBUG/INFO records emitted by known-noisy third-party loggers."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return not record.name.startswith(_NOISY_PREFIXES)

def setup_logging(level=logging.INFO):
    """
    Set up logging for the application.
    
    Args:
        level: The logging level to use for console output (e.g., logging.INFO, logging.DEBUG)
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a filename with the current date and time
    log_filename = datetime.now().strftime(os.path.join(log_dir, 'dubbing_%Y-%m-%d_%H-%M-%S.log'))

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Set root logger to capture all levels

    # Clear existing handlers to avoid duplicate logging
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Set logging level for noisy libraries to reduce verbosity
    for noisy in (
        "httpx",
        "httpcore",
        "http",
        "urllib3",
        "asyncio",
        "google_genai.models",
        "speechbrain.utils.fetching",
        "speechbrain.utils.parameter_transfer",
        "speechbrain.utils.checkpoints",
        "matplotlib",
        "matplotlib.font_manager",
        "matplotlib.pyplot",
        "matplotlib.backends",
        "matplotlib.backends.backend_tkagg",
        "matplotlib.backend_bases",
        "PIL",
        "PIL.PngImagePlugin",
        "PIL.Image",
        "fsspec",
        "fsspec.local",
        "hpack",
        "gradio",
        "gradio.processing_utils",
        "gradio_client",
        "filelock",
        "numba",
        "numba.core",
        "numba.core.ssa",
        "numba.core.byteflow",
        "numba.core.interpreter",
        "numba.core.typeinfer",
        "numba.core.compiler",
        "numba.core.rewrites",
        "numba.core.lowering",
        "numba.core.bytecode",
        "numba.core.controlflow",
        "librosa",
        "resampy",
        "soundfile",
        "audioread",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Apply noisy-prefix filter directly on the root logger so any handler
    # added later (e.g. the streaming capture handler in runner.py) inherits
    # the suppression automatically.
    root_logger.addFilter(NoisyPrefixFilter())

    # Console Handler (prints INFO and above to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.addFilter(NoisyPrefixFilter())
    # Use a simpler format for the console
    console_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File Handler (prints DEBUG and above to a file)
    file_handler = logging.FileHandler(log_filename, 'a', 'utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(NoisyPrefixFilter())
    # Use a more detailed format for the file
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: The name of the logger (usually __name__)
        
    Returns:
        A logger instance
    """
    return logging.getLogger(name) 