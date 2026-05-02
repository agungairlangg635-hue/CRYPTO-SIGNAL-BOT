"""Logging configuration untuk seluruh aplikasi."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup logger dengan format konsisten untuk console dan file output.
    
    Args:
        name: Logger name (biasanya __name__)
        log_file: Optional path untuk file log
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Hindari duplicate handlers
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger dengan default config."""
    return setup_logger(name, log_file='logs/app.log')