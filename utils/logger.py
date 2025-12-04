import logging
import sys
from pathlib import Path
from typing import Optional
from config import LOG_FILE


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Get or create a logger with both file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)
        logger.propagate = False
        
        # File formatter (no colors)
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console formatter (with colors)
        console_formatter = ColoredFormatter(
            "[%(asctime)s] [%(levelname)s] - %(message)s",
            datefmt="%H:%M:%S"
        )
        
        # File handler
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def log_section(logger: logging.Logger, title: str, width: int = 70):
    """Log a section header"""
    logger.info("[logger.py] =" * width)
    logger.info(title.center(width))
    logger.info("[logger.py] =" * width)


def log_subsection(logger: logging.Logger, title: str, width: int = 70):
    """Log a subsection header"""
    logger.info("[logger.py] -" * width)
    logger.info(title)
    logger.info("[logger.py] -" * width)