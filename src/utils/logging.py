"""
Enhanced logging utilities for OLMo training.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, Union
import json
import time
from datetime import datetime

from olmo_core.distributed.utils import is_distributed, get_rank, get_world_size

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# Format for distributed training - includes rank information
DISTRIBUTED_LOG_FORMAT = "%(asctime)s - [Rank %(rank)s/%(world_size)s] - %(name)s - %(levelname)s - %(message)s"

class DistributedFormatter(logging.Formatter):
    """
    Custom formatter that includes rank information for distributed training.
    """
    
    def __init__(self, fmt=None, datefmt=None):
        """Initialize with default or provided format."""
        if fmt is None:
            fmt = DISTRIBUTED_LOG_FORMAT
        super().__init__(fmt, datefmt)
    
    def format(self, record):
        """Format log record with rank information."""
        if not hasattr(record, 'rank'):
            record.rank = get_rank() if is_distributed() else 0
        if not hasattr(record, 'world_size'):
            record.world_size = get_world_size() if is_distributed() else 1
        return super().format(record)

class RankFilter(logging.Filter):
    """
    Filter that only allows logs from a specific rank or all ranks.
    """
    
    def __init__(self, rank: Optional[int] = None):
        """
        Initialize filter.
        
        Args:
            rank: If provided, only allow logs from this rank. None allows all ranks.
        """
        self.rank = rank
        super().__init__()
    
    def filter(self, record):
        """Filter based on rank."""
        if self.rank is None:
            return True
        current_rank = get_rank() if is_distributed() else 0
        return current_rank == self.rank

def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    main_rank_only: bool = False,
    log_format: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_file: File to save logs to. If None, logs are only sent to console.
        log_level: Logging level.
        main_rank_only: If True, only log from rank 0 in distributed setting.
        log_format: Format for logs. If None, use distributed format if distributed.
        console: Whether to log to console.
        
    Returns:
        logging.Logger: Root logger with configured handlers.
    """
    # Reset root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(log_level)
    
    # Determine if we're in a distributed setting
    distributed = is_distributed()
    current_rank = get_rank() if distributed else 0
    
    # Only log from main rank if specified and we're not the main rank
    if main_rank_only and current_rank != 0:
        return root_logger  # Return without adding handlers
    
    # Select appropriate format
    if log_format is None:
        log_format = DISTRIBUTED_LOG_FORMAT if distributed else DEFAULT_LOG_FORMAT
    
    # Create formatter
    formatter = DistributedFormatter(log_format)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if a log file is specified
    if log_file is not None:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Add timestamp to filename if not already included
        if "%Y" not in log_file and "%m" not in log_file:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename, extension = os.path.splitext(log_file)
            log_file = f"{filename}_{timestamp}{extension}"
        
        # If distributed, add rank to filename
        if distributed:
            filename, extension = os.path.splitext(log_file)
            log_file = f"{filename}_rank{current_rank}{extension}"
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log setup information
    if distributed:
        root_logger.info(f"Logging configured for rank {current_rank}/{get_world_size()}")
    else:
        root_logger.info("Logging configured")
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name for the logger.
        
    Returns:
        logging.Logger: Logger with the specified name.
    """
    return logging.getLogger(name)

def log_distributed(
    message: str,
    level: int = logging.INFO,
    logger: Optional[logging.Logger] = None,
    main_rank_only: bool = False
) -> None:
    """
    Log a message with rank information for distributed training.
    
    Args:
        message: Message to log.
        level: Logging level.
        logger: Logger to use. If None, use root logger.
        main_rank_only: If True, only log from rank 0 in distributed setting.
    """
    # Determine if we're in a distributed setting
    distributed = is_distributed()
    current_rank = get_rank() if distributed else 0
    
    # Only log from main rank if specified and we're not the main rank
    if main_rank_only and distributed and current_rank != 0:
        return
    
    # Get logger
    if logger is None:
        logger = logging.getLogger()
    
    # Log message
    if distributed:
        message = f"[Rank {current_rank}/{get_world_size()}] {message}"
    
    logger.log(level, message)

def log_dict(
    data: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    prefix: str = "",
    main_rank_only: bool = True
) -> None:
    """
    Log a dictionary in a nicely formatted way.
    
    Args:
        data: Dictionary to log.
        logger: Logger to use. If None, use root logger.
        level: Logging level.
        prefix: Prefix for the log message.
        main_rank_only: If True, only log from rank 0 in distributed setting.
    """
    # Only log from main rank if specified and we're not the main rank
    if main_rank_only and is_distributed() and get_rank() != 0:
        return
    
    # Get logger
    if logger is None:
        logger = logging.getLogger()
    
    # Convert to JSON string with indentation
    formatted_data = json.dumps(data, indent=2, default=str)
    
    # Add prefix if provided
    if prefix:
        message = f"{prefix}\n{formatted_data}"
    else:
        message = formatted_data
    
    logger.log(level, message)

def log_training_stats(
    stats: Dict[str, Any],
    step: int,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    main_rank_only: bool = True
) -> None:
    """
    Log training statistics in a consistent format.
    
    Args:
        stats: Dictionary of training statistics.
        step: Current training step.
        logger: Logger to use. If None, use root logger.
        level: Logging level.
        main_rank_only: If True, only log from rank 0 in distributed setting.
    """
    # Only log from main rank if specified and we're not the main rank
    if main_rank_only and is_distributed() and get_rank() != 0:
        return
    
    # Get logger
    if logger is None:
        logger = logging.getLogger()
    
    # Format message
    message = f"Step {step} | "
    message += " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()])
    
    logger.log(level, message)