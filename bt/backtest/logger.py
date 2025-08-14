import logging
import sys
from typing import Optional
from datetime import datetime


class BacktestLogger:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BacktestLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not BacktestLogger._initialized:
            self._setup_logging()
            BacktestLogger._initialized = True
    
    def _setup_logging(self):
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        root_logger = logging.getLogger('bt.backtest')
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        
        root_logger.propagate = False
    
    def get_logger(self, name: str, level: str = "INFO") -> logging.Logger:
        logger = logging.getLogger(f"bt.backtest.{name}")
        logger.setLevel(getattr(logging, level.upper()))
        return logger
    
    def set_log_level(self, level: str):
        root_logger = logging.getLogger('bt.backtest')
        root_logger.setLevel(getattr(logging, level.upper()))
        
        for handler in root_logger.handlers:
            handler.setLevel(getattr(logging, level.upper()))
    
    def add_file_handler(self, filename: Optional[str] = None, level: str = "DEBUG"):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{timestamp}.log"
        
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(getattr(logging, level.upper()))
        
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        root_logger = logging.getLogger('bt.backtest')
        root_logger.addHandler(file_handler)


_logger_inst = BacktestLogger()


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    return _logger_inst.get_logger(name, level)


def set_log_level(level: str):
    _logger_inst.set_log_level(level)


def enable_file_logging(filename: Optional[str] = None, level: str = "DEBUG"):
    _logger_inst.add_file_handler(filename, level)


ENGINE_LOGGER = get_logger("engine")
STRATEGY_LOGGER = get_logger("strategy") 
EXECUTOR_LOGGER = get_logger("executor")
PERFORMANCE_LOGGER = get_logger("performance")