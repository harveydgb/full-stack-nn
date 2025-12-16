import logging


class Logger:
    """Logging utility for application.
    
    Purpose:
        This class provides a wrapper around Python's logging module
        suitable for training and application logs. 
    
    Attributes:
        logger (logging.Logger): The underlying Python logger instance
    
    Methods:
        __init__: Initialize the logger with a logging level
        info: Log an info message
        warning: Log a warning message
        error: Log an error message
        debug: Log a debug message
    """
    
    def __init__(self, level: int = logging.INFO) -> None:
        """Initialize a Logger instance for application logging.
        
        Parameters:
            level (int): Logging level (default: logging.INFO)
        
        Returns:
            None
        
        Raises:
            ValueError: If logging level is invalid
        """
        try:
            logging.basicConfig(
                level=level,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            self.logger = logging.getLogger("TrainingLogger")
        except Exception as e:
            raise ValueError(f"Failed to initialize logger with level {level}: {e}")

    def info(self, msg: str) -> None:
        """Log an info message.
        
        Parameters:
            msg (str): The message to log
        
        Returns:
            None
        """
        try:
            self.logger.info(msg)
        except Exception as e:
            print(f"Logging error (info): {e}")

    def warning(self, msg: str) -> None:
        """Log a warning message.
        
        Parameters:
            msg (str): The message to log
        
        Returns:
            None
        """
        try:
            self.logger.warning(msg)
        except Exception as e:
            print(f"Logging error (warning): {e}")

    def error(self, msg: str) -> None:
        """Log an error message.
        
        Parameters:
            msg (str): The message to log
        
        Returns:
            None
        """
        try:
            self.logger.error(msg)
        except Exception as e:
            print(f"Logging error (error): {e}")

    def debug(self, msg: str) -> None:
        """Log a debug message.
        
        Parameters:
            msg (str): The message to log
        
        Returns:
            None
        """
        try:
            self.logger.debug(msg)
        except Exception as e:
            print(f"Logging error (debug): {e}")

