import requests
import datetime
from abc import ABC, abstractmethod
from pympler import asizeof
import sys
import logging
import functools
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Union, Type, Callable
from types import TracebackType
from urllib.parse import urlparse


class Logger:
    def __init__(self,
                    name: str = __name__,
                    log_file: Optional[Union[str, Path]] = None,
                    log_level: int = logging.INFO) -> None:
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # Create handlers
        self.file_handler: logging.FileHandler = logging.FileHandler(log_file or Path().cwd().joinpath(f"{name}.log"))
        self.stream_handler: logging.StreamHandler = logging.StreamHandler()

        # Define the logging format
        formatter: logging.Formatter = logging.Formatter(
            fmt="%(levelname)-6s %(name)-15s %(asctime)s.%(msecs)03d %(message)-5s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.file_handler.setFormatter(formatter)
        self.stream_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stream_handler)

    def get_logger(self) -> logging.Logger:
        """Return the configured logger."""
        return self.logger

# Logger Usage
# logger_util = Logger(name="my_logger", log_level=logging.DEBUG)
logger_util = Logger(log_level=logging.DEBUG)
_logger = logger_util.get_logger()


class LogDecorator:
    """Class-based decorator for logging function execution and retrying on failure."""

    def __init__(self, max_retries: int = 1, log_memory_usage: bool = False) -> None:
        """
        Initialize the LogDecorator.

        Args:
            max_retries (int): Number of retries for the decorated function.
            log_memory_usage (bool): Whether to log memory usage of inputs and outputs.
        """
        self.max_retries: int = max_retries
        self.log_memory_usage: bool = log_memory_usage

    def log_memory(self, state: str, obj: Optional[Any], func_name: str) -> None:
        """
        Log memory usage for the given object.
        """
        if self.log_memory_usage and obj is not None:
            memory_usage: int = asizeof.asizeof(obj)
            _logger.info(f"{func_name} {state}: Memory usage: {memory_usage} bytes")

    def handle_exception(self, attempt: int, error: Exception, func_name: str, args: tuple) -> None:
        """
        Handle exceptions during retries.
        """
        _logger.error(f"Error on attempt {attempt}: {error}")
        if self.log_memory_usage:
            self.log_memory("during error", args, func_name)

    def __call__(self, func: Callable) -> Callable:
        """
        Make the class instance callable as a decorator.

        Returns:
            Callable: The wrapped function with logging and retry logic.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[Any]:
            """
            Wrapper function to execute the decorated function with logging and retries.

            Returns:
                Optional[Any]: The result of the decorated function, or None if it fails.
            """
            for attempt in range(self.max_retries):
                if self.max_retries > 1:
                    _logger.info(f"Attempt {attempt + 1}/{self.max_retries}: {func.__name__} started")
                else:
                    _logger.info(f"{func.__name__} started")

                self.log_memory("inputs", args, func.__name__)
                try:
                    result = func(*args, **kwargs)  # Execute the decorated function
                    self.log_memory("completed", result, func.__name__)
                    _logger.info(f"{func.__name__} completed successfully.")
                    return result
                except requests.RequestException as e:
                    self.handle_exception(attempt + 1, e, func.__name__, args)
                    if attempt == self.max_retries - 1:  # If last attempt fails, raise the exception
                        _logger.error(f"{func.__name__} failed after maximum retries.")
                        raise
                    _logger.info(f"Retrying {func.__name__}...")
            return None

        return wrapper


# --- Abstract Base Class ---
class ETLBase(ABC):
    """Abstract base class for ETL processes."""

    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def execute(self):
        pass



# # Descriptor for positive value
# class PositiveValueDescriptor:
#     def __get__(self, instance, owner):
#         return instance.__dict__.get(self.name)
#
#     def __set__(self, instance, value):
#         if value < 0:
#             raise ValueError(f"{self.name} must be non-negative!")
#         instance.__dict__[self.name] = value
#
#     def __set_name__(self, owner, name):
#         self.name = name


class DescriptorSource:
    """Descriptor to validate if the source is a valid URL or network file path."""

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        # Retrieve the value from the instance's dictionary
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        # Validate the source value
        if (
                self._is_valid_url(value)
                or self._is_valid_network_path(value)
        ):
            instance.__dict__[self.name] = value
        else:
            raise ValueError(f"Invalid source: {value}. Must be a valid URL or network path.")

    def __delete__(self, instance):
        raise AttributeError(f"Cannot delete the '{self.name}' attribute.")

    def _is_valid_url(self, value):
        """Validate if the source is a URL."""
        try:
            parsed = urlparse(value)
            return all([parsed.scheme, parsed.netloc])
        except Exception:
            return False

    def _is_valid_network_path(self, value):
        """Validate if the source is a network path."""
        # Assuming a valid network path starts with '\\' or '/' for UNIX systems
        return os.path.isabs(value) or value.startswith("\\\\")


class DescriptorFileDestination:
    """Descriptor to validate if the file destination is valid."""

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name, None)

    def __set__(self, instance, value):
        # Validate the destination file path
        if not self._is_valid_file_destination(value):
            raise ValueError(f"Invalid file destination: {value}. Ensure the path exists and is writable.")
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        raise AttributeError(f"Cannot delete the '{self.name}' attribute.")

    def _is_valid_file_destination(self, file_path):
        """Validate if the destination file path is valid."""
        try:
            path = Path(file_path)
            # Check if the path is absolute and writable
            if not path.is_absolute():
                return False
            if not path.parent.exists():
                return False
            # Test if the file can be created (if it doesn't already exist)
            if not os.access(path.parent, os.W_OK):
                return False
            return True
        except Exception:
            return False

# --- Context Manager ---
class FileManager:
    """Context manager for handling file operations."""

    def __init__(self, file_name: str, mode: str) -> None:
        self.file_name: str = file_name  # File name to be opened
        self.mode: str = mode  # Mode for opening the file (e.g., 'r', 'w')
        self.file: Optional[object] = None  # Placeholder for the file object

    def __enter__(self) -> object:
        self.file = open(self.file_name, self.mode)
        return self.file

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        if self.file:
            self.file.close()


# --- ETL Pipeline ---
class MemoryOptimizedETL(ETLBase):
    __slots__ = ('source', 'data', 'transformed_data', 'destination')

    validated_source = DescriptorSource("validated_source")  # Renamed descriptor
    validated_destination = DescriptorFileDestination("validated_destination")

    def __init__(self, source: str, destination: Union[str, Path]) -> None:
        self.validated_source: str = source
        self.data: Optional[List[Dict[str, Any]]] = None  # Data is a list of dictionaries
        self.transformed_data: Optional[List[Dict[str, Any]]] = None  # Transformed data (optional)
        self.validated_destination: str = str(destination)  # Always store destination as a string

    # Extract method
    @LogDecorator(max_retries=3, log_memory_usage=True)
    def extract(self) -> List[Dict[str, Any]]:
        """Extract data from a remote API."""
        response = requests.get(self.validated_source)
        response.raise_for_status()
        self.data = response.json()
        _logger.info(self.data)
        return self.data

    @LogDecorator(max_retries=1, log_memory_usage=True)
    def transform(self) -> None:
        """Transform data."""
        self.transformed_data = [{**record, "id": record["id"] * 2} for record in self.data]

    @LogDecorator(max_retries=1, log_memory_usage=True)
    def load(self) -> None:
        """Load data into a file using a context manager."""
        with FileManager(self.validated_destination, "w") as file:
            for record in self.transformed_data:
                file.write(f"{record}\n")

    @LogDecorator(max_retries=1, log_memory_usage=True)
    def execute(self) -> None:

        # Extract stage
        _data = self.extract()

        # Transform stage
        if not _data:
            return
        self.transform()

        # Load stage
        self.load()


# --- Example Usage ---
if __name__ == "__main__":
    etl = MemoryOptimizedETL(
        # source="",
        source="https://jsonplaceholder.typicode.com/posts",  # Mock API source
        # destination="https://jsonplaceholder.typicode.com/posts"  # Mock API source
        destination=str(Path().cwd() / "output.txt")
    )

    etl.execute()
