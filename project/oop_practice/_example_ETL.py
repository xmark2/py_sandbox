import requests
import datetime
from abc import ABC, abstractmethod
from pympler import asizeof
import sys
import logging
import functools
from pathlib import Path
import os
from urllib.parse import urlparse


class Logger:
    def __init__(self, name=__name__, log_file=None, log_level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # Create handlers
        self.file_handler = logging.FileHandler(log_file or Path().cwd().joinpath(f"{name}.log"))
        self.stream_handler = logging.StreamHandler()

        # Define the logging format
        formatter = logging.Formatter(
            fmt="%(levelname)-6s %(name)-15s %(asctime)s.%(msecs)03d %(message)-5s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.file_handler.setFormatter(formatter)
        self.stream_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stream_handler)

    def get_logger(self):
        """Return the configured logger."""
        return self.logger

# Logger Usage
# logger_util = Logger(name="my_logger", log_level=logging.DEBUG)
logger_util = Logger(log_level=logging.DEBUG)
_logger = logger_util.get_logger()


class LogDecorator:
    """Class-based decorator for logging function execution and retrying on failure."""

    def __init__(self, max_retries=1, log_memory_usage=False):
        self.max_retries = max_retries
        self.log_memory_usage = log_memory_usage

    def log_memory(self, state, obj, func_name):
        """Log memory usage."""
        if self.log_memory_usage and obj is not None:
            memory_usage = asizeof.asizeof(obj)
            _logger.info(f"{func_name} {state}: Memory usage: {memory_usage} bytes")

    def handle_exception(self, attempt, error, func_name, args):
        """Handle exceptions during retries."""
        _logger.error(f"Error on attempt {attempt}: {error}")
        if self.log_memory_usage:
            self.log_memory("during error", args, func_name)

    def __call__(self, func):
        """Make the class instance callable as a decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
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

class ETLPipeline:
    def __init__(self):
        # Store dynamic attributes in a dictionary
        self._attributes = {}

    def __getattr__(self, name):
        # Retrieve dynamic attributes
        if name in self._attributes:
            return self._attributes[name]
        raise AttributeError(f"'ETLPipeline' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Allow dynamic attributes for metadata
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            print(f"Setting dynamic attribute: {name} = {value}")
            self._attributes[name] = value

    def __delattr__(self, name):
        # Delete dynamic attributes
        if name in self._attributes:
            print(f"Deleting dynamic attribute: {name}")
            del self._attributes[name]
        else:
            raise AttributeError(f"'ETLPipeline' object has no attribute '{name}'")

#
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



class SourceDescriptor:
    """Descriptor to validate if the source is a valid URL or network file path."""

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        # Retrieve the value from the instance's dictionary
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        # Validate the source value
        if self._is_valid_url(value) or self._is_valid_network_path(value):
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


# --- Descriptor ---
class TransformDescriptor:
    """Descriptor to track transformation metadata."""
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name, None)

    def __set__(self, instance, value):
        _logger.info(f"Setting transformation metadata: {self.name} = {value}")
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        _logger.info(f"Deleting transformation metadata: {self.name}")
        del instance.__dict__[self.name]



# --- Context Manager ---
class FileManager:
    """Context manager for handling file operations."""

    def __init__(self, file_name, mode):
        self.file_name = file_name
        self.mode = mode

    def __enter__(self):
        self.file = open(self.file_name, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()


# --- ETL Pipeline ---
class MemoryOptimizedETL(ETLBase):
    __slots__ = ('source', 'data', 'transformed_data', 'destination')

    validated_source = SourceDescriptor("validated_source")  # Renamed descriptor
    transformation_metadata = TransformDescriptor("transformation_metadata")

    def __init__(self, source, destination):
        self.validated_source = source
        self.data = None
        self.transformed_data = None
        self.destination = destination
        self.transformation_metadata = None

    @LogDecorator(max_retries=3, log_memory_usage=True)
    def extract(self):
        # """Extract data from a remote API."""
        response = requests.get(self.validated_source)
        response.raise_for_status()
        self.data = response.json()
        _logger.info(self.data)
        return self.data

    @LogDecorator(max_retries=1, log_memory_usage=True)
    def transform(self):
        """Transform data."""
        self.transformation_metadata = "Scaling values by 2"
        self.transformed_data = [{**record, "id": record["id"] * 2} for record in self.data]
        # print(f"Data transformed with metadata: {self.transformation_metadata}")

    @LogDecorator(max_retries=1, log_memory_usage=True)
    def load(self):
        """Load data into a file using a context manager."""
        with FileManager(self.destination, "w") as file:
            for record in self.transformed_data:
                file.write(f"{record}\n")

    @LogDecorator(max_retries=1, log_memory_usage=True)
    def execute(self):

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
        destination="output.txt"
    )

    etl.execute()