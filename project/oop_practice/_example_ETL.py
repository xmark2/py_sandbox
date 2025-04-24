import requests
import datetime
from abc import ABC, abstractmethod
# from functools import wraps

# from joblib.testing import raises
from pympler import asizeof
import sys
import logging
import functools
from pathlib import Path
import os
from urllib.parse import urlparse

_logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(levelname)-6s %(name)-15s %(asctime)s.%(msecs)03d %(message)-5s',
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(Path().cwd().joinpath(f'{__name__}.log')),
        logging.StreamHandler()
    ]
)

_logger = logging.getLogger(__name__)

def log_decorator(max_retries=1, log_memory_usage=True):
    """Decorator to log function execution, retry on failure, and measure memory usage."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def log_memory(state, obj):
                """Log memory usage."""
                if log_memory_usage:
                    memory_usage = asizeof.asizeof(obj)
                    _logger.info(f"{func.__name__} {state}: Memory usage: {memory_usage} bytes")

            def handle_exception(attempt, error):
                """Handle exceptions during retries."""
                _logger.error(f"Error on attempt {attempt}: {error}")
                if log_memory_usage:
                    log_memory("during error", args)

            for attempt in range(max_retries):
                _logger.info(f"Attempt {attempt + 1}/{max_retries}: {func.__name__} started")
                log_memory("inputs", args)

                try:
                    result = func(*args, **kwargs)
                    log_memory("completed", result)
                    _logger.info(f"{func.__name__} completed successfully.")
                    return result

                except requests.RequestException as e:
                    handle_exception(attempt + 1, e)
                    if attempt == max_retries - 1:
                        _logger.error("Function failed after maximum retries.")
                        raise
                    _logger.info("Retrying...")

            return None

        return wrapper

    return decorator


# --- Abstract Base Class ---
class ETLProcess(ABC):
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
#
#
# class MetadataDescriptor:
#     def __init__(self, name):
#         self.name = name
#
#     def __get__(self, instance, owner):
#         return instance.__dict__.get("metadata", None)
#
#     def __set__(self, instance, value):
#         instance.__dict__["metadata"] = f"{value} at {datetime.datetime.now()}"
#         print(f"Metadata updated: {instance.metadata}")
#
#     def __delete__(self, instance):
#         print(f"Deleting transformation metadata: {self.name}")
#         del instance.__dict__[self.name]


# # --- Descriptor ---
# class TransformDescriptor:
#     """Descriptor to track transformation metadata."""
#     def __init__(self, name):
#         self.name = name
#
#     def __get__(self, instance, owner):
#         return instance.__dict__.get(self.name, None)
#
#     def __set__(self, instance, value):
#         print(f"Setting transformation metadata: {self.name} = {value}")
#         instance.__dict__[self.name] = value
#
#     def __delete__(self, instance):
#         print(f"Deleting transformation metadata: {self.name}")
#         del instance.__dict__[self.name]

class SourceValidator:
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
class MemoryOptimizedETL(ETLProcess):
    __slots__ = ('source', 'data', 'transformed_data', 'destination', 'transformation_metadata')

    validated_source = SourceValidator("validated_source")  # Renamed descriptor

    # transformation_metadata = TransformDescriptor("transformation_metadata")
    # source = SourceValidator("source")  # Use the descriptor for the source

    def __init__(self, source, destination):
        self.validated_source = source
        self.data = None
        self.transformed_data = None
        self.destination = destination
        self.transformation_metadata = None

    # @log_decorator
    @log_decorator(max_retries=3)
    def extract(self):
        # try:
        # """Extract data from a remote API."""
        # _logger.info(f"Extracting data from source: {self.validated_source}")
        response = requests.get(self.validated_source)
        # if not response:
        response.raise_for_status()
            # _logger.error(error_msg)  # Raise exception for invalid responses
        self.data = response.json()
        _logger.info(self.data)
        # print("Data extracted successfully.")
        return self.data
        # except requests.exceptions as http_err:
            # Log specific HTTP error
            # _logger.error(f"HTTP error occurred: {http_err}")  # Captures the error message

    @log_decorator()
    def transform(self):
        """Transform data."""
        # print("Transforming data...")
        self.transformation_metadata = "Scaling values by 2"
        self.transformed_data = [{**record, "id": record["id"] * 2} for record in self.data]
        # print(f"Data transformed with metadata: {self.transformation_metadata}")

    @log_decorator()
    def load(self):
        """Load data into a file using a context manager."""
        # print(f"Loading data to destination: {self.destination}")
        with FileManager(self.destination, "w") as file:
            for record in self.transformed_data:
                file.write(f"{record}\n")
        # print("Data loaded successfully.")

    @log_decorator()
    def execute(self):

        # _logger.info(f"Memory usage of ETL instance (pympler): {asizeof.asizeof(self)} bytes")
        # Extract stage
        _data = self.extract()

        # print(f"Memory usage of ETL instance (pympler): {asizeof.asizeof(self)} bytes")
        # Transform stage
        if not _data:
            return
        self.transform()

        # print(f"Memory usage of ETL instance (pympler): {asizeof.asizeof(self)} bytes")
        # Load stage
        self.load()

        # print(f"Memory usage of ETL instance (pympler): {asizeof.asizeof(self)} bytes")


# --- Dynamic Attributes ---
def add_dynamic_metadata(etl_instance, name, value):
    """Add dynamic attributes for metadata tracking."""
    setattr(etl_instance, name, value)
    # print(f"Dynamic attribute added: {name} = {value}")


# --- Example Usage ---
if __name__ == "__main__":
    etl = MemoryOptimizedETL(
        # source="",
        source="https://jsonplaceholder.typicode.com/posts",  # Mock API source
        destination="output.txt"
    )

    etl.execute()

    # # Extract stage
    # data = etl.extract()
    #
    # # Transform stage
    # if not data:
    #     return
    # etl.transform()
    #
    # # Add dynamic metadata
    # # add_dynamic_metadata(etl, "pipeline_status", "Transformation complete")
    #
    # # Load stage
    # etl.load()

    # # Access dynamic and descriptor attributes
    # print(f"Transformation Metadata: {etl.transformation_metadata}")
    # print(f"Pipeline Status (Dynamic Attribute): {etl.pipeline_status}")
    #
    # Check memory usage
    # print(f"Memory usage of ETL instance (sys): {sys.getsizeof(etl)} bytes")
    # print(f"Memory usage of ETL instance (pympler): {asizeof.asizeof(etl)} bytes")
