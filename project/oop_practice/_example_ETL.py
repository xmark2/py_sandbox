import requests
import datetime
from abc import ABC, abstractmethod
from functools import wraps
from pympler import asizeof
import sys

# --- Retry Decorator ---
def retry_on_failure(max_retries=3):
    """Decorator to retry a function on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    print(f"Attempt {attempt + 1}...")
                    result = func(*args, **kwargs)
                    if result:
                        return result
                except requests.RequestException as e:
                    print(f"Error: {e}, retrying...")
            print("Failed after maximum retries.")
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


# Descriptor for positive value
class PositiveValueDescriptor:
    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        if value < 0:
            raise ValueError(f"{self.name} must be non-negative!")
        instance.__dict__[self.name] = value

    def __set_name__(self, owner, name):
        self.name = name


class MetadataDescriptor:
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__.get("metadata", None)

    def __set__(self, instance, value):
        instance.__dict__["metadata"] = f"{value} at {datetime.datetime.now()}"
        print(f"Metadata updated: {instance.metadata}")

    def __delete__(self, instance):
        print(f"Deleting transformation metadata: {self.name}")
        del instance.__dict__[self.name]


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
    __slots__ = ('source', 'data', 'transformed_data', 'destination', 'transformation_metadata')  # Slots for memory optimization

    # transformation_metadata = TransformDescriptor("transformation_metadata")

    def __init__(self, source, destination):
        self.source = source
        self.data = None
        self.transformed_data = None
        self.destination = destination
        self.transformation_metadata = None

    @retry_on_failure(max_retries=3)
    def extract(self):
        """Extract data from a remote API."""
        print(f"Extracting data from source: {self.source}")
        response = requests.get(self.source)
        response.raise_for_status()  # Raise exception for invalid responses
        self.data = response.json()
        print(self.data)
        print("Data extracted successfully.")
        return self.data

    def transform(self):
        """Transform data."""
        print("Transforming data...")
        self.transformation_metadata = "Scaling values by 2"
        self.transformed_data = [{**record, "id": record["id"] * 2} for record in self.data]
        print(f"Data transformed with metadata: {self.transformation_metadata}")

    def load(self):
        """Load data into a file using a context manager."""
        print(f"Loading data to destination: {self.destination}")
        with FileManager(self.destination, "w") as file:
            for record in self.transformed_data:
                file.write(f"{record}\n")
        print("Data loaded successfully.")

# --- Dynamic Attributes ---
def add_dynamic_metadata(etl_instance, name, value):
    """Add dynamic attributes for metadata tracking."""
    setattr(etl_instance, name, value)
    print(f"Dynamic attribute added: {name} = {value}")

# --- Example Usage ---
if __name__ == "__main__":
    etl = MemoryOptimizedETL(
        source="https://jsonplaceholder.typicode.com/posts",  # Mock API source
        destination="output.txt"
    )

    # Extract stage
    data = etl.extract()

    # Transform stage
    if data:
        etl.transform()

    # Add dynamic metadata
    # add_dynamic_metadata(etl, "pipeline_status", "Transformation complete")

    # Load stage
    etl.load()

    # # Access dynamic and descriptor attributes
    # print(f"Transformation Metadata: {etl.transformation_metadata}")
    # print(f"Pipeline Status (Dynamic Attribute): {etl.pipeline_status}")
    #
    # # Check memory usage
    # print(f"Memory usage of ETL instance (sys): {sys.getsizeof(etl)} bytes")
    # print(f"Memory usage of ETL instance (pympler): {asizeof.asizeof(etl)} bytes")