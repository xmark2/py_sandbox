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

    def extract(self, source):
        # Simulate data extraction
        self.source = source  # Set dynamic metadata
        self.data = [{"id": 1, "value": 100}, {"id": 2, "value": 200}]  # Mock data
        print(f"Data extracted from: {self.source}")

    def transform(self):
        # Simulate data transformation
        self.transformation = "Scaling values"  # Set dynamic metadata
        self.data = [{"id": item["id"], "value": item["value"] * 2} for item in self.data]
        print(f"Data transformed using: {self.transformation}")

    def load(self, destination):
        # Simulate data loading
        self.destination = destination  # Set dynamic metadata
        print(f"Data loaded to: {self.destination}")


if __name__ == "__main__":
    # Using the ETL pipeline with dynamic attributes
    pipeline = ETLPipeline()

    # Extract data
    pipeline.extract(source="Database")

    # Access dynamic attributes
    print(f"Source used: {pipeline.source}")

    # Transform data
    pipeline.transform()

    # Access dynamic attributes
    print(f"Transformation applied: {pipeline.transformation}")

    # Load data
    pipeline.load(destination="Analytics Dashboard")

    # Access dynamic attributes
    print(f"Destination: {pipeline.destination}")
