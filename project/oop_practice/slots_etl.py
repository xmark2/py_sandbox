class ETLStage:
    __slots__ = ('source', 'data', 'transformed_data', 'destination')  # Restrict attributes

    def __init__(self, source=None, destination=None):
        self.source = source
        self.data = None
        self.transformed_data = None
        self.destination = destination

    def extract(self):
        print(f"Extracting data from: {self.source}")
        self.data = [{"id": 1, "value": 100}, {"id": 2, "value": 200}]  # Mock data
        print(f"Data extracted: {self.data}")

    def transform(self):
        print("Transforming data...")
        self.transformed_data = [{"id": item["id"], "value": item["value"] * 2} for item in self.data]
        print(f"Data transformed: {self.transformed_data}")

    def load(self):
        print(f"Loading data to: {self.destination}")
        print(f"Data loaded: {self.transformed_data}")

if __name__ == "__main__":
    # Example usage of the ETL pipeline with `__slots__`
    pipeline = ETLStage(source="Database", destination="Data Warehouse")
    pipeline.extract()
    pipeline.transform()
    pipeline.load()

    # Attempting to add an undefined attribute will raise an AttributeError
    try:
        pipeline.new_attr = "This will fail"
    except AttributeError as e:
        print(f"Error: {e}")
