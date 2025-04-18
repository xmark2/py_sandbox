from abc import ABC, abstractmethod

# Abstract Base Class for the ETL Process
class ETL(ABC):
    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    @abstractmethod
    def load(self, transformed_data):
        pass

# Concrete ETL Class Implementation
class CSVToDatabaseETL(ETL):
    def extract(self):
        print("Extracting data from CSV...")
        # Mock data from a CSV file
        data = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25}
        ]
        return data

    def transform(self, data):
        print("Transforming data...")
        # Simple transformation: Capitalize names and filter by age
        transformed = [
            {**record, "name": record["name"].upper()}
            for record in data if record["age"] > 26
        ]
        return transformed

    def load(self, transformed_data):
        print("Loading data to the database...")
        for record in transformed_data:
            print(f"Inserted record into database: {record}")

# Descriptor for ETL stages
class ETLStage:
    def __init__(self, name):
        print(f'initialize {name}')
        self.name = name

    def __get__(self, instance, owner):
        print(f'get {self.name}')
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        print(f"Running stage: {self.name}")
        instance.__dict__[self.name] = value

class ETLPipeline:
    extract_stage = ETLStage("extract_stage")
    transform_stage = ETLStage("transform_stage")
    load_stage = ETLStage("load_stage")

    def run(self, etl_class):
        self.extract_stage = etl_class.extract()
        self.transform_stage = etl_class.transform(self.extract_stage)
        self.load_stage = etl_class.load(self.transform_stage)


if __name__ == "__main__":
    # Running the ETL Pipeline
    pipeline = ETLPipeline()
    etl_process = CSVToDatabaseETL()
    pipeline.run(etl_process)