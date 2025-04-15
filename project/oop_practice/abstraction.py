from abc import ABC, abstractmethod

# Abstract Base Class
class ETLProcess(ABC):
    @abstractmethod
    def extract(self):
        """Method to extract data"""
        pass

    @abstractmethod
    def transform(self, data):
        """Method to transform data"""
        pass

    @abstractmethod
    def load(self, transformed_data):
        """Method to load data"""
        pass

# Concrete Class: Implementing the ETL process for a CSV file
class CSVETLProcess(ETLProcess):
    def extract(self):
        print("Extracting data from CSV file...")
        data = [
            {"id": 1, "name": "Mark", "age": 30},
            {"id": 2, "name": "John", "age": 25}
        ]
        return data

    def transform(self, data):
        print("Transforming data...")
        for record in data:
            record["age"] += 1  # Example transformation: increment age
        return data

    def load(self, transformed_data):
        print("Loading transformed data to database...")
        for record in transformed_data:
            print(f"Loaded record: {record}")

# Concrete Class: Implementing the ETL process for an API
class APIETLProcess(ETLProcess):
    def extract(self):
        print("Extracting data from API...")
        data = [
            {"id": 101, "name": "Alice", "score": 85},
            {"id": 102, "name": "Bob", "score": 90}
        ]
        return data

    def transform(self, data):
        print("Transforming data...")
        for record in data:
            record["status"] = "pass" if record["score"] > 80 else "fail"
        return data

    def load(self, transformed_data):
        print("Loading transformed data to dashboard...")
        for record in transformed_data:
            print(f"Dashboard record: {record}")


if __name__ == '__main__':
    # Using the ETL processes
    csv_etl = CSVETLProcess()
    data = csv_etl.extract()
    transformed_data = csv_etl.transform(data)
    csv_etl.load(transformed_data)

    api_etl = APIETLProcess()
    data = api_etl.extract()
    transformed_data = api_etl.transform(data)
    api_etl.load(transformed_data)
