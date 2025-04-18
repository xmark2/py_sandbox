class Employee:
    raise_percentage = 1.05  # Class attribute

    def __init__(self, name, position, salary):
        self.name = name
        self.position = position
        self.salary = salary

    @classmethod
    def set_raise_percentage(cls, amount):
        cls.raise_percentage = amount

    @classmethod
    def from_string(cls, employee_string):
        """
        Alternative constructor that parses an employee's data from a string.

        Args:
            employee_string (str): A string with the format 'Name-Position-Salary'.
        """
        name, position, salary = employee_string.split("-")
        return cls(name, position, float(salary))

class MathOperations:
    @staticmethod
    def add_numbers(a, b):
        return a + b

    @staticmethod
    def multiply_numbers(a, b):
        return a * b


class Circle:
    def __init__(self, radius):
        self._radius = radius  # Protected attribute

    @property
    def radius(self):
        """Getter for radius"""
        return self._radius

    @radius.setter
    def radius(self, value):
        """Setter for radius"""
        if value >= 0:
            self._radius = value
        else:
            raise ValueError("Radius must be non-negative")

    @radius.deleter
    def radius(self):
        """Deleter for radius"""
        self._radius = None

    @property
    def area(self):
        """Area is a read-only property (no setter)"""
        return 3.14 * (self._radius ** 2)



if __name__ == "__main__":
    # Example usage
    circle = Circle(5)
    print(circle.radius)
    circle.radius = 10
    print(circle.radius)

    del circle.radius
    print(circle.radius)
    # print(f"Radius: {circle.radius}")  # Accessing the radius (getter)
    #
    # circle.radius = 10  # Updating the radius (setter)
    # print(f"Updated Radius: {circle.radius}")
    # print(f"Area: {circle.area}")  # Accessing the area (read-only property)
    #
    # print(circle.radius)
    #
    # # Trying to set a negative radius (raises error)
    # # circle.radius = -3  # Uncommenting this line will raise ValueError



    # # Using @classmethod
    # Employee.set_raise_percentage(1.10)
    # print(Employee.raise_percentage)  # Outputs: 1.10
    #
    # # employee_str = "Alice,50000"
    # # new_employee = Employee.from_string(employee_str)
    # # print(new_employee.name)  # Outputs: Alice
    # # print(new_employee.salary)  # Outputs: 50000.0
    #
    # # Using the main constructor
    # emp1 = Employee("Alice", "Manager", 75000)
    # print(f"Employee 1: {emp1.name}, {emp1.position}, ${emp1.salary}")
    #
    # # Using the @classmethod constructor
    # employee_string = "Bob-Developer-60000"
    # emp2 = Employee.from_string(employee_string)
    # # emp2.set_raise_percentage(1.10)
    # print(f"Employee 2: {emp2.name}, {emp2.position}, ${emp2.salary}, {emp2.raise_percentage}")
    #
    # # Using @staticmethod
    # print(MathOperations.add_numbers(5, 3))  # Outputs: 8
    # print(MathOperations.multiply_numbers(5, 3))  # Outputs: 15
    #
