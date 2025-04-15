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



if __name__ == '__main__':
    # Using @classmethod
    Employee.set_raise_percentage(1.10)
    print(Employee.raise_percentage)  # Outputs: 1.10

    # employee_str = "Alice,50000"
    # new_employee = Employee.from_string(employee_str)
    # print(new_employee.name)  # Outputs: Alice
    # print(new_employee.salary)  # Outputs: 50000.0

    # Using the main constructor
    emp1 = Employee("Alice", "Manager", 75000)
    print(f"Employee 1: {emp1.name}, {emp1.position}, ${emp1.salary}")

    # Using the @classmethod constructor
    employee_string = "Bob-Developer-60000"
    emp2 = Employee.from_string(employee_string)
    # emp2.set_raise_percentage(1.10)
    print(f"Employee 2: {emp2.name}, {emp2.position}, ${emp2.salary}, {emp2.raise_percentage}")
