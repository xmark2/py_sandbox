class NonNegative:
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if value < 0:
            raise ValueError(f"{self.name} must be non-negative!")
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        raise AttributeError(f"Cannot delete {self.name}")

class Person:
    age = NonNegative("age")  # Descriptor instance

    def __init__(self, name, age):
        self.name = name
        self.age = age

if __name__ == "__main__":
    # Example usage
    try:
        person = Person("Mark", 30)
        print(f"Name: {person.name}, Age: {person.age}")  # Outputs: Name: Mark, Age: 30

        person.age = -10  # Raises ValueError
    except ValueError as e:
        print(e)

    try:
        del person.age  # Attempt to delete the age attribute (raises AttributeError)
    except AttributeError as e:
        print(e)