# Base class
class Shape:
    def area(self):
        pass  # Abstract method to be overridden by subclasses

# Derived classes with different implementations of the area method
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * (self.radius ** 2)

class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return 0.5 * self.base * self.height

# Polymorphism in action
def print_area(shape):
    print(f"The area is: {shape.area()}")

if __name__ == "__main__":
    # Creating different shape objects
    rectangle = Rectangle(10, 5)
    circle = Circle(7)
    triangle = Triangle(8, 6)

    # Using the same interface (area method) for different types of shapes
    print_area(rectangle)  # Outputs: The area is: 50
    print_area(circle)     # Outputs: The area is: 153.86
    print_area(triangle)   # Outputs: The area is: 24.0