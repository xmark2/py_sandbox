class Meta(type):
    def __new__(cls, name, bases, dct):
        if 'speak' not in dct:
            raise TypeError(f"Class {name} must implement a 'speak' method")
        return super().__new__(cls, name, bases, dct)

class Animal(metaclass=Meta):
    def speak(self):
        pass  # Classes inheriting from Animal must implement this method

class Dog(Animal):
    def speak(self):
        print("Woof!")

# Uncommenting the following will raise an error since 'speak' is missing:
class Fish(Animal):
    pass

if __name__ == "__main__":
    dog_inst = Dog()
    dog_inst.speak()
    fish_inst = Fish()