from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def move(self):
        pass

class Car(Vehicle):
    def move(self):
        print("Driving on the road")

class Boat(Vehicle):
    def move(self):
        print("Sailing on water")

# Uncommenting the following will raise an error since 'move' is not implemented:
class Bicycle(Vehicle):
    pass


if __name__ == "__main__":
    car_inst = Car()
    car_inst.move()
    boat_inst = Boat()
    boat_inst.move()
    bic_inst = Bicycle()