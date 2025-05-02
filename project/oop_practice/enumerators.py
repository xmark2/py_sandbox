from enum import StrEnum

class Direction(StrEnum):
    UP = "UP"
    DOWN = "DOWN"

def move(direction:Direction)->None:
    if direction == Direction.UP:
        print("Going up")
    elif direction == Direction.DOWN:
        print("Going down")
    else:
        raise ValueError()

if __name__=="__main__":
    move(Direction.UP)
    move("DOWN")