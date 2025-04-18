class Point:
    __slots__ = ('x', 'y')  # Limits attributes to 'x' and 'y'

    def __init__(self, x, y):
        self.x = x
        self.y = y

if __name__ == "__main__":
    p = Point(10, 20)
    print(p.x, p.y)  # 10 20
    # p.z = 30  # Raises AttributeError: 'Point' object has no attribute 'z'
    print(dir(p))