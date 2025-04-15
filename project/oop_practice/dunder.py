class Book:
    """
    The Book class represents a book with a title, author, and price.
    """

    def __init__(self, title, author, price):
        self.title = title
        self.author = author
        self.price = price

    def __str__(self):
        return f"'{self.title}' by {self.author} costs ${self.price}"

    def __add__(self, other):
        # Adding the price of two books
        return self.price + other.price

    def __len__(self):
        # Length of the title as an example
        return len(self.title)

    # def __doc__(self):
    #     return f'here is a documentation: {self.title}, {self.author}'


if __name__ == '__main__':
    # Using dunder methods
    book1 = Book("Python 101", "Mark", 30)
    book2 = Book("AI for Beginners", "John", 40)

    print(str(book1))  # __str__ method: 'Python 101' by Mark costs $30
    print(book1 + book2)  # __add__ method: 70
    print(len(book1))  # __len__ method: 10 (length of 'Python 101')
    print(book1.__doc__)
