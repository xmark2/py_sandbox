## Advanced OOP Topics Summary

| **Topic**               | **Purpose**                                     |
|--------------------------|-------------------------------------------------|
| **Metaclasses**          | Control class creation and behavior.           |
| **Abstract Base Classes**| Define blueprints for subclasses.              |
| **Property Decorators**  | Add computed properties and validation logic.   |
| **Descriptors**          | Manage attributes with custom `__get__`/`__set__`. |
| **Dynamic Attributes**   | Handle attributes dynamically at runtime.       |
| **Mixins**               | Modularize reusable behaviors.                 |
| **Slots**                | Optimize memory usage for classes.             |
| **Context Managers**     | Efficiently manage resources.                  |


# functools

The functools module in Python is part of the standard library and provides higher-order functions that operate on or return other functions. These tools are especially useful for functional programming and improving code reusability. Here's a summary of some of the most useful methods in the functools module:

1. functools.lru_cache
- Purpose: Implements a caching mechanism for functions to store recently computed results, reducing repetitive computations.
- When to Use: Useful when dealing with expensive or repetitive function calls (e.g., recursion).

Example:
```
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))  # Outputs: 55
```

- Features:- maxsize limits the cache size. Set it to None for unlimited caching.
- Access cached results for faster performance on repeated inputs.



2. functools.partial
- Purpose: Creates a new function with some arguments of the original function pre-filled.
- When to Use: Simplifies function calls when you repeatedly use the same arguments.

Example:
```commandline
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)  # Pre-fill exponent as 2
print(square(4))  # Outputs: 16

```

- Features:- Reduces redundancy when frequently calling a function with fixed parameters.



3. functools.reduce
- Purpose: Applies a rolling computation to a sequence of items (e.g., summing, multiplying, or finding a maximum).
- When to Use: Useful when performing reductions where each step depends on the previous step.

Example:
```
from functools import reduce

nums = [1, 2, 3, 4, 5]
result = reduce(lambda x, y: x * y, nums)  # Multiply all numbers
print(result)  # Outputs: 120
```

- Features:- Often replaced by Python’s built-in functions like sum() or max() for simple use cases.


4. functools.wraps
- Purpose: Preserves the metadata (name, docstring, etc.) of the original function when creating decorators.
- When to Use: Use in custom decorators to avoid losing the original function's information.

Example:
```
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Before the function call")
        result = func(*args, **kwargs)
        print("After the function call")
        return result
    return wrapper

@my_decorator
def say_hello():
    """This function says hello."""
    print("Hello!")

say_hello()
print(say_hello.__doc__)  # Outputs: This function says hello.
```

- Features:- Makes the decorated function look and behave like the original function.



5. functools.cached_property (Added in Python 3.8)
- Purpose: Creates a read-only property that is computed once and cached for future access.
- When to Use: Use for expensive or one-time calculations that don't change over an object's lifetime.

Example:
```
from functools import cached_property

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @cached_property
    def area(self):
        print("Computing area...")
        return 3.14 * (self.radius ** 2)

c = Circle(10)
print(c.area)  # Computes and outputs: 314.0
print(c.area)  # Returns cached value without recomputing
```

- Features:- Great for optimizing attribute access in classes.



6. functools.total_ordering
- Purpose: Simplifies implementing all comparison methods (<, <=, >, >=) for a class based on one or two methods.
- When to Use: Use when creating custom classes with ordering.

Example:
```
from functools import total_ordering

@total_ordering
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        return self.age == other.age

    def __lt__(self, other):
        return self.age < other.age

# Only __eq__ and __lt__ are required; the rest are auto-generated.
p1 = Person("Alice", 30)
p2 = Person("Bob", 25)
print(p1 > p2)  # Outputs: True
```

- Features:- Reduces boilerplate when defining custom comparison logic.



7. functools.singledispatch
- Purpose: Implements function overloading based on the type of the first argument.
- When to Use: Use when you want a generic function to behave differently depending on input type.

Example:
```
from functools import singledispatch

@singledispatch
def process(data):
    print(f"Default processing for {data}")

@process.register(int)
def _(data):
    print(f"Processing integer: {data}")

@process.register(str)
def _(data):
    print(f"Processing string: {data}")

process(42)  # Outputs: Processing integer: 42
process("hello")  # Outputs: Processing string: hello
process([1, 2, 3])  # Outputs: Default processing for [1, 2, 3]
```


- Features:- Enables clean, type-dependent function behavior without if-elif checks.



8. functools.partialmethod
- Purpose: Similar to partial, but specifically for use with methods in a class.
- When to Use: Use when you need to partially apply arguments to instance methods.

Example:
```
from functools import partialmethod

class MathOperations:
    def power(self, base, exponent):
        return base ** exponent

    square = partialmethod(power, exponent=2)  # Pre-fill exponent as 2

math = MathOperations()
print(math.square(5))  # Outputs: 25
```

- Features:- Simplifies repetitive method calls in classes with pre-filled arguments.



Summary Table:

| **Function** | **Purpose** | 
|--------------------------|-------------------------------------------------|
| lru_cache | Caches results of expensive functions for repeated use. | 
| partial | Creates a new function with pre-filled arguments. | 
| reduce | Performs rolling computations on an iterable. | 
| wraps | Preserves metadata of decorated functions. | 
| cached_property | Computes and caches a read-only property value. | 
| total_ordering | Automatically generates missing comparison methods. | 
| singledispatch | Overloads functions based on the input argument type. | 
| partialmethod | Simplifies partial application for class methods. | 





## the * and ** operators

In Python, the * and ** operators are versatile tools with different purposes depending on the context in which they are used. Here's a detailed breakdown of their usage:

1. Function Arguments
Unpacking Positional Arguments with *
- * allows you to pass a variable number of positional arguments to a function.
- Inside a function definition, it is commonly named *args.

Example:
```
def my_function(*args):
    print(f"Positional arguments: {args}")

my_function(1, 2, 3, 4)  # Outputs: Positional arguments: (1, 2, 3, 4)
```

Here, args is a tuple that collects all the additional positional arguments.

Unpacking Keyword Arguments with **
- ** allows you to pass a variable number of keyword arguments to a function.
- Inside a function definition, it is commonly named **kwargs.

Example:
```
def my_function(**kwargs):
    print(f"Keyword arguments: {kwargs}")

my_function(name="Mark", age=30, city="Budapest")
# Outputs: Keyword arguments: {'name': 'Mark', 'age': 30, 'city': 'Budapest'}
```

Here, kwargs is a dictionary that collects all additional keyword arguments.

2. Function Argument Unpacking
Unpacking a List or Tuple with *
- * is used to unpack a list or tuple into separate arguments when calling a function.

Example:
```
def add(a, b, c):
    return a + b + c

numbers = [1, 2, 3]
print(add(*numbers))  # Outputs: 6
```


Unpacking a Dictionary with **
- ** is used to unpack a dictionary into keyword arguments when calling a function.

Example:
```
def greet(name, age):
    print(f"Hello, {name}. You are {age} years old.")

person = {"name": "Mark", "age": 30}
greet(**person)
# Outputs: Hello, Mark. You are 30 years old.
```


3. Packing and Unpacking
The * and ** operators can be used in function definitions (to pack arguments) and function calls (to unpack arguments).
Example (Both Packing and Unpacking):
```
def my_function(*args, **kwargs):
    print(f"Positional arguments: {args}")
    print(f"Keyword arguments: {kwargs}")

my_function(1, 2, 3, name="Mark", age=30)
# Outputs:
# Positional arguments: (1, 2, 3)
# Keyword arguments: {'name': 'Mark', 'age': 30}
```


4. Iterable Unpacking in Assignments (Using *)
- * can also unpack parts of an iterable during assignments.

Example:
```
numbers = [1, 2, 3, 4, 5]
a, *b, c = numbers
print(a)  # Outputs: 1
print(b)  # Outputs: [2, 3, 4]
print(c)  # Outputs: 5
```

Here, the middle part of the list is unpacked into b as a sublist.

5. Combining Multiple Iterables
- * can unpack multiple iterables into a single one, e.g., lists or tuples.

Example:
```
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = [*list1, *list2]
print(combined)  # Outputs: [1, 2, 3, 4, 5, 6]
```


6. Passing Arguments to print()
The * operator can be used to pass all elements of a list or tuple as separate arguments to a function like print.
Example:
```
items = ["Python", "is", "great"]
print(*items)  # Outputs: Python is great
```


7. Advanced Use in Function Decorators
- In decorators, *args and **kwargs are commonly used to pass arguments dynamically.

Example:
```
def decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@decorator
def add(a, b):
    return a + b

print(add(2, 3))
# Outputs:
# Before function call
# After function call
# 5

```

| Symbol | Usage | 
|--------------------------|-------------------------------------------------|
| * | Unpacks positional arguments, iterable unpacking, list/tuple merging. | 
| ** | Unpacks keyword arguments or dictionary into named arguments. | 



## Multiprocessing

```commandline
from multiprocessing import Pool

# Function to compute square
def compute_square(n):
    return n ** 2
    

with Pool(processes=4) as pool:  # Use 4 parallel processes
    numbers = [1, 2, 3, 4, 5]
    results = pool.map(compute_square, numbers)
    print("Squares:", results)
```

## Threading

```commandline
def add_nums(*args):
  print(sum(args))
  

nums_ls = [(1,2,3), (4,5,6), (7,8,9)]

threads = []
for nums in nums_ls:
  t = threading.Thread(target=add_nums, args=nums)
  threads.append(t)
  t.start()

for t in threads:
  t.join()

print('process completed')
```

