import statistics
from functools import wraps


# from python_functions_parameters_decorators.assignment2.fncode_subgroup import debugger

def debugger(func):
    def wrapper(*args, **kwargs):
        # before
        res = func(*args, **kwargs)
        # after
        print(res)
        return res

    return wrapper


def limit(length):
    def decorator(function):
        def wrapper(*args, **kwargs):
            # before
            result = function(*args, **kwargs)
            result = result[:length]
            # after
            return result

        return wrapper

    return decorator


@limit(5)
def echo(foo):
    return foo

# echo = limit(5)(echo)


@debugger
def seq_avg(seq=[1, 2]):
    """Returns the mean/ avg of a sequence
    >>> seq_avg([1,2,3])
    2
    """
    return statistics.mean(seq)


@debugger
def seq_range(seq: list) -> float:
    "return range of sequence"
    return max(seq) - min(seq)


def seq_avg_old(seq):
    return sum(seq) / len(seq)


def subgroup_ranges(data, group_size=4, group=None):
    group = group if group is not None else []
    ranges = []
    it = iter(data)

    while True:
        try:
            group.append(next(it))
        except StopIteration:
            break

        if len(group) == group_size:
            ranges.append(seq_range(group))
            group = []
    return ranges


def subgroup_apply(data, group_size=4, group=None, agg_func=seq_range):
    group = group if group is not None else []
    ranges = []
    it = iter(data)

    while True:
        try:
            group.append(next(it))
        except StopIteration:
            break

        if len(group) == group_size:
            ranges.append(agg_func(group))
            group = []
    return ranges


def subgroup_apply2(data, *, group_size=4, group=None, agg_func=seq_range):
    group = group if group is not None else []
    ranges = []
    it = iter(data)

    while True:
        try:
            group.append(next(it))
        except StopIteration:
            break

        if len(group) == group_size:
            ranges.append(agg_func(group))
            group = []
    return ranges


# assignment 6
def subgroup_agg_gen(agg_func=seq_range):
    def wrapper(data, *, group_size=4, group=None):

        group = group if group is not None else []
        ranges = []
        it = iter(data)

        while True:
            try:
                group.append(next(it))
            except StopIteration:
                break

            if len(group) == group_size:
                ranges.append(agg_func(group))
                group = []
        return ranges

    return wrapper


def count2(func):
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return func(*args, **kwargs)

    wrapper.call_count = 0
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


@count2
def bar():
    "my docstring"
    pass


# decorator template
import functools


def decorator(func_to_decorate):
    @functools.wraps(func_to_decorate)
    def wrapper(*args, **kwargs):
        # do something before invocation
        result = func_to_decorate(*args, **kwargs)
        ## do something after
        return result

    return wrapper


# def contextmanager(func):
#     @wraps(func)
#     def helper(*args, **kwargs):
#         return GeneratorContextManager(func(*args, **kwargs))
#     return helper


if __name__ == '__main__':
    import doctest

    doctest.testmod()
