import statistics


def seq_avg(seq):
    """Returns the mean/ avg of a sequence
    >>> seq_avg([1,2,3])
    2
    """
    return statistics.mean(seq)


def seq_range(seq: list) -> float:
    "return range of sequence"
    return max(seq) - min(seq)


def seq_avg_old(seq):
    return sum(seq) / len(seq)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
