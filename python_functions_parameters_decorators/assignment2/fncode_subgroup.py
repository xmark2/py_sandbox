from python_functions_parameters_decorators.assignment1.fncode import seq_range


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
            group  = []
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