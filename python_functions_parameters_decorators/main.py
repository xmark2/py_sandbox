import doctest
from python_functions_parameters_decorators.assignment1 import fncode, test_fncode
from python_functions_parameters_decorators.assignment2 import fncode_subgroup

def run_assignment1():
    output = fncode.seq_avg([1, 2])
    print(output)
    # print(fncode.seq_range.__doc__)
    help(fncode.seq_avg)
    print(fncode.seq_avg.__defaults__)
    test_fncode.run_test()
    doc_out = doctest.testmod()
    print(doc_out)

def run_assignment2():
    output = fncode_subgroup.subgroup_ranges(data=[1,2,3,5], group_size=4)
    print(output)
    output = fncode_subgroup.subgroup_apply(data=[1, 2, 3, 5], group_size=4)
    print(output)

if __name__ == '__main__':
    # run_assignment1()
    run_assignment2()


    # iter([1, 2, 3])
