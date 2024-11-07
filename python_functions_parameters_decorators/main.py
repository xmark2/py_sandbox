import doctest
from python_functions_parameters_decorators.assignment1 import fncode, test_fncode

if __name__ == '__main__':
    output = fncode.seq_avg([1, 2])
    print(output)
    # print(fncode.seq_range.__doc__)
    help(fncode.seq_avg)
    print(fncode.seq_avg.__defaults__)
    test_fncode.run_test()
    doc_out = doctest.testmod()
    print(doc_out)

    # iter([1, 2, 3])
