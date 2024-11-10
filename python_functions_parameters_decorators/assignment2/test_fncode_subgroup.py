import unittest

from python_functions_parameters_decorators.assignment1.fncode import seq_range
from python_functions_parameters_decorators.assignment2.fncode_subgroup import subgroup_apply, subgroup_apply2


class TestFnCode(unittest.TestCase):
    def test_subgroup_apply(self):
        # res = seq_avg([1,2,3])
        res = subgroup_apply(data=[1, 2, 3, 5], group_size=4, agg_func=seq_range)

        self.assertListEqual(res, [4])

    def test_subgroup_apply2(self):
        # res = seq_avg([1,2,3])
        res = subgroup_apply2(data=[1, 2, 3, 5], group_size=4, agg_func=seq_range)

        self.assertListEqual(res, [4])
def run_test():
    unittest.main()

if __name__ == '__main__':
    unittest.main()
