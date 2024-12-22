import unittest

from project.python_functions_parameters_decorators.assignment.fncode import seq_avg, seq_range, subgroup_apply, subgroup_apply2


class TestFnCode(unittest.TestCase):
    def test_seq_avg(self):
        res = seq_avg([1,2,3])

        self.assertEqual(res, 2.0)

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
