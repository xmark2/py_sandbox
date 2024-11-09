import unittest

from python_functions_parameters_decorators.assignment2.fncode_subgroup import subgroup_apply


class TestFnCode(unittest.TestCase):
    def test_subgroup_apply(self):
        # res = seq_avg([1,2,3])
        res = subgroup_apply(data=[1, 2, 3, 5], group_size=4)

        self.assertListEqual(res, [4])

def run_test():
    unittest.main()

if __name__ == '__main__':
    unittest.main()
