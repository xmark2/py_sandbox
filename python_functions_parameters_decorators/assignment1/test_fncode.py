import unittest

from python_functions_parameters_decorators.assignment1.fncode import seq_avg


class TestFnCode(unittest.TestCase):
    def test_seq_avg(self):
        res = seq_avg([1,2,3])

        self.assertEqual(res, 2.0)

def run_test():
    unittest.main()

if __name__ == '__main__':
    unittest.main()
