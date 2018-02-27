import numpy as np
from ann_classification import *
import unittest
from simple_classification import generate_data

class NN_Backprop_Test(unittest.TestCase):
    def test_backprop_standard(self):
        """
        """
        X, y = generate_data()
        model = build_model(X, y, 3)
        visualize(X, y, model)


if __name__ == '__main__':
    backprop_test = unittest.TestLoader().loadTestsFromTestCase(NN_Backprop_Test)
    unittest.TextTestRunner(verbosity=1).run(backprop_test)