import unittest
import numpy as np
from hill_climbing import random_bitstring, score_bitstring

def candidate_funct_numerical(bounds):
    return np.random.uniform(bounds[0], bounds[1])    

from stock_data import quandl_stocks
from visualization import plot_stock_data
from hill_climbing import maximum_number_move, hill_climbing

class Hill_Climbing_Tests(unittest.TestCase):
    def test_hill_climbing_pos_slope(self):
        arr = range(0,10)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        move_funct = lambda X, i: maximum_number_move(X, i, step_size=1, bounds = (0,9))
        pos, score = hill_climbing(eval_funct, move_funct, 0)
        self.assertEquals(score, 9)
        self.assertAlmostEqual(round(pos), 9)

    def test_hill_climbing_neg_slope(self):
        arr = range(9, -1, -1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        move_funct = lambda X, i: maximum_number_move(X, i, step_size=1, bounds = (0,9))
        pos, score = hill_climbing(eval_funct, move_funct, 9)
        self.assertEquals(score, 9)
        self.assertAlmostEqual(round(pos), 0)

    # Tests function of form /\, see if we can find peak
    def test_hill_climbing_mountain_from_left(self):
        arr = range(0,10) + range(10,-1,-1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        move_funct = lambda X, i: maximum_number_move(X, i, step_size=1, bounds = (0, 20))
        pos, score = hill_climbing(eval_funct, move_funct, 0)
        self.assertEquals(score, 10)
        self.assertAlmostEqual(round(pos), 10)

    def test_hill_climbing_mountain_from_right(self):
        arr = range(0,10) + range(10,-1,-1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        move_funct = lambda X, i: maximum_number_move(X, i, step_size=1, bounds = (0,20))
        pos, score = hill_climbing(eval_funct, move_funct, 0)
        self.assertEquals(score, 10)
        self.assertAlmostEqual(round(pos), 10)

    def test_hill_climbing_left_peak_first(self):
        arr = range(0,10) + range(10,-1,-1) + range(1, 6)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        move_funct = lambda X, i: maximum_number_move(X, i, step_size=1, bounds = (0,25))
        pos, score = hill_climbing(eval_funct, move_funct, 20)
        self.assertTrue(score == 10 or score == 5)
        self.assertTrue(round(pos) == 10 or round(pos) == 25)

    def test_hill_climbing_aapl_data(self):
        aapl = quandl_stocks('AAPL')
        close = aapl['WIKI/AAPL - Adj. Close']
        arr = close.values

        eval_funct = lambda x: arr[int(round(abs(x)))]
        move_funct = lambda X, i: maximum_number_move(X, i, step_size=5, bounds = (0,len(arr)))
        pos, score = hill_climbing(eval_funct, move_funct, 6)
        plot_stock_data(aapl['WIKI/AAPL - Adj. Close'], close.index[pos], score)
        # self.assertEquals(score, 10)
        # self.assertAlmostEqual(round(pos), 10)

    # TODO: Bitstring hill climbing

from hill_climbing import randomized_hill_climbing

class Random_Hill_Climbing_Tests(unittest.TestCase):
    def test_pos_slope(self):
        arr = range(0,10)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        move_funct = lambda X, i: maximum_number_move(X, i, step_size=1, bounds=(0,9))
        pos, score = randomized_hill_climbing(eval_funct, move_funct, 0)
        self.assertEquals(score, 9)
        self.assertAlmostEqual(round(pos), 9)

    def test_slope(self):
        arr = range(9, -1, -1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        move_funct = lambda X, i: maximum_number_move(X, i, step_size=1, bounds=(0,9))
        pos, score = randomized_hill_climbing(eval_funct, move_funct, 9)
        self.assertEquals(score, 9)
        self.assertAlmostEqual(round(pos), 0)

    # Tests function of form /\, see if we can find peak
    def test_mountain_from_left(self):
        arr = range(0,10) + range(10,-1,-1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        move_funct = lambda X, i: maximum_number_move(X, i, step_size=1, bounds=(0,20))
        pos, score = randomized_hill_climbing(eval_funct, move_funct, 0)
        self.assertEquals(score, 10)
        self.assertAlmostEqual(round(pos), 10)

    def test_mountain_from_right(self):
        arr = range(0,10) + range(10,-1,-1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        move_funct = lambda X, i: maximum_number_move(X, i, step_size=1, bounds=(0,20))
        pos, score = randomized_hill_climbing(eval_funct, move_funct, 0)
        self.assertEquals(score, 10)
        self.assertAlmostEqual(round(pos), 10)

    def test_two_peaks(self):
        arr = range(0,10) + range(10,-1,-1) + range(1, 6)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        move_funct = lambda X, i: maximum_number_move(X, i, step_size=1, bounds=(0,25))
        pos, score = randomized_hill_climbing(eval_funct, move_funct, 20)
        self.assertTrue(score == 10 or score == 5)
        self.assertTrue(round(pos) == 10 or round(pos) == 25)

    def test_two_peaks_high_prob_jump(self):
        arr = range(0,10) + range(10,-1,-1) + range(1, 6)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        move_funct = lambda X, i: maximum_number_move(X, i, step_size=1, bounds=(0,25))
        pos, score = randomized_hill_climbing(eval_funct, move_funct, 20, prob_jump=1)
        self.assertTrue(score == 10 or score == 5)
        self.assertTrue(round(pos) == 10 or round(pos) == 25)

    def test_hill_climbing_aapl_data(self):
        aapl = quandl_stocks('AAPL')
        close = aapl['WIKI/AAPL - Adj. Close']
        arr = close.values

        eval_funct = lambda x: arr[int(round(abs(x)))]
        move_funct = lambda X, i: maximum_number_move(X, i, step_size=100, bounds = (0,len(arr)))
        pos, score = randomized_hill_climbing(eval_funct, move_funct, 6, prob_jump=0.25)
        plot_stock_data(aapl['WIKI/AAPL - Adj. Close'], close.index[pos], score)

    # def test_all_ones_bitstring(self):
    #     pos, score = randomized_hill_climbing(score_bitstring, random_bitstring, [0,0,0,0], prob_jump=0.5,bounds=4)
    #     self.assertEqual(pos, [1,1,1,1])
    #     self.assertEqual(score, 4)

from hill_climbing import simulated_annealing

class Simulated_Annealing_Tests(unittest.TestCase):
    def test_pos_slope(self):
        arr = range(0,10)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = simulated_annealing(eval_funct, candidate_funct_numerical, 0, iters_per_decay=1, bounds = (0, 9))
        self.assertEquals(score, 9)
        self.assertAlmostEqual(round(pos), 9)

    def test_neg_slope(self):
        arr = range(9, -1, -1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = simulated_annealing(eval_funct, candidate_funct_numerical, 9, iters_per_decay=1, bounds = (0, 9))
        self.assertEquals(score, 9)
        self.assertAlmostEqual(round(pos), 0)

    # Tests function of form /\, see if we can find peak
    def test_mountain_from_left(self):
        arr = range(0,10) + range(10,-1,-1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = simulated_annealing(eval_funct, candidate_funct_numerical, 0, bounds = (0, 20))
        self.assertEquals(score, 10)
        self.assertAlmostEqual(round(pos), 10)

    def test_mountain_from_right(self):
        arr = range(0,10) + range(10,-1,-1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = simulated_annealing(eval_funct, candidate_funct_numerical, 0, bounds = (0, 20))
        self.assertEquals(score, 10)
        self.assertEquals(round(pos), 10)

    def test_two_peaks(self):
        arr = range(0,10) + range(10,-1,-1) + range(1, 6)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = simulated_annealing(eval_funct, candidate_funct_numerical, 0, bounds = (0, 25))
        self.assertEquals(score, 10)
        self.assertEquals(round(pos), 10)

if __name__ == '__main__':
    # hill_climbing_test = unittest.TestLoader().loadTestsFromTestCase(Hill_Climbing_Tests)
    # unittest.TextTestRunner(verbosity=1).run(hill_climbing_test)
    rand_hill_climbing_test = unittest.TestLoader().loadTestsFromTestCase(Random_Hill_Climbing_Tests)
    unittest.TextTestRunner(verbosity=1).run(rand_hill_climbing_test)
    # sim_annealing_test = unittest.TestLoader().loadTestsFromTestCase(Simulated_Annealing_Tests)
    # unittest.TextTestRunner(verbosity=1).run(sim_annealing_test)