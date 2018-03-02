import numpy as np
from ann_classification import *
import unittest
from visualization import plot_scatterplot_matrix, visualize
from data_utils import *
from sklearn.neural_network import MLPClassifier

def candidate_funct_numerical(bounds):
    return np.random.uniform(bounds[0], bounds[1])    

class NN_Backprop_Test(unittest.TestCase):
    def test_backprop_standard_moons(self):
        """
        Check moons data with backprop classifier
        Moons has 2 features
        """
        target = 'label'
        df = moons_data()
        X = df.loc[:, df.columns != target].values
        y = df[target].values
        print "Moons"
        model = build_model(X, y, 3, print_loss=True)

    def test_backprop_standard_iris_two_features(self):
        """
        Check iris data with backprop weight optimizer
        Iris has 2 features and binary classification outputs
        """
        target = 'species'
        df = iris_data()
        mappings = enumerate_strings(df)
        X = df[['sepal_length', 'sepal_width']].values
        y = np.array([int(a > 0) for a in df[target]])
        print "Iris (2-feature, binary)"
        model = build_model(X, y, 3, print_loss=True)

    def test_backprop_standard_iris_binary(self):
        """
        Check iris data with backprop weight optimizer
        Iris has 4 features and binary classification outputs
        """
        target = 'species'
        df = iris_data()
        mappings = enumerate_strings(df)
        X = df.loc[:, df.columns != target].values
        y = np.array([int(a > 0) for a in df[target]])
        print "Iris (4-feature, binary)"
        model = build_model(X, y, 3, print_loss=True)

    def test_backprop_standard_iris(self):
        """
        Check iris data with backprop weight optimizer
        Iris has 4 features and 3 classification outputs
        """
        target = 'species'
        df = iris_data()
        mappings = enumerate_strings(df)
        X = df.loc[:, df.columns != target].values
        y = df[target].values
        print "Iris (4-feature, trinary)"
        model = build_model(X, y, 3, print_loss=True)

class Boundary_Visualization_Test(unittest.TestCase):
    # Linear Classifier
    def test_linear_classifier_moons(self):
        target = 'label'
        df = moons_data()
        X = df.loc[:, df.columns != target].values
        y = df[target].values
        clf = linear_model.LogisticRegressionCV()
        clf.fit(X,y)
        visualize(X, y, lambda x: clf.predict(x))

    def test_linear_classifier_iris_two(self):
        target = 'species'
        df = iris_data()
        mappings = enumerate_strings(df)
        X = df[['sepal_length', 'sepal_width']].values
        y = np.array([int(a > 0) for a in df[target]])
        clf = linear_model.LogisticRegressionCV()
        clf.fit(X,y)
        visualize(X, y, lambda x: clf.predict(x))

    def test_linear_classifier_iris_all(self):
        target = 'species'
        df = iris_data()
        mappings = enumerate_strings(df)
        X = df[['sepal_length', 'sepal_width']].values
        y = df[target].values
        clf = linear_model.LogisticRegressionCV()
        clf.fit(X,y)
        visualize(X, y, lambda x: clf.predict(x))

    def test_backprop_standard_moons(self):
        """
        Check moons data with backprop classifier
        Moons has 2 features
        """
        target = 'label'
        df = normalize_data(moons_data(), target)
        X = df.loc[:, df.columns != target].values
        y = df[target].values
        model = build_model(X, y, 3)
        visualize(X, y, lambda x:predict(model,x))

    def test_backprop_standard_iris_two_species(self):
        """
        Check iris data with backprop weight optimizer
        Iris has 2 features and binary classification outputs
        """
        target = 'species'
        df = normalize_data(iris_data(), target)
        mappings = enumerate_strings(df)
        X = df[['sepal_length', 'sepal_width']].values
        y = np.array([int(a > 0) for a in df[target]])
        model = build_model(X, y, 3)
        visualize(X, y, lambda x:predict(model,x))

    def test_backprop_standard_iris_all_species(self):
        """
        Check iris data with backprop weight optimizer
        Iris has 2 features and binary classification outputs
        """
        target = 'species'
        df = normalize_data(iris_data(), target)
        mappings = enumerate_strings(df)
        X = df[['sepal_length', 'sepal_width']].values
        y = df[target].values
        model = build_model(X, y, 3)
        visualize(X, y, lambda x:predict(model,x))

    def test_backprop_standard_flights_months(self):
        """
        Check iris data with backprop weight optimizer
        Iris has 2 features and binary classification outputs
        """
        target = 'month'
        df = normalize_data(flights_data(), target)
        mappings = enumerate_strings(df)
        X = df.loc[:, df.columns != target].values
        y = df[target].values
        model = build_model(X, y, 3)
        visualize(X, y, lambda x:predict(model,x))

from optimization_algs import hill_climbing_step, hill_climbing

class Hill_Climbing_Tests(unittest.TestCase):
    def test_hill_climbing_step_pos_slope(self):
        arr = range(0,10)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = hill_climbing_step(eval_funct, candidate_funct_numerical, 0, 2, bounds = (0, 9))
        self.assertEquals(score, 2)
        self.assertAlmostEqual(round(pos), 2)

    def test_hill_climbing_step_neg_slope(self):
        arr = range(10,-1,-1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = hill_climbing_step(eval_funct, candidate_funct_numerical, 0, 2, bounds = (0, 9))
        self.assertEquals(score, 10)
        self.assertEquals(pos, 0)

    def test_hill_climbing_pos_slope(self):
        arr = range(0,10)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = hill_climbing(eval_funct, candidate_funct_numerical, 0, epsilon=0.01, bounds = (0, 9))
        self.assertEquals(score, 9)
        self.assertAlmostEqual(round(pos), 9)

    def test_hill_climbing_neg_slope(self):
        arr = range(9, -1, -1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = hill_climbing(eval_funct, candidate_funct_numerical, 9, epsilon=0.01, bounds = (0, 9))
        self.assertEquals(score, 9)
        self.assertAlmostEqual(round(pos), 0)

    # Tests function of form /\, see if we can find peak
    def test_hill_climbing_mountain_from_left(self):
        arr = range(0,10) + range(10,-1,-1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = hill_climbing(eval_funct, candidate_funct_numerical, 0, epsilon=0.01, bounds = (0, 20))
        self.assertEquals(score, 10)
        self.assertAlmostEqual(round(pos), 10)

    def test_hill_climbing_mountain_from_right(self):
        arr = range(0,10) + range(10,-1,-1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = hill_climbing(eval_funct, candidate_funct_numerical, 0, epsilon=0.01, bounds = (0, 20))
        self.assertEquals(score, 10)
        self.assertAlmostEqual(round(pos), 10)

    def test_hill_climbing_left_peak_first(self):
        arr = range(0,10) + range(10,-1,-1) + range(1, 6)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = hill_climbing(eval_funct, candidate_funct_numerical, 20, epsilon=0.01, bounds = (0, 25))
        self.assertTrue(score == 10 or score == 5)
        self.assertTrue(round(pos) == 10 or round(pos) == 25)

from optimization_algs import randomized_hill_climbing

class Random_Hill_Climbing_Tests(unittest.TestCase):
    def test_pos_slope(self):
        arr = range(0,10)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = randomized_hill_climbing(eval_funct, candidate_funct_numerical, 0, epsilon=0.01, bounds = (0, 9))
        self.assertEquals(score, 9)
        self.assertAlmostEqual(round(pos), 9)

    def test_slope(self):
        arr = range(9, -1, -1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = randomized_hill_climbing(eval_funct, candidate_funct_numerical, 9, epsilon=0.01, bounds = (0, 9))
        self.assertEquals(score, 9)
        self.assertAlmostEqual(round(pos), 0)

    # Tests function of form /\, see if we can find peak
    def test_mountain_from_left(self):
        arr = range(0,10) + range(10,-1,-1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = randomized_hill_climbing(eval_funct, candidate_funct_numerical, 0, epsilon=0.01, bounds = (0, 20))
        self.assertEquals(score, 10)
        self.assertAlmostEqual(round(pos), 10)

    def test_mountain_from_right(self):
        arr = range(0,10) + range(10,-1,-1)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = randomized_hill_climbing(eval_funct, candidate_funct_numerical, 0, epsilon=0.01, bounds = (0, 20))
        self.assertEquals(score, 10)
        self.assertAlmostEqual(round(pos), 10)

    def test_two_peaks(self):
        arr = range(0,10) + range(10,-1,-1) + range(1, 6)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = randomized_hill_climbing(eval_funct, candidate_funct_numerical, 20, epsilon=0.01, bounds = (0, 25))
        self.assertTrue(score == 10 or score == 5)
        self.assertTrue(round(pos) == 10 or round(pos) == 25)

    def test_two_peaks_high_prob_jump(self):
        arr = range(0,10) + range(10,-1,-1) + range(1, 6)
        eval_funct = lambda x: arr[int(round(abs(x)))]
        pos, score = randomized_hill_climbing(eval_funct, candidate_funct_numerical, 20, epsilon=0.01, prob_jump=0.75, bounds = (0, 25))
        self.assertTrue(score == 10 or score == 5)
        self.assertTrue(round(pos) == 10 or round(pos) == 25)

from optimization_algs import simulated_annealing

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
    # backprop_test = unittest.TestLoader().loadTestsFromTestCase(NN_Backprop_Test)
    # unittest.TextTestRunner(verbosity=1).run(backprop_test)
    # b_visualization_test = unittest.TestLoader().loadTestsFromTestCase(Boundary_Visualization_Test)
    # unittest.TextTestRunner(verbosity=1).run(b_visualization_test)
    # hill_climbing_test = unittest.TestLoader().loadTestsFromTestCase(Hill_Climbing_Tests)
    # unittest.TextTestRunner(verbosity=1).run(hill_climbing_test)
    # rand_hill_climbing_test = unittest.TestLoader().loadTestsFromTestCase(Random_Hill_Climbing_Tests)
    # unittest.TextTestRunner(verbosity=1).run(rand_hill_climbing_test)
    sim_annealing_test = unittest.TestLoader().loadTestsFromTestCase(Simulated_Annealing_Tests)
    unittest.TextTestRunner(verbosity=1).run(sim_annealing_test)