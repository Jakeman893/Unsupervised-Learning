import numpy as np
from ann_classification import *
import unittest
from visualization import plot_scatterplot_matrix, visualize
from data_utils import iris_data, moons_data, enumerate_strings, normalize_data
from sklearn.neural_network import MLPClassifier

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

    def test_linear_classifier_iris(self):
        target = 'species'
        df = iris_data()
        mappings = enumerate_strings(df)
        X = df[['sepal_length', 'sepal_width']].values
        y = np.array([int(a > 0) for a in df[target]])
        clf = linear_model.LogisticRegressionCV()
        clf.fit(X,y)
        visualize(X, y, lambda x: clf.predict(x))

    # def test_sklearn_NN_classifier_moons(self):
    #     target = 'label'
    #     df = moons_data()
    #     X = df.loc[:, df.columns != target].values
    #     y = df[target].values
    #     clf = MLPClassifier()
    #     clf.fit(X,y)
    #     visualize(X, y, lambda x: clf.predict(x))

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

    def test_backprop_standard_iris_two_features(self):
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

    def test_backprop_standard_iris_all_features(self):
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

if __name__ == '__main__':
    # backprop_test = unittest.TestLoader().loadTestsFromTestCase(NN_Backprop_Test)
    # unittest.TextTestRunner(verbosity=1).run(backprop_test)
    b_visualization_test = unittest.TestLoader().loadTestsFromTestCase(Boundary_Visualization_Test)
    unittest.TextTestRunner(verbosity=1).run(b_visualization_test)