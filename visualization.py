import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_scatterplot_matrix(df, target):
    sns.set(style="ticks")
    sns.pairplot(df, hue=target)
    plt.show()

# Visualization of the boundary information of two attributes
def visualize(X, y, predict_lambda):
    # TODO: Fix so we can see plots over higher dimensional inputs and outputs
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(predict_lambda, X, y)
    plt.title("Logistic Regression")

# Plots decision boundary given the prediction function
def plot_decision_boundary(pred_func, X, y):
    cmap_light = ListedColormap(np.random.rand(256,3))
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Accent)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.jet)
    plt.axis([np.min(X[:, 0])-0.5, np.max(X[:, 0])+0.5, np.min(X[:, 1])-0.5, np.max(X[:, 1])+0.5])
    plt.show()
