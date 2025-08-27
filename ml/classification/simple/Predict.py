import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from ml.classification.simple.Training import training
from ml.classification.simple.adeline.Adeline import AdalineSGD, AdelineGD
from ml.classification.simple.perceptron.Perceptron import Perceptron

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    map = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=map)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            marker=markers[idx],
            color=colors[idx],
            label=f'Class {cl}',
            edgecolor='black')
    plt.show()
        
if __name__ == "__main__":
    #ppn, X, y = training(Perceptron)
    #ppn, X, y = training(AdelineGD)
    ppn, X, y = training(AdalineSGD)
    plot_decision_regions(X, y, ppn)
