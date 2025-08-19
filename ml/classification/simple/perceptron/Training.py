import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ml.classification.simple.perceptron.Loader import loadIrisData
from ml.classification.simple.perceptron.Perceptron import Perceptron

def training_perceptron():
    print("Training perceptron...")
    X, y = loadIrisData()

    ppn = Perceptron(eta=0.01, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()

    return ppn, X, y