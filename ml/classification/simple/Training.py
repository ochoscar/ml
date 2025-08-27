import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ml.classification.simple.Loader import loadIrisData
from ml.classification.simple.adeline.Adeline import AdalineSGD, AdelineGD
from ml.classification.simple.perceptron.Perceptron import Perceptron

def training(classifier):
    print("Training perceptron...")
    X, y = loadIrisData()

    ppn = classifier(eta=0.01, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()

    return ppn, X, y


if __name__ == "__main__":
    #training(Perceptron)
    #training(AdelineGD)
    training(AdalineSGD)