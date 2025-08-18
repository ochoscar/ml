import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ml.classification.simple.perceptron.Perceptron import Perceptron

print("Training perceptron...")
df = pd.read_csv(r'E:\ml\datasets\iris\iris.data', header=None, encoding='utf-8')

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:100, [0, 2]].values

ppn = Perceptron(eta=0.01, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()