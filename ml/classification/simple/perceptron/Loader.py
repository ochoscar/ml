import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
print("Loading data...")
df = pd.read_csv(r'E:\ml\datasets\iris\iris.data', header=None, encoding='utf-8')
print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Petal length (cm)')
plt.legend(loc='upper left')
plt.show()