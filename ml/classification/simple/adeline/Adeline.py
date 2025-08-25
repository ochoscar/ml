import numpy as np


class AdelineGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = np.float64(0.)
        self.errors = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b += self.eta * 2.0 * errors.mean()
            error = (errors**2).mean()
            self.errors.append(error)
        return self

    def net_input(self, X):
        return np.dot(X, self.w) + self.b
    
    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)