import numpy as np
import pandas as pd

class Perceptron(object):

    """
    eta = learning rate
    we can specify the weights if we want
    """
    def __init__(self, eta=0.01, weights=[]):
        self.eta = eta
        self.errors_ =[]
        self.w_ = weights

    # we say how many iterations we want it to go through for the learning process
    def fit(self, X, y, n_iter=1):

        for _ in range(n_iter):
            errors = 0
            for xi, target, in zip(X, y):
                predicted_value = self.predict(xi)
                update = self.eta * (target - predicted_value)
                self.w_ += update * xi
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# We'll implement the logical AND function with a Perceptron

samples = pd.DataFrame([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
biases = pd.DataFrame([[1.0], [1.0], [1.0], [1.0]])
X = pd.concat([biases, samples], axis=1).values

y = pd.DataFrame([[-1.0], [-1.0], [-1.0], [1.0]]).iloc[0:4, 0].values

perceptron = Perceptron(eta=.1, weights=[.5, 0.0, 1.0])
print(perceptron.w_)

perceptron.fit(X, y, 3)

#It takes 3 iterations to make the perceptron learn the whole function

print(perceptron.w_)
