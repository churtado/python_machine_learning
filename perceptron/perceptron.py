import numpy as np
import numpy as np
import pandas as pd

class Perceptron(object):

    """
    eta = learning rate
    we can specify the weights if we want
    """
    def __init__(self, eta=0.01, weights=[]):
        self.eta = eta
        self.n_iter = n_iter
        self.errors_ =[]

    # we say how many iterations we want it to go through for the learning process
    def fit(self, X, y, n_iter=1):

        #self.w_ = np.zeros(1 + X.shape[1])
        #self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target, in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

samples = [[-1, -1, -1], [-1, 1, -1], [1, -1, -1], [1, 1, -1]]
df = pd.DataFrame(samples)
y = df.iloc[0:4, 2].values
X = df.iloc[0:4, [0,1]].values

#print(X)
#print(y)

perceptron = Perceptron(eta=.1)
