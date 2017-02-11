import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import perceptron as pt

"""
Implementing AND function in a perceptron
"""

samples = [[-1, -1, -1], [-1, 1, -1], [1, -1, -1], [1, 1, -1]]
df = pd.DataFrame(samples)
y = df.iloc[0:4, 2].values
X = df.iloc[0:4, [0,1]].values

perceptron = pt.Perceptron(eta=.1, n_iter=10)
perceptron.w_ = [.5, 0, 1]

perceptron.fit_1_iter(X, y)
