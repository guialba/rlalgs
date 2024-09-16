import numpy as np

class Estimator:
    def __init__(self, *inputs):
        self.n = np.zeros(inputs)     
        self.v = np.zeros(inputs) 

    def train(self, value=1, *inputs):
        self.n[inputs] += 1
        self.v[inputs] += (value -self.v[inputs]) / np.sum(self.n[inputs[:-1]])

    def forward(self, *inputs):
        return self.v[inputs]
