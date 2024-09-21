import numpy as np
from scipy.special import softmax

class Estimator:
    def __init__(self, *inputs):
        self.n = np.zeros(inputs)     
        self.v = np.ones(inputs) / inputs[0]

    def train(self, value=1, *inputs):
        self.n[inputs] += 1
        self.v[inputs] += (value -self.v[inputs]) / np.sum(self.n[inputs[:-1]])

    def predict(self, *inputs):
        return self.v[inputs]


class Model:
    def __init__(self, S,A, tModel=None, rModel=None):
        self.S = S
        self.A = A
        self.N = 0

        self.t = tModel if tModel is not None else Estimator(len(self.S), len(self.A) ,len(self.S)) 
        self.r = rModel if rModel is not None else Estimator(len(self.S), len(self.A) ,len(self.S))

    def learn(self, s,a,s_,r):
        self.N += 1 
        self.t.train(1, s,a,s_)
        self.r.train(r, s,a,s_)
    
    def sample(self, s, a): 
        # s_ = np.random.choice(len(self.S), p=softmax(self.t.predict(s,a)))
        s_ = np.random.choice(len(self.S), p=self.t.predict(s,a))
        r = self.r.predict(s,a,s_)
        return s_,r
        
    def T(self, s,a,s_):
        return self.t.predict(s,a,s_)
    def R(self, s,a,s_):
        return self.r.predict(s,a,s_)


class Dyna:
    def __init__(self, model, n=100, alpha=.9, gamma=.9, epsilon=.1):
        self.model = model
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((self.model.S.size, self.model.A.size))
        self.hist = np.zeros((self.model.S.size, self.model.A.size))

    def run(self,s,a):
        self.hist[s,a] += 1

        Ss = np.random.choice([i for i,v in enumerate(np.sum(self.hist, axis=1)) if v>0], self.n) # n random seen states 
        for s in Ss:
            a = np.random.choice([i for i,v in enumerate(self.hist[s]) if v>0]) # a random taken action in state s
            s_,r = self.model.sample(s,a)
            self.Q[s,a] += self.alpha*(r + self.gamma*np.max(self.Q[s_]) - self.Q[s,a]) 

    def v(self):
        return np.max(self.Q, axis=1)
    def pi(self):
        return np.argmax(self.Q, axis=1)