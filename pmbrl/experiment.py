import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pmbrl.model import *
from pmbrl.data import *

class Execution:
    def __init__(self, agents_param, data_param, **others):
        self.agents_param = agents_param
        self.data_param = data_param
        self.others = others

        self.agents = {
            name : agent['class'](learning_rate=agent['learning_rate']) # Instantiate the Agent
            for name, agent in agents_param.items()
        }

        self.data = generate_episode()
        

    def run(self, training_param):
        X, y = build_train_data(self.data)
        err = {}

        for name, agent in self.agents.items():
            agent.train(X[:-1], y[:-1], **training_param)
            s,r = agent.sample(X[-1:], mode=training_param['mode'])
            s_err = torch.mean((s - y[-1:,:-1])**2, axis=0)**.5
            r_err = torch.mean((r - y[-1:,-1])**2, axis=0)**.5
            err[name] = (s_err, r_err)

        return err
