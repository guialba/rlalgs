import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Estimator(nn.Module):
    def __init__(self, s,a,p, hidden_size, output):
        super(Estimator, self).__init__()
        self._s = s
        self._a = a
        self._p = p

        self.input_param = sum([s,a,s])
        self.input_state = sum([s,a,p])
        self.inputs_format = {
            'all': self.input_param+a,
            'param': self.input_param,
            'state': self.input_state
        }

        self.par1 = nn.Linear(self.input_param, hidden_size)
        self.par2 = nn.Linear(hidden_size, p)
        self.s1 = nn.Linear(self.input_state, hidden_size)
        self.s2 = nn.Linear(hidden_size, output)
        self.relu = nn.ReLU()
        self.than = nn.Tanh()

    def forward_param(self, x):
        """
            x: Input data of format (s, a, s') for estimating only the parameters (p)
        """
        out = self.par1(x)
        out = self.relu(out)
        out = self.par2(out)
        return out
    
    def forward_state(self, x):
        """
            x: Input data of format (s', a', p) for estimating only the state (s'')
        """
        out = self.s1(x)
        out = self.than(out)
        out = self.s2(out)
        return out

    def forward(self, x, mode='all'):
        """
            x: Input data
            mode: one of the oprions:
                'all' - Uses the set of inputs (s, a, s', a') for estimating the parameters and using it to estimate the state (s'')
                'param' - Uses the set of inputs (s, a, s') for estimating only the parameters (p)
                'state' - Uses the set of inputs (s', a', p) for estimating only the state (s'')
        """
        assert x.shape[1] == self.inputs_format[mode], f"Invalid input format ({x.shape[1]}) for mode {mode}, expected ({self.inputs_format[mode]})"
        if mode in ('all', 'param'):
            out = self.forward_param(x[:,:self.input_param])
        if mode == 'all':
            input_state = self.input_param - self._s
            x = torch.concat([x[:,input_state:], out], axis=1)
        if mode in ('all', 'state'):
            out = self.forward_state(x)
        return out


class Model():
    def __init__(self,
                input_size = (2, 1, 2), # 2d for s and s_, 1d for a and a_, and 2d for p,
                hidden_size = 10,
                output = 2, #2d for s_
                learning_rate = 0.001,
                criterion_clss = nn.MSELoss,
                optimizer_clss = optim.Adam
            ) -> None:
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output = output
        self.learning_rate = learning_rate

        self.estimator = Estimator(*input_size, hidden_size, output)
        
        self.criterion = criterion_clss()
        self.optimizer = optimizer_clss(self.estimator.parameters(), lr=learning_rate)

    def train(self, X_train, y_train, num_epochs=100, debug=False, mode='all'):
        """
            X_train: Training dataset, its format depends on the mode. It can be (s,a,s',a'), (s,a,s'), or (s',a',p)
            y_train: Test dataset, its format depends on the mode. It can be (s,a,s',a'), (s,a,s'), or (s',a',p)
            num_epochs: number of epochs to train (default is 100)
            debug: True or False to show debug message for every completed epoch (default is False)
            mode: one of the oprions:
                'all' - (Default) Uses the set of inputs (s, a, s', a') for estimating the parameters and using it to estimate the state (s'')
                'param' - Uses the set of inputs (s, a, s') for estimating only the parameters (p)
                'state' - Uses the set of inputs (s', a', p) for estimating only the state (s'')
        """
        for epoch in range(num_epochs):
            outputs = self.estimator(X_train, mode)
            loss = self.criterion(outputs, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  

            if debug and (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  

    def predict(self, x, mode='all'):
        """
            x: Input data, its format depends on the mode. It can be (s,a,s',a'), (s,a,s'), or (s',a',p)
            mode: one of the oprions:
                'all' - (Default) Uses the set of inputs (s, a, s', a') for estimating the parameters and using it to estimate the state (s'')
                'param' - Uses the set of inputs (s, a, s') for estimating only the parameters (p)
                'state' - Uses the set of inputs (s', a', p) for estimating only the state (s'')
        """
        value = None
        with torch.no_grad():
            value = self.estimator(x, mode)
        return value