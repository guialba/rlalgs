import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

"""
Simple Model Based Reinforcement Learning (MBRL):
Modelo amostral de transição:
    input: S, A
    outuput: S_
"""

class EstimatorBase(nn.Module):
    def __init__(self, s,a, hidden_size, output):
        super(EstimatorBase, self).__init__()
        self.input_set = sum([s,a,s,a])

        self.l1 = nn.Linear(self.input_set, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output)
        self.relu = nn.ReLU()
        self.than = nn.Tanh()

    def forward(self, x, mode='all'):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.than(out)
        out = self.l3(out)
        return out
    
class ModelBase():
    def __init__(self,
                input_size = (2, 1), # 2d for s and s_, 1d for a and a_
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

        self.estimator = EstimatorBase(*input_size, hidden_size, output)
        
        self.criterion = criterion_clss()
        self.optimizer = optimizer_clss(self.estimator.parameters(), lr=learning_rate)

    def train(self, X_train, y_train, num_epochs=100, debug=False, mode='all'):
        for epoch in range(num_epochs):
            outputs = self.estimator(X_train, mode)
            loss = self.criterion(outputs, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  

            if debug and (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  

    def predict(self, x, mode='all'):
        value = None
        with torch.no_grad():
            value = self.estimator(x, mode)
        return value
    
"""
Parameterized Model Based Reinforcement Learning (PMBRL):
Modelo amostral de transição:
    input: S, A, P (parâmetro)
    outuput: S_
"""

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

        s,a,p = input_size
        self.inputs_dimensions = {'s':s, 'a':a, 'p':p}
        self.normalizer_params = {
            k: {
                "min": torch.zeros(v),
                'max': torch.zeros(v)
            }
            for k,v in self.inputs_dimensions.items()
        }

        self.normalizer_input = {
            "min": torch.zeros(sum([*input_size])),
            'max': torch.zeros(sum([*input_size]))
        }
        self.normalizer_output = {
            "min": torch.zeros(output),
            'max': torch.zeros(output)
        }

    def setNormalizationParams(self, mode):
        inputs = {
            'all' : ['s','a','s','a'],
            'param' : ['s','a','s'],
            'state' : ['s','a','p'],
        }

        self.normalizer_input = {
            'min': torch.concat([self.normalizer_params[i]['min'] for i in inputs[mode]], axis=0),
            'max': torch.concat([self.normalizer_params[i]['max'] for i in inputs[mode]], axis=0)
        }

    def updateNormalizationParams(self, x,y, mode):
        inputs = {
            'all' : ['s','a','s','a'],
            'param' : ['s','a','s'],
            'state' : ['s','a','p'],
        }
        offset=0
        pointer=0
        for i in inputs[mode]:
            pointer = offset+self.inputs_dimensions[i]
            temp = torch.concat([self.normalizer_params[i]['max'].reshape(1,-1), self.normalizer_params[i]['min'].reshape(1,-1), x[:,offset:pointer]], axis=0)
            self.normalizer_params[i]['min'] = torch.min(temp, axis=0).values
            # Gambiarra para anular valores inf
            max_mask = torch.max(temp, axis=0).values 
            self.normalizer_params[i]['max'] = torch.where(torch.isinf(max_mask), torch.tensor(0.0), max_mask)
            ###
            offset = pointer

        temp = torch.concat([self.normalizer_output['max'].reshape(1,-1), self.normalizer_output['min'].reshape(1,-1), y], axis=0)
        self.normalizer_output['min'] = torch.min(temp, axis=0).values
        self.normalizer_output['max'] = torch.max(temp, axis=0).values

    def normilize(self, value, params):
        range = params['max'] - params['min']
        return (value - params['min']) / range
    def denormilize(self, value, params):
        range = params['max'] - params['min']
        return value * range + params['min']


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
        self.updateNormalizationParams(X_train, y_train, mode)
        self.setNormalizationParams(mode)
        # Training loop
        for epoch in range(num_epochs):
            outputs = self.estimator(self.normilize(X_train, self.normalizer_input), mode)
            loss = self.criterion(outputs, self.normilize(y_train, self.normalizer_output))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.estimator.parameters(), max_norm=1.0) 
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
        self.setNormalizationParams(mode)
        value = None
        with torch.no_grad():
            value = self.estimator(self.normilize(x, self.normalizer_input), mode)
        return self.denormilize(value, self.normalizer_output)