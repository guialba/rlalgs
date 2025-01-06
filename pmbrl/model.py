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

class Normalizer:
    def __init__(self, s,a,p=None, output=2):
        self.inputs_dimensions = {'s':s, 'a':a, 'p':p} if p else {'s':s, 'a':a}
        self.inputs = {
            'all' : ['s','a','s','a'],
            'param' : ['s','a','s'],
            'state' : ['s','a','p'],
        }

        self.normalizer_params = {
            k: {
                "min": torch.zeros(v),
                'max': torch.zeros(v)
            }
            for k,v in self.inputs_dimensions.items()
        }

        self.normalizer_input = {
            "min": torch.zeros(sum([s,a,p])) if p else torch.zeros(sum([s,a])),
            'max': torch.zeros(sum([s,a,p])) if p else torch.zeros(sum([s,a]))
        }
        self.normalizer_output = {
            "min": torch.zeros(output),
            'max': torch.zeros(output)
        }
    
    def setNormalizationParams(self, mode='all'):
        self.normalizer_input = {
            'min': torch.concat([self.normalizer_params[i]['min'] for i in self.inputs[mode]], axis=0),
            'max': torch.concat([self.normalizer_params[i]['max'] for i in self.inputs[mode]], axis=0)
        }

    def updateNormalizationParams(self, x,y, mode='all'):
        offset=0
        pointer=0
        for i in self.inputs[mode]:
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

class RewardEstimatorBase(nn.Module):
    def __init__(self, s,a, hidden_size, output=1):
        super(RewardEstimatorBase, self).__init__()
        self.input_set = sum([s])

        self.l1 = nn.Linear(self.input_set, output)
        # self.l1 = nn.Linear(self.input_set, hidden_size)
        # self.l2 = nn.Linear(hidden_size, output)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        # out = self.l2(out)
        out = self.sig(out)
        return out
class TransitionEstimatorBase(nn.Module):
    def __init__(self, s,a, hidden_size, output):
        super(TransitionEstimatorBase, self).__init__()
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
                learning_rate = [0.001, 0.001],
                criterion_clss = [nn.MSELoss, nn.BCELoss],
                optimizer_clss = [optim.Adam, optim.Adam]
            ) -> None:
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output = output
        self.learning_rate = learning_rate

        self.estimatorT = TransitionEstimatorBase(*input_size, hidden_size, output)
        self.estimatorR = RewardEstimatorBase(*input_size, hidden_size)
        
        self.criterionT = criterion_clss[0]()
        self.criterionR = criterion_clss[1]()
        self.optimizerT = optimizer_clss[0](self.estimatorT.parameters(), lr=learning_rate[0])
        self.optimizerR = optimizer_clss[1](self.estimatorR.parameters(), lr=learning_rate[1])

        self.normalizer = Normalizer(*input_size, output=output)

    def train(self, X_train, y_train, num_epochs=100, debug=False, mode='all'):
        self.normalizer.updateNormalizationParams(X_train, y_train[:,:-1], mode)
        self.normalizer.setNormalizationParams(mode)
        for epoch in range(num_epochs):
            # Training Transition Estimator
            # outputs = self.estimatorT(X_train, mode)
            outputs = self.estimatorT(self.normalizer.normilize(X_train, self.normalizer.normalizer_input), mode)
            # loss = self.criterionT(outputs, y_train[:,:-1])
            loss = self.criterionT(outputs, self.normalizer.normilize(y_train[:,:-1], self.normalizer.normalizer_output))

            self.optimizerT.zero_grad()
            loss.backward()
            self.optimizerT.step()  

            if debug and (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Transition Loss: {loss.item():.4f}')  

            # Training Reward Estimator
            outputs = self.estimatorR(X_train[:,3:-1])
            # loss = self.criterionR(outputs.squeeze(), y_train[:,-1])
            try:
                loss = self.criterionR(outputs.squeeze(), y_train[:,-1])
            except Exception as e:
                raise e

            self.optimizerR.zero_grad()
            loss.backward()
            self.optimizerR.step()  

            if debug and (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Reward Loss: {loss.item():.4f}')  

    def sample(self, x, mode='all'):
        self.normalizer.setNormalizationParams(mode)
        s,r = None, None
        with torch.no_grad():
            # s = self.estimatorT(x, mode)
            s = self.estimatorT(self.normalizer.normilize(x, self.normalizer.normalizer_input), mode)
            r = self.estimatorR(x[:,:self.input_size[0]])
        # return s,r
        return self.normalizer.denormilize(s, self.normalizer.normalizer_output),r
    
"""
Parameterized Model Based Reinforcement Learning (PMBRL):
Modelo amostral de transição:
    input: S, A, P (parâmetro)
    outuput: S_
"""

class TransitionEstimator(nn.Module):
    def __init__(self, s,a,p, hidden_size, output):
        super(TransitionEstimator, self).__init__()
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
                learning_rate = [0.001, 0.001],
                criterion_clss = [nn.MSELoss, nn.BCELoss],
                optimizer_clss = [optim.Adam, optim.Adam]
            ) -> None:
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output = output
        self.learning_rate = learning_rate

        self.estimatorT = TransitionEstimator(*input_size, hidden_size, output)
        self.estimatorR = RewardEstimatorBase(input_size[0],input_size[1], hidden_size)
        
        self.criterionT = criterion_clss[0]()
        self.criterionR = criterion_clss[1]()
        self.optimizerT = optimizer_clss[0](self.estimatorT.parameters(), lr=learning_rate[0])
        self.optimizerR = optimizer_clss[1](self.estimatorR.parameters(), lr=learning_rate[1])

        self.normalizer = Normalizer(*input_size, output=output)

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
        self.normalizer.updateNormalizationParams(X_train, y_train[:,:-1], mode)
        self.normalizer.setNormalizationParams(mode)
        # Training loop
        for epoch in range(num_epochs):
            # Training Transition Estimator
            outputs = self.estimatorT(self.normalizer.normilize(X_train, self.normalizer.normalizer_input), mode)
            loss = self.criterionT(outputs, self.normalizer.normilize(y_train[:,:-1], self.normalizer.normalizer_output))

            self.optimizerT.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.estimatorT.parameters(), max_norm=1.0) 
            self.optimizerT.step()  

            if debug and (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Transition Loss: {loss.item():.4f}')  

            # Training Reward Estimator
            if mode == 'all':
                outputs = self.estimatorR(X_train[:,3:-1])
                # loss = self.criterionR(outputs.squeeze(), y_train[:,-1])
                try:
                    loss = self.criterionR(outputs.squeeze(), y_train[:,-1])
                except Exception as e:
                    raise e

                self.optimizerR.zero_grad()
                loss.backward()
                self.optimizerR.step()  

                if debug and (epoch+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Reward Loss: {loss.item():.4f}')  
            else:
                print(f'Trainning Reward model is not implemented for {mode} mode')

    def sample(self, x, mode='all'):
        """
            x: Input data, its format depends on the mode. It can be (s,a,s',a'), (s,a,s'), or (s',a',p)
            mode: one of the oprions:
                'all' - (Default) Uses the set of inputs (s, a, s', a') for estimating the parameters and using it to estimate the state (s'')
                'param' - Uses the set of inputs (s, a, s') for estimating only the parameters (p)
                'state' - Uses the set of inputs (s', a', p) for estimating only the state (s'')
        """
        self.normalizer.setNormalizationParams(mode)
        s,r = None, None
        with torch.no_grad():
            s = self.estimatorT(self.normalizer.normilize(x, self.normalizer.normalizer_input), mode)
            r = self.estimatorR(x[:,:self.input_size[0]])
        return self.normalizer.denormilize(s, self.normalizer.normalizer_output), r