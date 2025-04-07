import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import pandas as pd

from pmbrl.data import Experiment_Data, get_data_expanded 


## Loss Functions
class Regularized_Reference_Loss(nn.MSELoss):
    def forward(self, s:torch.Tensor, y:torch.Tensor, p:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor,
                alpha=1,
                beta=1,
                gamma=1,
                lambda_=0.001, 
                *args, **kargs
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mse_s_ = torch.mean(f.mse_loss(s,y, reduction='none'), axis=0)
        mse_positive = torch.mean(f.mse_loss(p,positive, reduction='none'), axis=0)
        mse_negative = torch.mean(f.mse_loss(p,negative, reduction='none'), axis=0)
        
        # mse_s_ = super().forward(s, y)
        # mse_positive = super().forward(p, positive)
        # mse_negative = super().forward(p, negative)
        regularize = torch.mean(torch.pow(p, 2) + torch.pow(positive, 2) + torch.pow(negative, 2))

        loss = alpha*mse_positive.sum() + beta*(-mse_negative.sum()) + gamma*mse_s_.sum() + lambda_*regularize
        return (
            loss.float(),
            {
                'mse_positive': mse_positive.tolist(), 'mse_negative': mse_negative.tolist(), 'mse_s': mse_s_.tolist(), 'regularize': regularize.item(),
            }
        )

class Relative_Reference_Loss(nn.MSELoss):
    def forward(self, s:torch.Tensor, y:torch.Tensor, p:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor,
                _s:torch.Tensor, s_positive:torch.Tensor, s_negative:torch.Tensor, _s_negative:torch.Tensor,
                *args, **kargs
    )-> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mse_s = super().forward(s, y)
        mse_positive = super().forward(p, positive)
        mse_negative = super().forward(p, negative)
        mse_s_positive = super().forward(s, s_positive)
        mse_s_negative = super().forward(s, s_negative)
        mse_s_negative_ = super().forward(_s, _s_negative)

        loss = torch.mean(mse_s + mse_positive + (-mse_negative/(mse_s_negative-mse_s_negative_)))
        return (
            loss.float(),
            {'mse_positive': mse_positive.item(), 'inverse_mse_negative': -mse_negative.item(), 'difference_mse_negative_step': (mse_s_negative-mse_s_negative_).item(), 'mse_s': mse_s.item()}
        )
    
class Triplet_Loss(nn.MSELoss):
    def forward(self, s:torch.Tensor, y:torch.Tensor, p:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor, *args, **kargs)-> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # mse_s = super().forward(s, y)
        triplet = f.triplet_margin_loss(p, positive, negative)
        mse_s = torch.mean(f.mse_loss(s,y, reduction='none'), axis=0)
        loss = triplet + mse_s.sum()
        return (
            loss.float(),
            {'triplet': triplet.item(), 'mse_s': mse_s.tolist()}
        )

## Estimators
class Reward_Estimator_Base(nn.Sequential):
    def __init__(self, input_size:int, output_size:int):
        super(Reward_Estimator_Base, self).__init__(
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )

class State_Estimator(nn.Sequential):
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(State_Estimator, self).__init__(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

class Param_Estimator(nn.Sequential):
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(Param_Estimator, self).__init__(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
class Transition_Estimator(nn.Module):
    def __init__(self, param_layer_params:tuple[int], state_layer_params:tuple[int]):
        super(Transition_Estimator, self).__init__()
        self.param_layer = Param_Estimator(*param_layer_params)
        self.state_layer = State_Estimator(*state_layer_params)

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor]:
        param_inputs = x[:, :-1]
        param = self.param_layer(param_inputs.float())
        state_inputs = torch.concat([x[:, 5:], param], dim=1)
        out = self.state_layer(state_inputs.float())
        return out, param
    
## Model
class Base_Line_Simple_Model():
    inference_schema = {'transition': ['s'], 'reward': ['r']}
    features_lables = ['s_0','s_1','s_2','s_3', 'a_']
    inference_features_lables = ['s_0','s_1','s_2','s_3', 'a_']
    targets_lables = ['s0','s1','s2','s3', 'r']

    grouped_features_lables = ['s', 'a']
    grouped_targets_lables = ['estimated_s', 'estimated_r']

    def __init__(self,
                transition_learning_rate:float=0.001,
                transition_estimator_clss=State_Estimator,
                transition_criterion_clss=nn.MSELoss,
                transition_optimizer_clss=optim.Adam,
                reward_learning_rate:float=0.001,
                reward_estimator_clss=Reward_Estimator_Base,
                reward_criterion_clss=nn.MSELoss, #nn.BCELoss,
                reward_optimizer_clss=optim.Adam
            ) -> None:
        ## Transition Estimator Set Up
        self.transition_estimator = transition_estimator_clss((4+1), 10, 4)
        self.transition_criterion = transition_criterion_clss(reduction='none')
        self.transition_optimizer = transition_optimizer_clss(self.transition_estimator.parameters(), lr=transition_learning_rate)
        ## Reward Estimator Set Up
        self.reward_estimator = reward_estimator_clss(4,1)
        self.reward_criterion = reward_criterion_clss()
        self.reward_optimizer = reward_optimizer_clss(self.reward_estimator.parameters(), lr=reward_learning_rate)


    def get_features_targets(self, data:pd.DataFrame)-> tuple[torch.Tensor, torch.Tensor]:
        data = data.copy()
        features = data[['s', 'a', 's_', 'a_', 'p']].copy()
        target = data[['s__', 'r', 'p']].copy()
        features = get_data_expanded(features, {'s_':['s_0','s_1','s_2','s_3']})
        target = get_data_expanded(target, {'s__':['s0','s1','s2','s3']})
        np_features = np.array(features[self.features_lables].values, dtype = np.float32)
        np_targets = np.array(target[self.targets_lables].values, dtype = np.float32)
        return (
            torch.tensor(np_features), 
            torch.tensor(np_targets)
        )


    def prepare_inference_data(self)-> tuple[torch.Tensor, list[str]]:
        pass

    def train(self, X:torch.Tensor, y:torch.Tensor, num_epochs:int=100, **kargs):
        for _ in range(num_epochs):
            # Transition
            self.transition_optimizer.zero_grad()
            transition_outputs = self.transition_estimator(X)
            # mse_loss = torch.mean(f.mse_loss(transition_outputs, y[:,Experiment_Data.targets_slices['s']], reduction='none'), axis=0)
            mse_loss = torch.mean(self.transition_criterion(
                transition_outputs[0], 
                y[:,Experiment_Data.targets_slices['s']]
            ), axis=0)
            transition_loss = mse_loss.sum()
            transition_loss.backward(retain_graph=True)
            # transition_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.transition_estimator.parameters(), max_norm=1.0) 
            self.transition_optimizer.step()  

            # Transition
            self.reward_optimizer.zero_grad()
            reward_outputs = self.reward_estimator(transition_outputs.detach())
            reward_loss = self.reward_criterion(reward_outputs, y[:,Experiment_Data.targets_slices['r']])
            reward_loss.backward()
            self.reward_optimizer.step()  
            
            yield (
                (transition_outputs,), transition_loss, {"mse_s": mse_loss.tolist()},
                reward_outputs, reward_loss
            )

    def sample(self, x:torch.Tensor):
        with torch.no_grad():
            s = self.transition_estimator(x)
            r = self.reward_estimator(s[0])
        return (s,),r

class Base_Line_Model():
    inference_schema = {'transition': ['s'], 'reward': ['r']}
    features_lables = ['s0','s1','s2','s3', 'a', 's_0','s_1','s_2','s_3', 'a_']
    inference_features_lables = ['s0','s1','s2','s3', 'a', 's_0','s_1','s_2','s_3', 'a_']
    targets_lables = ['s0','s1','s2','s3', 'r']

    grouped_features_lables = ['s', 'a', 's_', 'a_']
    grouped_targets_lables = ['estimated_s', 'estimated_r']

    def __init__(self,
                transition_learning_rate:float=0.001,
                transition_estimator_clss=State_Estimator,
                transition_criterion_clss=nn.MSELoss,
                transition_optimizer_clss=optim.Adam,
                reward_learning_rate:float=0.001,
                reward_estimator_clss=Reward_Estimator_Base,
                reward_criterion_clss=nn.MSELoss, #nn.BCELoss,
                reward_optimizer_clss=optim.Adam
            ) -> None:
        ## Transition Estimator Set Up
        self.transition_estimator = transition_estimator_clss((4+1+4+1), 10, 4)
        self.transition_criterion = transition_criterion_clss(reduction='none')
        self.transition_optimizer = transition_optimizer_clss(self.transition_estimator.parameters(), lr=transition_learning_rate)
        ## Reward Estimator Set Up
        self.reward_estimator = reward_estimator_clss(4,1)
        self.reward_criterion = reward_criterion_clss()
        self.reward_optimizer = reward_optimizer_clss(self.reward_estimator.parameters(), lr=reward_learning_rate)


    def get_features_targets(self, data:pd.DataFrame)-> tuple[torch.Tensor, torch.Tensor]:
        data = data.copy()
        features = data[['s', 'a', 's_', 'a_', 'p']].copy()
        target = data[['s__', 'r', 'p']].copy()
        features = get_data_expanded(features, {'s':['s0','s1','s2','s3'], 's_':['s_0','s_1','s_2','s_3']})
        target = get_data_expanded(target, {'s__':['s0','s1','s2','s3']})
        np_features = np.array(features[self.features_lables].values, dtype = np.float32)
        np_targets = np.array(target[ self.targets_lables].values, dtype = np.float32)
        return (
            torch.tensor(np_features), 
            torch.tensor(np_targets)
        )


    def prepare_inference_data(self)-> tuple[torch.Tensor, list[str]]:
        pass

    def train(self, X:torch.Tensor, y:torch.Tensor, num_epochs:int=100, **kargs):
        for _ in range(num_epochs):
            # Transition
            self.transition_optimizer.zero_grad()
            transition_outputs = self.transition_estimator(X[:,Experiment_Data.features_slices['model_ready']])
            # mse_loss = torch.mean(f.mse_loss(transition_outputs, y[:,Experiment_Data.targets_slices['s']], reduction='none'), axis=0)
            mse_loss = torch.mean(self.transition_criterion(
                transition_outputs[0], 
                y[:,Experiment_Data.targets_slices['s']]
            ), axis=0)
            transition_loss = mse_loss.sum()
            transition_loss.backward(retain_graph=True)
            # transition_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.transition_estimator.parameters(), max_norm=1.0) 
            self.transition_optimizer.step()  

            # Transition
            self.reward_optimizer.zero_grad()
            reward_outputs = self.reward_estimator(transition_outputs.detach())
            reward_loss = self.reward_criterion(reward_outputs, y[:,Experiment_Data.targets_slices['r']])
            reward_loss.backward()
            self.reward_optimizer.step()  
            
            yield (
                (transition_outputs,), transition_loss, {"mse_s": mse_loss.tolist()},
                reward_outputs, reward_loss
            )

    def sample(self, x:torch.Tensor):
        with torch.no_grad():
            s = self.transition_estimator(x)
            r = self.reward_estimator(s[0])
        return (s,),r

class Model():
    inference_schema = {'transition': ['s', 'p'], 'reward': ['r']}
    features_lables = [
            's0','s1','s2','s3', 'a', 's_0','s_1','s_2','s_3', 'a_', 
            'positive_s0','positive_s1','positive_s2','positive_s3', 'positive_a', 'positive_s_0','positive_s_1','positive_s_2','positive_s_3',
            'negative_s0','negative_s1','negative_s2','negative_s3', 'negative_a', 'negative_s_0','negative_s_1','negative_s_2','negative_s_3'
        ]
    inference_features_lables = ['s0','s1','s2','s3', 'a', 's_0','s_1','s_2','s_3', 'a_']
    targets_lables = ['s0','s1','s2','s3', 'r', 'p0','p1', 'positive_p0','positive_p1', 'negative_p0','negative_p1']

    grouped_features_lables = ['s', 'a', 's_', 'a_', 'positive_s', 'positive_s_', 'negative_s', 'negative_']
    grouped_targets_lables = ['estimated_s', 'estimated_p', 'estimated_r']

    def __init__(self,
                transition_learning_rate:float=0.001,
                transition_estimator_clss=Transition_Estimator,
                transition_criterion_clss=Regularized_Reference_Loss, #nn.MSELoss,
                transition_optimizer_clss=optim.Adam,
                reward_learning_rate:float=0.001,
                reward_estimator_clss=Reward_Estimator_Base,
                reward_criterion_clss=nn.MSELoss, #nn.BCELoss,
                reward_optimizer_clss=optim.Adam
            ) -> None:
        ## Transition Estimator Set Up
        self.transition_estimator = transition_estimator_clss(
            param_layer_params=(4+1+4, 10, 2), 
            state_layer_params=(4+1+2, 10, 4)
        )
        self.transition_criterion = transition_criterion_clss()
        self.transition_optimizer = transition_optimizer_clss(self.transition_estimator.parameters(), lr=transition_learning_rate)
        ## Reward Estimator Set Up
        self.reward_estimator = reward_estimator_clss(4,1)
        self.reward_criterion = reward_criterion_clss()
        self.reward_optimizer = reward_optimizer_clss(self.reward_estimator.parameters(), lr=reward_learning_rate)
        self.history = None

    def search(self, anchor:pd.Series) -> np.array:
        df = self.history[self.history['a'] == anchor['a']]#.reset_index(drop=True)
        df = get_data_expanded(df,
            {
                'p':['p0','p1'],
                's':['s0','s1','s2','s3'],
                's_':['s_0','s_1','s_2','s_3'],
            }
        )
        # distance
        df['p_distance'] = np.sqrt(np.pow(df.p0 - tuple(anchor['p'])[0],2)) + np.sqrt(np.pow(df.p1 - tuple(anchor['p'])[1],2))

        # get references
        negative_filtered = df[df['p_distance'] > df.p_distance.min()].reset_index(drop=True)
        positive_index = df.p_distance.idxmin()
        negative_index = np.random.randint(negative_filtered.shape[0])  
        positive_reff = df.loc[positive_index]
        negative_reff = negative_filtered.loc[negative_index]

        return np.concat([positive_reff[['s','a','s_', 'p']].values, negative_reff[['s','a','s_', 'p']].values])

    def get_features_targets(self, data:pd.DataFrame)-> tuple[torch.Tensor, torch.Tensor]:
        self.history = data if self.history is None else pd.concat([self.history, data])
        data = data.copy()
        features = data[['s', 'a', 's_', 'a_', 'p']].copy()
        target = data[['s__', 'r', 'p']].copy()

        # get refferences 
        features[[
            'positive_s','positive_a','positive_s_','positive_p',
            'negative_s','negative_a','negative_s_','negative_p'
        ]] = features.apply(lambda row: self.search(row), axis=1, result_type='expand')

        features = get_data_expanded(features,
            {
                's':['s0','s1','s2','s3'], 's_':['s_0','s_1','s_2','s_3'],
                'positive_s':['positive_s0','positive_s1','positive_s2','positive_s3'],
                'positive_s_':['positive_s_0','positive_s_1','positive_s_2','positive_s_3'],
                'negative_s':['negative_s0','negative_s1','negative_s2','negative_s3'],
                'negative_s_':['negative_s_0','negative_s_1','negative_s_2','negative_s_3'],
                'positive_p':['positive_p0','positive_p1'], 'negative_p':['negative_p0','negative_p1']
            }
        )
        target = get_data_expanded(target, {'s__':['s0','s1','s2','s3'], 'p':['p0','p1']})
        target[['positive_p0','positive_p1', 'negative_p0','negative_p1']] = features[['positive_p0','positive_p1', 'negative_p0','negative_p1']]
        features.drop('p', axis=1)
        np_features = np.array(features[self.features_lables].values, dtype = np.float32)
        np_targets = np.array(target[self.targets_lables].values, dtype = np.float32)
        return (
            torch.tensor(np_features), 
            torch.tensor(np_targets)
        )

    def train(self, X:torch.Tensor, y:torch.Tensor, num_epochs:int=100, **kargs):
        for _ in range(num_epochs):
            # Transition
            self.transition_optimizer.zero_grad()
            transition_outputs = self.transition_estimator(X[:,Experiment_Data.features_slices['model_ready']])
            transition_loss, loss_components = self.transition_criterion(
                s = transition_outputs[0], 
                y = y[:,Experiment_Data.targets_slices['s']],
                p = transition_outputs[1],
                positive = y[:,Experiment_Data.targets_slices['positive_p']],
                negative = y[:,Experiment_Data.targets_slices['negative_p']],
                _s = X[:,Experiment_Data.features_slices['s_']],
                s_positive = X[:,Experiment_Data.features_slices['positive_s']],
                s_negative = X[:,Experiment_Data.features_slices['negative_s']],
                _s_negative = X[:,Experiment_Data.features_slices['negative_s_']],
                **kargs
            )
            self.transition_optimizer.zero_grad()
            transition_loss.backward(retain_graph=True)
            # transition_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.transition_estimator.parameters(), max_norm=1.0) 
            self.transition_optimizer.step()  

            # Transition
            self.reward_optimizer.zero_grad()
            reward_outputs = self.reward_estimator(transition_outputs[0].detach())
            reward_loss = self.reward_criterion(reward_outputs, y[:,Experiment_Data.targets_slices['r']])
            reward_loss.backward()
            self.reward_optimizer.step()  
            
            yield (
                transition_outputs, transition_loss, loss_components,
                reward_outputs, reward_loss
            )

    def sample(self, x:torch.Tensor):
        with torch.no_grad():
            s = self.transition_estimator(x)
            r = self.reward_estimator(s[0])
        return s,r



if __name__=="__main__":
    x = torch.tensor([
        [1.,2.,3.,4.,5.,6.,7.,8.,9.,0.], 
        [10.,20.,30.,40.,50.,60.,70.,80.,90.,00.]
    ])

    se = State_Estimator(10, 20, 2)
    pe = Param_Estimator(10, 20, 2)
    te = Transition_Estimator((9, 20, 2), (7, 20, 4))

    print(type(se), se(x))
    print(type(pe), pe(x))
    print(type(te), te(x))