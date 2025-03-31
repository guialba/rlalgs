import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from pmbrl.data import Experiment_Data 


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
        mse_s = super().forward(s, y)
        triplet = torch.nn.functional.triplet_margin_loss(p, positive, negative)
        loss = triplet + mse_s
        return (
            loss.float(),
            {'triplet': triplet.item(), 'mse_s': mse_s.item()}
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
class Model():
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