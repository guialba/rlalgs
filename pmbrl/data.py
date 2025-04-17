from typing import Any, Dict, Iterator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import rlenvs
import pandas as pd
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

default_params = {
    'gravity': 9.8,
    'masscart': 1.0,
    'masspole': 0.1,
    'length': 0.5,  # actually half the pole's length
    'force_mag': 10.0,
    'tau': 0.02,  # seconds between state updates
    # 'total_mass': self.masspole + self.masscart,
    # 'polemass_length': self.masspole * self.length,
}


random_policy = lambda _: int(np.random.choice([0,1], size=1)[0])

def generate_episode(size_limit=100, env=None, policy=None, options=None, seed=None):
    seed = seed or np.random.randint(1000)
    options = options if options else {
        'masspole': round(np.random.rand(), 2),
        'length': np.random.randint(low=0, high=20)/10
    }
    policy = policy if policy else random_policy

    if env is None:
        env = gym.make("custom/DiscreteCartPole-v1")
        
    observation, info = env.reset(seed=seed, options=options)
    hist = []
    while len(hist) < size_limit:
    # for i in range(size_limit):
        action = policy(observation)
        s_,reward, terminated, truncated, info = env.step(action)
        
        d = [round(n, 4) for n in observation[2:].tolist() + [action, reward] + list(options.values()) + s_[2:].tolist()]
        hist.append(np.array(d).flatten())
        
        observation = s_
        if terminated or truncated:
            if len(hist) > 1:
                break
            else:
                # observation, info = env.reset(seed=seed, options=options) 
                observation, info = env.reset(options=options)
                hist = []

    env.close()
    return  np.array(hist)

def build_train_data(data, dimensions=None):
    dimensions = dimensions if dimensions is not None else {
        's': 2, 'a':1, 'r': 1, 'p':2
    }
    iS = dimensions['s']
    iA = iS+dimensions['a']
    iR = iA+dimensions['r']
    iP = iR+dimensions['p']
    iS_ = iP+dimensions['s']

    s,a,r,p,s_ = data[:-1,:iS], data[:-1,iS:iA], data[:-1,iA:iR], data[:-1,iR:iP], data[:-1,iP:iS_] 
    s_,a_,r_,p_,s__ =  data[1:,:iS], data[1:,iS:iA], data[1:,iA:iR], data[1:,iR:iP], data[1:,iP:iS_] 

    input_all = np.concat([s, a, s_, a_], axis=1)
    output_all = np.concat([s__, r_, p], axis=1)

    X = torch.tensor(input_all, dtype=torch.float32)
    y = torch.tensor(output_all, dtype=torch.float32)

    return X, y

def build_data(data, dimensions=None):
    dimensions = dimensions if dimensions is not None else {
        's': 2, 'a':1, 'r': 1, 'p':2
    }
    iS = dimensions['s']
    iA = iS+dimensions['a']
    iR = iA+dimensions['r']
    iP = iR+dimensions['p']
    iS_ = iP+dimensions['s']

    s,a,r,p,s_ = data[:-1,:iS], data[:-1,iS:iA], data[:-1,iA:iR], data[:-1,iR:iP], data[:-1,iP:iS_] 
    s_,a_,r_,p_,s__ =  data[1:,:iS], data[1:,iS:iA], data[1:,iA:iR], data[1:,iR:iP], data[1:,iP:iS_] 

    input_all = np.concat([s, a, s_, a_, s__, r_, p], axis=1)

    return input_all

def train_data(data, random=True, dimensions=None):
    dimensions = dimensions if dimensions is not None else {
        's': 2, 'a':1, 'r': 1, 'p':2
    }

    if random:
        n = data.shape[0]
        index = np.random.choice(n, n, replace=False)  
    else:
        index = np.arange(n)

    dataset = data[index]

    input_index = dimensions['s']*2 + dimensions['a']*2
    input_all = dataset[:,:input_index]
    output_all = dataset[:, input_index:]

    X = torch.tensor(input_all, dtype=torch.float32)
    y = torch.tensor(output_all, dtype=torch.float32)
    return X,y



def generate_hist(n=500, m=10):
    """
        n: total number of steps in any amount of episodes
        m: max number of different params to be used
    """
    # env = gym.make("CartPole-v1")
    # observation, info = env.reset(seed=82)
    env = gym.make("custom/DiscreteCartPole-v1")
    envs = []
    options = {
        'masspole': round(np.random.rand(), 2),
        'length': np.random.randint(low=0, high=20)/10
    }
    envs.append(options)
    observation, info = env.reset(seed=82, options=options)

    hist_s = np.array([observation])
    hist_a = np.array([])
    hist_r = np.array([])
    hist_p = np.array([list(options.values())])
    for i in range(n):
        action = int(np.random.choice([0,1], size=1)[0])
        observation,reward, terminated, truncated, info = env.step(action)
        
        hist_s = np.concat([hist_s, [observation]])
        hist_a = np.concat([hist_a, [action]])
        hist_r = np.concat([hist_r, [reward]])
        hist_p = np.concat([hist_p, [list(options.values())]])

        if terminated or truncated:
            # break
            if len(envs) < m:
                options = {
                    'masspole': round(np.random.rand(), 2),
                    'length': np.random.randint(low=0, high=20)/10
                }
                envs.append(options)
            else:
                options = np.random.choice(envs)
            observation, info = env.reset(seed=i, options=options)
            
    env.close()
    return hist_s, hist_a, hist_r, hist_p

def train_test_split(hist_s, hist_a, hist_r, hist_p, mode='all', p=.3):
    """
        hist_s: Historic data for s 
        hist_a: Historic data for a
        hist_p: Historic data for p 
        mode: mode of dataset (all, param, state) default is all
        p: proportion of test split, default is 0.3
    """
    input_s = hist_s[:-2,2:] # Getting ony two dimensions of state
    input_s_ = hist_s[1:-1,2:] # Getting ony two dimensions of state
    input_s_2 = hist_s[2:,2:] # Getting ony two dimensions of state
    input_a = np.expand_dims(hist_a[:-1], axis=1)
    input_a_ = np.expand_dims(hist_a[1:], axis=1)
    input_r = np.expand_dims(hist_r[:-1], axis=1)
    input_p = hist_p[:-2]
    input_all = np.concat([input_s, input_a, input_s_, input_a_, input_p], axis=1)
    output_all = np.concat([input_s_2, input_r, input_p], axis=1)

    split = int(input_all.shape[0] * (1-p))

    X = torch.tensor(input_all, dtype=torch.float32)
    y = torch.tensor(output_all, dtype=torch.float32)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    if mode=='all':
        X_train, y_train = X_train[:,:-2], y_train[:,:-2]
        X_test, y_test = X_test[:,:-2], y_test[:,:-2]
        return X_train, y_train, X_test, y_test
    elif mode=='param':
        X_train, y_train = X_train[:,:-3], y_train[:,2:]
        X_test, y_test = X_test[:,:-3], y_test[:,2:]
        return X_train, y_train, X_test, y_test
    elif mode=='state':
        X_train, y_train = X_train[:,-5:], y_train[:,:-2]
        X_test, y_test = X_test[:,-5:], y_test[:,:-2]
        return X_train, y_train, X_test, y_test


# class Model_Profile:

def get_data_expanded(data:pd.DataFrame, cols:dict[str,list[str]]) -> pd.DataFrame:
        data_copy = data.copy()
        for col, into in cols.items():
            data_copy[into] = data_copy.apply(lambda row:pd.Series(row[col]), axis=1)
        return data_copy
def get_data_compacted(data:pd.DataFrame, cols:dict[str,list[str]]) -> pd.DataFrame:
        data_copy = data.copy()
        for col, into in cols.items():
            data_copy[col] = list(zip(*[data_copy[c] for c in into]))
        return data_copy


class Experiment_Data:
    features_slices = {
        'model_ready': slice(0,10,1),
        's': slice(0,4,1),
        'a': slice(4,5,1),
        's_': slice(5,9,1),
        'a_': slice(9,10,1),
        'positive_s': slice(10,14,1),
        'positive_a': slice(14,15,1),
        'positive_s_': slice(15,19,1),
        'negative_s': slice(19,23,1),
        'negative_a': slice(23,24,1),
        'negative_s_': slice(24,28,1),
    }
    targets_slices = {
        's': slice(0,4,1),
        'r': slice(4,5,1),
        'p': slice(5,7,1),
        'positive_p': slice(7,9,1),
        'negative_p': slice(9,11,1),
    }

    def load(self, path:str) -> None:
        self.path = path
        data = pd.read_csv(path)
        raw_data = get_data_compacted(data, {'p': ['p0','p1'], 's': ['s0','s1','s2','s3'], 's_': ['s_0','s_1','s_2','s_3']})
        self.raw_data = raw_data[['episode', 'step','s','a','r','s_','p']]
        return self.raw_data

    def save(self, path:str) -> None:
        assert self.raw_data is not None, "No data to export"
        data = get_data_expanded(self.raw_data, {'p': ['p0','p1'], 's': ['s0','s1','s2','s3'], 's_': ['s_0','s_1','s_2','s_3']})

        self.path = path
        data[['episode', 'step','s0','s1','s2','s3','a','r','s_0','s_1','s_2','s_3','p0','p1']].to_csv(path, index=False)

    def generate_episodes(self, n_episodes:int = 100, env:Any = None) -> pd.DataFrame:
        data:list[pd.DataFrame] = [Experiment_Data.episode(env=env, seed=i).assign(episode=i) for i in range(n_episodes)]
        self.raw_data = pd.concat(data)
        return self.raw_data

    def _build_inference_dataset(self, data:pd.DataFrame) -> pd.DataFrame:
        next_steps = data.groupby('episode')[['s','a','r','s_']].shift(-1)
        data[['s_','a_','r_','s__']] = next_steps
        return data[['step', 'episode', 'p',  's', 'a', 'r', 's_', 'a_', 'r_', 's__']].dropna().reset_index(drop=True)

    def build_training_dataset(self, data:pd.DataFrame=None, randomize=True) -> pd.DataFrame:
        if data is None:
            data = self.raw_data.copy()
        else:
            data = data.copy()

        self.training_dataset = self._build_inference_dataset(data)
        
        n = self.training_dataset.shape[0]
        index = np.random.choice(n, n, replace=False) if randomize else np.arange(n)
        self.training_dataset = self.training_dataset.iloc[index]
        return self.training_dataset
    
    def get_training_data(self, trainer:Iterator, inference_schema={'transition': ['s', 'p'], 'reward': ['r']}) -> pd.DataFrame:
        colums = inference_schema['transition'] + inference_schema['reward']
        train_loss, train_loss_terms, train_data  = zip(*[(
            (epoch, transition_loss.tolist(), reward_loss.tolist()), 
            dict({'epoch': epoch}, **transition_loss_terms),
            # ((epoch, transition_etimates[0].tolist(), transition_etimates[1].tolist(), reward_etimates.tolist()) if not base_line else (epoch, transition_etimates.tolist(), reward_etimates.tolist()))
            ((epoch, *[v.tolist() for v in transition_etimates], reward_etimates.tolist()))
            ) 
            for epoch, (transition_etimates, transition_loss, transition_loss_terms, reward_etimates, reward_loss) in enumerate(trainer)
        ])
        
        train_loss = pd.DataFrame(train_loss, columns=['epoch', 'transition', 'reward'])
        train_loss_terms = pd.DataFrame(train_loss_terms)
        train_data = pd.DataFrame(train_data, columns=['epoch']+colums).explode(colums)
        self.training_results = train_loss.join(train_loss_terms, on='epoch', lsuffix='_caller', rsuffix='_other')[train_loss_terms.columns.to_list()+['transition', 'reward']]
        self.training_data = train_data

        return self.training_results
    
    def _predict_from_row(self, row:pd.Series, model:Any) -> tuple[list[float], tuple[float], float]:
            np_features = np.array([row[model.inference_features_lables].values], dtype = np.float32)
                    
            x = torch.tensor(np_features)
            schema = model.inference_schema['transition'] + model.inference_schema['reward']
            s, r = model.sample(x)
            return [
                tuple(round(v, 3) for v in value[0].tolist())
                for lable, value in zip(model.inference_schema['transition'], s)
            ] + [ round(r[0].item(), 1) ]

    def evaluate_model(self, model:Any, n_episodes:int=10, env:Any = None, path:str=None) -> pd.DataFrame:
        if env is None:
            d:list[pd.DataFrame] = [Experiment_Data.episode(env=env, seed=i).assign(episode=i) for i in range(n_episodes)]
            data = pd.concat(d)
        if path is not None:
            raw_data = pd.read_csv(path)
            data = get_data_compacted(raw_data, {'p': ['p0','p1'], 's': ['s0','s1','s2','s3'], 's_': ['s_0','s_1','s_2','s_3']})
            data = data[['episode', 'step','s','a','r','s_','p']]

        self.evaluation_data = self._build_inference_dataset(data)

        expanded_data = get_data_expanded(self.evaluation_data, {
                's': ['s0', 's1', 's2', 's3'],
                's_': ['s_0', 's_1', 's_2', 's_3'],
                's__': ['s__0', 's__1', 's__2', 's__3'],
                'p': ['p0', 'p1']
        })
        self.evaluation_data[model.grouped_targets_lables] = expanded_data.apply(lambda row: self._predict_from_row(row, model), axis=1, result_type='expand')


        return self.evaluation_data

    def get_evaluation_metrics(self, data:pd.DataFrame = None) -> pd.DataFrame:
        data = self.evaluation_data.copy() if data is None else data.copy()
        expansions = {
            's__': ['s0', 's1', 's2', 's3'],
            'estimated_s': ['estimated_s0', 'estimated_s1', 'estimated_s2', 'estimated_s3'],
            'p': ['p0', 'p1'],
            'estimated_p': ['estimated_p0', 'estimated_p1'],
        }

        results = get_data_expanded(data, expansions)

        def rse(v1:pd.Series, v2:pd.Series) -> pd.Series:
            return np.sqrt(np.pow(v1-v2, 2))

        # RMSE for state
        for v1, v2 in zip(expansions['s__'], expansions['estimated_s']):
            results[f'rse_{v1}'] = rse(results[v1], results[v2])

        results[f'rse_r'] = rse(results.r, results.estimated_r)
        return results

    def __add__(self, val):
        self.raw_data = pd.concat([self.raw_data, val.raw_data])
        return self
    def __repr__(self):
        return str(self.raw_data.head())

    @staticmethod
    def episode(env:Any = None, policy:callable = None, options:Dict[str, Any] = None, seed:int = None) -> pd.DataFrame:
        seed = seed or np.random.randint(1000)
        policy = policy or (lambda _: int(np.random.choice([0,1], size=1)[0]))
        env = env or gym.make("custom/CartPole-v1")
        options = options or {
            'masspole': round(np.random.rand(), 2),
            'length': np.random.randint(low=0, high=20)/10
        }
        # Generate episode logic
        s, info = env.reset(seed=seed, options=options)
        data, step = [], 0
        while True:
            a = policy(s)
            s_,r, terminated, truncated, info = env.step(a)
            data.append({'step':step, 's':s.round(3), 'a':a, 'r':r, 's_':s_.round(3), 'p':list(options.values())})
            s = s_
            step += 1
            if terminated or truncated:
                break
        env.close()
        return pd.DataFrame.from_dict(data)

    def plot_episodes_progression(self, axs:axes.Axes) -> axes.Axes:
        random.seed(100)

        # df = self.get_raw_data_expanded()
        df = get_data_expanded(self.raw_data, {'s':['s0','s1','s2','s3'], 's_':['s_0','s_1','s_2','s_3'], 'p':['p_0','p_1']})

        actions = ['<','>']
        actions_colors = ['white','black']

        axs.set_title('Episodes Progression')
        axs.set_xlabel('pole_angle')
        axs.set_ylabel('angular_velocity')

        colors = random.choices(list(mcolors.CSS4_COLORS.values()), k=len(df.episode.unique()))
        for epi in df.episode.unique():
            axs.plot(df[df.episode == epi].s_2, df[df.episode == epi].s_3, color=colors[epi])
            axs.plot(df[df.episode == epi].s2.values[:2], df[df.episode == epi].s3.values[:2], color=colors[epi], label=f'episode {epi}')
            axs.plot(df[df.episode == epi].s2.values[:1], df[df.episode == epi].s3.values[:1], color=colors[epi], 
                    marker=actions[df[df.episode == epi].a.values[0]],
                    markerfacecolor=actions_colors[df[df.episode == epi].a.values[0]]
            )
            for step in df[df.episode == epi].step.unique():
                axs.plot(
                    df[(df.episode == epi) & (df.step == step)].s_2.values[0], 
                    df[(df.episode == epi) & (df.step == step)].s_3.values[0], 
                    color=colors[epi], 
                    marker=actions[df[(df.episode == epi) & (df.step == step)].a.values[0]], 
                    markerfacecolor=actions_colors[df[(df.episode == epi) & (df.step == step)].a.values[0]])
            axs.legend()

        return axs
    
    def plot_episode_progression_with_predictions(self, axs:axes.Axes, episode:int=0) -> axes.Axes:
        # df = self.get_raw_data_expanded()
        df = get_data_expanded(self.evaluation_data[self.evaluation_data.episode==episode], {
            's_':['s0','s1','s2','s3'], 's__':['s_0','s_1','s_2','s_3'], 'p':['p_0','p_1'],
            'estimated_s':['estimated_s0','estimated_s1','estimated_s2','estimated_s3'], #'estimated_p':['estimated_p_0','estimated_p_1']
        })

        actions = ['<','>']
        actions_colors = ['white','black']

        axs.set_title(f'Episodes Progression P({df[df.episode == episode].p.values[0]})')
        axs.set_xlabel('pole_angle')
        axs.set_ylabel('angular_velocity')


        for epi in df.episode.unique():
            # axs.text(0, 0, f'p({df[df.episode == epi].p.values[0]})', color='r')
            for step in df[df.episode==epi].step.unique():
                x, y = (
                    pd.concat([df[(df.episode == epi) & (df.step == step)].s2, df[(df.episode == epi) & (df.step == step)].s_2]),
                    pd.concat([df[(df.episode == epi) & (df.step == step)].s3, df[(df.episode == epi) & (df.step == step)].s_3])
                )
                x_,y_ = (
                    pd.concat([df[(df.episode == epi) & (df.step == step)].s2, df[(df.episode == epi) & (df.step == step)].estimated_s2]),
                    pd.concat([df[(df.episode == epi) & (df.step == step)].s3, df[(df.episode == epi) & (df.step == step)].estimated_s3])
                )
                xErr,yErr = (
                    np.array([x.iloc[1], x_.iloc[1]]),
                    np.array([y.iloc[1], y_.iloc[1]])
                )
                axs.plot(x, y, color='b') # Real Step
                axs.plot(x_, y_, color='g', linestyle='dotted') # Estimated Step
                axs.plot(xErr, yErr, color='r', linestyle='dotted') # Estimated Step
            axs.plot(df[df.episode == epi].s2.values[:2], df[df.episode == epi].s3.values[:2], color='b', label=f'episode {epi}') # Initial Step END
            axs.plot(df[df.episode == epi].s2.values[:1], df[df.episode == epi].s3.values[:1], color='b',  # Initial Step BEGIN
                    marker=actions[df[df.episode == epi].a.values[0]],
                    markerfacecolor=actions_colors[df[df.episode == epi].a.values[0]]
            )
            for step in df[df.episode == epi].step.unique():
                axs.plot( # Actions
                    df[(df.episode == epi) & (df.step == step)].s_2.values[0], 
                    df[(df.episode == epi) & (df.step == step)].s_3.values[0], 
                    color='b', 
                    marker=actions[df[(df.episode == epi) & (df.step == step)].a.values[0]], 
                    markerfacecolor=actions_colors[df[(df.episode == epi) & (df.step == step)].a.values[0]])
        axs.legend()
        return axs

    def plot_training_loss(self, axs:axes.Axes, data:pd.Series=None, color:str='g') -> axes.Axes:
        if data is None:
            data = self.training_results.transition

        reff = np.zeros(data.shape[0])
        
        axs.set_title(f'Training Progression ({data.name})')
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Loss')
        axs.plot(data, color=color, label={data.name})
        axs.plot(reff, linestyle = 'dotted', color=color)
        return axs
        
    def plot_value_through_episodes(self, axs:axes.Axes, data:pd.DataFrame, col:str, 
                                    color:str='r', y_label:str='Rooted Squared Error'
        ) -> axes.Axes:
        results = data.copy()

        episodes = results.reset_index().groupby('episode').index.min().values
        # tickers = np.arange(4)
        values = results[col].values
        n = results.shape[0]

        axs.set_title(col)
        axs.set_ylabel(y_label)
        axs.set_xlabel('step')

        axs.plot(values, color = color)
        
        # Zero Line
        axs.plot(np.zeros(n), linestyle = 'dotted', color = 'g')
        for i, epi in enumerate(episodes):
            # Avg Reference
            epi_values = results[results.episode==i][col].values
            epi_n = len(epi_values)
            axs.text(epi_n//2+epi, np.mean(epi_values), f'{round(np.mean(epi_values),1)}', color='r', alpha=.4) # adjust y position as needed
            axs.plot(np.arange(epi_n)+epi, np.ones(epi_n)*np.mean(epi_values), linestyle = 'dotted', color = 'r', alpha=.2)
            
            # Episode References
            axs.axvline(x=epi, color='y', linestyle='--', linewidth=1, alpha=.4)
            axs.text(epi+.2, np.mean(values), f'{i}', color='y', alpha=.8) # adjust y position as needed
        axs.text(-3, np.mean(values), 'Episode', rotation=90, verticalalignment='center', color='y', alpha=.5) # adjust y position as needed
        # axs.set_yticks(tickers)

        # Avg Reference
        # axs.text(n-1, np.mean(values), f'{round(np.mean(values),1)}', color='r') # adjust y position as needed
        # axs.plot(np.ones(n)*np.mean(values), linestyle = 'dotted', color = 'r')

        return axs

    