import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import rlenvs


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
