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
    hist_p = np.array([list(options.values())])
    for i in range(n):
        action = int(np.random.choice([0,1], size=1)[0])
        observation,reward, terminated, truncated, info = env.step(action)
        
        hist_s = np.concat([hist_s, [observation]])
        hist_a = np.concat([hist_a, [action]])
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
    return hist_s, hist_a, hist_p

def train_test_split(hist_s, hist_a, hist_p, mode='all', p=.3):
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
    input_p = hist_p[:-2]
    input_all = np.concat([input_s, input_a, input_s_, input_a_, input_p], axis=1)
    output_all = np.concat([input_s_2, input_p], axis=1)

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
