import gymnasium as gym
import rlenvs
import pandas as pd

n=500

# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("custom/CartPole-v1", render_mode="human")
env.action_space.seed(82)

observation, info = env.reset(seed=82, masspole=.45, length=1.0)
hist_s = [observation]
hist_a = []
for _ in range(n):
    # action = env.action_space.sample()
    # action = 1
    action = int(input())-1
    if not(action in [0,1]):
        break
    # print('action: ', action)
    observation,reward, terminated, truncated, info = env.step(action)
    # print("observation : ",observation);
    
    hist_s.append(observation)
    hist_a.append(action)

    if terminated or truncated:
        break
        # observation, info = env.reset()
        
env.close()

data = [{'cart_position': s1, 'cart_velocity': s2, 'pole_angle': s3, 'angular_velocity': s4, "a": a} for (s1,s2,s3,s4),a in zip(hist_s, hist_a)]
df = pd.DataFrame(data)
df.index.name = 't'
df.to_csv('notebooks/data.csv')