from rlenvs import Hunting 

n=500

env = Hunting(3,3)
env.plot()

observation, _, _, _, _ = env.last()
hist_s = [observation]
hist_a = []
for _ in range(n):
    print("observation : ",observation);
    action = int(input())-1
    if not(action in [0,1,2,3]):
        break
    # print('action: ', action)
    observation,reward, terminated, truncated, info = env.step(observation, action)
    # print("observation : ",observation);
    # print(reward, terminated, truncated);
    env.plot()
    
    hist_s.append(observation)
    hist_a.append(action)


    if terminated or truncated:
        break
        # observation, info = env.reset()

print([{'s': s, "a": a} for s,a in zip(hist_s, hist_a)])