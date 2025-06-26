import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO, TD3
import matplotlib.pyplot as plt
from gymnasium.envs.classic_control.pendulum import PendulumEnv


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

class modified_pendulum(PendulumEnv):

    def __init__(self):
        super().__init__()
        self.render_mode = "rgb_array"
        self.max_speed = 100
        self.max_torque = 3
        self.l = 0.35
        self.m = 0.5
        self.dt = 0.1
        self.q = 30

        self.m1 = 0.19
        self.m2 = 0.21
        self.m3 = 1.0
        self.r1 = 0.06
        self.l1 = 0.39
        self.l2 = 0.26

        self.prev_u = 0.0

    
    def step(self, u_s):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        # fac = self.l / 1.0
        fac = 1.0

        th_s = th
        thdot_s = thdot 

        u = u_s * self.max_torque 
        max_torque_s = self.max_torque # * (m * l)            # scale max torque
        dt_s = dt * np.sqrt(fac)
        
        u = np.clip(u, -max_torque_s, max_torque_s)[0]
        # add noise to the action
        u += np.random.normal(0, 0.01)
        # add noise to the state
        th += np.random.normal(0, 0.0001)
        self.last_u = u  # for rendering
        
        costs = (angle_normalize(th_s) ** 2 + 0.1 * thdot_s**2 + 0.001 * (u**2) ) #* dt_s

        I = m * l ** 2  # inertia

        # self.prev_u = u

        thdot = thdot / np.sqrt(fac)

        newthdot = thdot + ( g / l * np.sin(th) + self.prev_u / I ) * dt_s
        # newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) / np.sqrt(fac)
        newth = th + newthdot * dt_s

        self.prev_u = u

        # dimensionless states
        newth_s = newth
        newthdot_s = newthdot * np.sqrt(fac)

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        truncation = abs(th) > np.pi*6
        # truncation = False

        self.state = np.array([newth_s, newthdot_s])
    
        return self._get_obs(), -costs, False, truncation, {}
    
    def dimensionless_cost(self, u):
        th, thdot = self.state

        g = 10.0
        m = self.m
        l = self.l
        max_torque = self.max_torque

        u = np.clip(u, -max_torque, max_torque)

        th_s = th
        thdot_s = thdot
        u_s = u * self.max_torque

        costs = 1 - (angle_normalize(th_s) ** 2 + 0.0 * thdot_s**2 + 0.001 * (u_s**2)) / (np.pi ** 2)
        return costs

    

    # def reset(self, seed=None):
    #     super().reset(seed=seed)
        
    #     self.state = np.array([np.pi, 0])

    #     obs = self._get_obs()
    #     return obs, {}
    


# create the environment

env = modified_pendulum()

env.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

max_torque = 4.0

env.max_speed = 100
env.l = 0.37
tube_m = 0.115 * 0.58**2 /3 / env.l**2
env.m = 2*0.06 + 0.196 + tube_m + 0.330*2
# env.l = 1.0
# env.m = 1.0
env.max_torque = max_torque * env.l * env.m
print("max_torque", env.max_torque)
env.dt = 0.015

env = gym.wrappers.TimeLimit(env, max_episode_steps=400) #200

model = SAC("MlpPolicy", env, verbose=1)
# model = SAC.load("sac_pendulum_adim", env=env)
model.learn(total_timesteps=30000)

model.save("sac_pendulum_adim_final")

model = SAC.load("sac_pendulum_adim_final", env=env)

# ################################################
# # testing on environment with different parameters

# # exponential distribution of m
# m_array = np.linspace(1.0, 5.0, 8)
# max_torque_array = m_array * 4.0
# # m_array = np.array([0.538, 0.918, 1.298])
# # max_torque_array = np.array([0.5, 0.75, 1.0, 1.5, 2.0])

# # 2d array to store the score
# score_array = []
# # 2d array to store the ratio of the score
# ratio_array = np.zeros((len(m_array), len(max_torque_array)))


# for k, max_torque in enumerate(max_torque_array):
#     score = []
#     for j, m in enumerate(m_array):
#         env.unwrapped.m = 1.0 
#         env.unwrapped.l = 1.0 * m 
#         # max_torque = 4.0 * m
#         env.unwrapped.max_torque = max_torque


#         model = SAC.load("sac_pendulum_adim_final", env=env)

#         vec_env = model.get_env()


#         ratio_array[j, k] = env.unwrapped.l / env.unwrapped.max_torque

#         obs = vec_env.reset()
#         obs_array = []
#         state_array = []
#         ep_reward = 0

#         # n_steps = int(400 * np.sqrt(m))
#         n_steps = 400

#         for i in range(n_steps):
#             action, _state = model.predict(obs, deterministic=True)
#             obs, reward, done, info = vec_env.step(action)
#             # vec_env.render("human")

#             reward = env.unwrapped.dimensionless_cost(action) 
#             # VecEnv resets automatically
#             ep_reward += reward
#             obs_array.append(obs)
#             state_array.append([np.arctan2(obs[0][1], obs[0][0]), obs[0][2]])

#             if done:
#                 break


#         # print reward of the episode in percentage
#         print("Episode reward:", ep_reward/n_steps * 100)
#         score.append(ep_reward/n_steps * 100)

#         obs_array = []
#         state_array = []
#         obs = vec_env.reset()
#         ep_reward = 0

#     score_array.append(score)


# # plot the reward in 3d

# score_array = np.array(score_array)
# print(score_array.shape)
# print(score_array)


# plt.figure()
# X, Y = np.meshgrid(m_array, max_torque_array)
# print(X, Y)
# Z = np.squeeze(np.array(score_array))
# plt.pcolormesh(X, Y, Z, shading='auto')
# plt.xlabel("m")
# plt.ylabel("max_torque")
# plt.colorbar()


# plt.figure()
# X, Y = np.meshgrid(m_array, max_torque_array)
# Z = np.log(np.squeeze(np.array(ratio_array)))
# plt.pcolormesh(X, Y, Z, shading='auto')
# plt.xlabel("m")
# plt.ylabel("max_torque")
# plt.colorbar()
# plt.show()


# env.unwrapped.max_torque = 8.0
# env.unwrapped.l = 2.0
# env.unwrapped.m = 1.0

model = SAC.load("sac_pendulum_adim_final", env=env)

obs = env.unwrapped.reset()

theta = np.linspace(-2*np.pi, 2*np.pi, 100)
theta_dot = np.linspace(-4, 4, 100)

X, Y = np.meshgrid(theta, theta_dot)

Z = np.zeros_like(X)
for i in range(len(X)):
    for j in range(len(Y)):
        obs = np.array([np.cos(X[i, j]), np.sin(X[i,j]) , Y[i, j]])
        Z[i, j] = model.predict(obs, deterministic=True)[0]

# plot heatmap of the value function
plt.figure()
plt.pcolormesh(X, Y, Z, shading='auto')
plt.xlabel("theta")
plt.ylabel("theta_dot")
plt.colorbar()
plt.show()

#####################################

vec_env = model.get_env()
obs, _ = env.unwrapped.reset()
obs_array = []
state_array = []
ep_reward = 0

# n_steps = int(200 * env.l * 2)
n_steps = 400

while True:
    for _ in range(n_steps):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info, _ = env.unwrapped.step(action)
        vec_env.render("human")

        reward = env.unwrapped.dimensionless_cost(action)
        # VecEnv resets automatically
        ep_reward += reward
        obs_array.append(obs)
        state_array.append([np.arctan2(obs[1], obs[0]), obs[2]])

    # print reward of the episode in percentage
    print("Episode reward:", ep_reward/n_steps * 100)

    obs_array = []
    state_array = []
    obs, _ = env.unwrapped.reset()
    ep_reward = 0