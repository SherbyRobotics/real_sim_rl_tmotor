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

    # def step(self, u_s):
    #     th, thdot = self.state  # th := theta

    #     g = self.g
    #     m = self.m
    #     l = self.l
    #     dt = self.dt

    #     th_s = th
    #     thdot_s = thdot * np.sqrt(m * l **2 / self.max_torque)
    #     u = u_s / self.max_torque
    #     dt_s = dt / np.sqrt(m * l **2 / self.max_torque)
        
    #     u = np.clip(u, -self.max_torque, self.max_torque)[0]

    #     self.last_u = u  # for rendering
    #     costs = ( angle_normalize(th_s) ** 2 + 0.1 * thdot_s**2 + 0.001 * (u_s**2) ) * dt_s

    #     I = m * l ** 2  # inertia

    #     newthdot = thdot + ( g / l * np.sin(th) + u / I ) * dt
    #     newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
    #     newth = th + newthdot * dt

    #     # dimensionless states
    #     newth_s = newth
    #     newthdot_s = newthdot / np.sqrt(g / l)

    #     self.state = np.array([newth, newthdot])

    #     if self.render_mode == "human":
    #         self.render()
    #     # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`

    #     obs  = np.array([np.cos(newth_s), np.sin(newth_s), newthdot_s], dtype=np.float32)
    
    #     return obs, -costs, False, False, {}
    
    def step(self, u_s):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        th_s = th
        thdot_s = thdot / np.sqrt(1 / l)
        u = u_s * (m * l)
        max_torque_s = self.max_torque * (m * l)            # scale max torque
        # dt_s = dt / np.sqrt(m * l **2 / self.max_torque)
        dt_s = dt * np.sqrt(1 / l)
        
        u = np.clip(u, -max_torque_s, max_torque_s)[0]
        # add noise to the action
        u += np.random.normal(0, 0.03)
        # add noise to the state
        th += np.random.normal(0, 0.001)

        self.last_u = u  # for rendering
        costs = (angle_normalize(th_s) ** 2 + 0.3 * thdot_s**2 + 0.001 * (u_s**2) ) #* dt_s

        I = m * l ** 2  # inertia

        newthdot = thdot + ( g / l * np.sin(th) + u / I ) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        # dimensionless states
        newth_s = newth
        newthdot_s = newthdot / np.sqrt(1 / l)

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        truncation = abs(th) > np.pi*6
        # truncation = False

        obs  = np.array([np.cos(newth_s), np.sin(newth_s), newthdot_s], dtype=np.float32)
    
        return obs, -costs, False, truncation, {}
    
    def dimensionless_cost(self, u):
        th, thdot = self.state

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        th_s = th
        thdot_s = thdot / np.sqrt(g / l)
        u_s = u / (m * g * l)
        # u = np.clip(u, -self.max_torque, self.max_torque)
        q_s = self.q / (m * g * l)

        costs = (1 - (q_s **2 * angle_normalize(th_s) ** 2 + (u_s**2)) / (q_s **2 * np.pi ** 2) )
        return costs

    

    # def reset(self, seed=None):
    #     super().reset(seed=seed)
        
    #     self.state = np.array([np.pi, 0])

    #     obs = self._get_obs()
    #     return obs, {}
    


# create the environment

env = modified_pendulum()
env.action_space = gym.spaces.Box(low=-2, high=2, shape=(1,))

env.max_speed = 10
env.max_torque = 2.0
env.l = 0.5
env.m = 1.0
env.dt = 0.05

env = gym.wrappers.TimeLimit(env, max_episode_steps=200) #200

model = SAC("MlpPolicy", env, verbose=1)
# model = SAC.load("sac_pendulum_adim", env=env)
model.learn(total_timesteps=10000)

model.save("sac_pendulum_adim")

model = SAC.load("sac_pendulum_adim", env=env)

# ################################################
# # testing on environment with different parameters

# # exponential distribution of m
# m_array = np.linspace(0.25, 10.0, 10)
# max_torque_array = m_array * 2.0
# # 2d array to store the score
# score_array = []
# # 2d array to store the ratio of the score
# ratio_array = np.zeros((len(m_array), len(max_torque_array)))


# for k, max_torque in enumerate(max_torque_array):
#     score = []
#     for j, m in enumerate(m_array):
#         env.m = 1.0
#         env.l = 1.0 * m  
#         env.max_torque = max_torque

#         ratio_array[j, k] = m / max_torque

#         model = SAC.load("sac_pendulum_adim", env=env)

#         vec_env = model.get_env()
#         obs = vec_env.reset()


#         obs = vec_env.reset()
#         obs_array = []
#         state_array = []
#         ep_reward = 0

#         # n_steps = int(300 * np.sqrt(m))
#         n_steps = 300

#         for i in range(n_steps):
#             action, _state = model.predict(obs, deterministic=True)
#             obs, reward, done, info = vec_env.step(action)
#             # vec_env.render("human")

#             reward = env.dimensionless_cost(action)
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
# print(ratio_array)


# plt.figure()
# X, Y = np.meshgrid(m_array, max_torque_array)
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



# env.max_torque = 4.0
env.l = 0.5
env.m = 0.5   

model = SAC.load("sac_pendulum_adim", env=env)

vec_env = model.get_env()
obs = vec_env.reset()

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
obs = vec_env.reset()
obs_array = []
state_array = []
ep_reward = 0

# n_steps = int(200 * env.l * 2)
n_steps = 200

while True:
    for _ in range(n_steps):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")

        reward = env.dimensionless_cost(action)
        # VecEnv resets automatically
        ep_reward += reward
        obs_array.append(obs)
        state_array.append([np.arctan2(obs[0][1], obs[0][0]), obs[0][2]])

    # print reward of the episode in percentage
    print("Episode reward:", ep_reward/n_steps * 100)

    obs_array = []
    state_array = []
    obs = vec_env.reset()
    ep_reward = 0