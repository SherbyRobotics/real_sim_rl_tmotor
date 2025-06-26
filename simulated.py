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

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        thdot = thdot 
        
        max_torque = self.max_torque
        u = u * max_torque
        u = np.clip(u, -max_torque, max_torque)[0]
        # # add noise to the action
        # u += np.random.normal(0, 0.03)
        # # add noise to the state
        # # thdot += np.random.normal(0, 0.01)
        # th += np.random.normal(0, 0.0001)

        self.last_u = u  # for rendering

        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        I = self.m * self.l ** 2  # inertia
        # I = 0.5 * self.m1 * self.r1**2 + self.m2  * (self.l1**3 + self.l2**3) / (3 * (self.l1 + self.l2)) + self.m3 * self.l1**2 # inertia 

        # self.prev_u = u

        newthdot = thdot + ( g / l * np.sin(th) + self.prev_u / I ) * dt 
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) 
        newth = th + newthdot * dt

        self.prev_u = u

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()

        truncation = abs(th) > np.pi*6
        # truncation=False #as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, False, truncation, {}
    
    def dimensionless_cost(self, u):
        th, thdot = self.state

        g = self.g
        m = self.m
        l = self.l

        u = np.clip(u, -self.max_torque, self.max_torque)[0]

        th_s = th
        thdot_s = thdot
        u_s = u 

        # costs = 1 - (q_s **2 * angle_normalize(th_s) ** 2 + (u_s**2)) / (q_s **2 * np.pi ** 2)

        costs = 1 - (angle_normalize(th_s) ** 2 + 0.1 * thdot_s**2 + 0.001 * (u_s**2)) / (np.pi ** 2)# * np.sqrt(1/l)
        return costs

    

    # def reset(self, seed=None):
    #     super().reset(seed=seed)
        
    #     self.state = np.array([np.pi, 0])

    #     obs = self._get_obs()
    #     return obs, {}
    


# Create the vectorized environment
# env = gym.make("Pendulum-v1", render_mode="rgb_array")

env = modified_pendulum()
env.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

env.max_speed = 1000
max_torque = 4.0
# env.l = 0.42
# env.m = 0.8
env.l = 0.37
env.m = 1.0
env.dt = 0.015
env.max_torque = max_torque * env.l * env.m

# env.unwrapped.fac = 20.
# env.unwrapped.max_torque = 3.11111
print(env.m, env.l)

model = SAC("MlpPolicy", env, verbose=1)
# model = SAC.load("sac_pendulum", env=env)
model.learn(total_timesteps=10000)

model.save("sac_pendulum")

model = SAC.load("sac_pendulum", env=env)

# #load npy file
# U = np.load("vi_policy_real.npy", allow_pickle=True)

# vec_env = model.get_env()
# obs = vec_env.reset()

theta = np.linspace(-np.pi, np.pi, 100)
# print(theta)
theta_dot = np.linspace(-10, 10, 100)

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
# save pgn picture
plt.savefig("sac_pendulum.png")
plt.show()

vec_env = model.get_env()

while True:
    obs = vec_env.reset()
    for i in range(200):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")




# exponential distribution of m
m_array = np.linspace(0.1, 5.0, 10)
print(m_array)
max_torque_array = m_array * 4

# 2d array to store the score
score_array = []
ratio_array = np.zeros((len(m_array), len(max_torque_array)))


for k, max_torque in enumerate(max_torque_array):
    score = []
    for j, m in enumerate(m_array):
        env.unwrapped.m = m
        env.unwrapped.max_torque = max_torque


        model = SAC.load("sac_pendulum", env=env)

        vec_env = model.get_env()

        ratio_array[j, k] = env.unwrapped.m / env.unwrapped.max_torque

        obs = vec_env.reset()
        obs_array = []
        state_array = []
        ep_reward = 0

        n_steps = 200

        for i in range(n_steps):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            # vec_env.render("human")

            reward = env.dimensionless_cost(action) 
            # VecEnv resets automatically
            ep_reward += reward
            obs_array.append(obs)
            state_array.append([np.arctan2(obs[0][1], obs[0][0]), obs[0][2]])

            if done:
                break


        # print reward of the episode in percentage
        print("Episode reward:", ep_reward/n_steps * 100)
        score.append(ep_reward/n_steps * 100)

        obs_array = []
        state_array = []
        obs = vec_env.reset()
        ep_reward = 0

    score_array.append(score)


# plot the reward in 3d

score_array = np.array(score_array)
print(score_array.shape)
print(score_array)


plt.figure()
X, Y = np.meshgrid(m_array, max_torque_array)
Z = np.log(np.squeeze(np.array(ratio_array)))
plt.pcolormesh(X, Y, Z, shading='auto')
plt.xlabel("mgl")
plt.ylabel("max_torque")
plt.colorbar()

plt.figure()
X, Y = np.meshgrid(m_array, max_torque_array)
print(X, Y)
Z = np.squeeze(np.array(score_array))
plt.pcolormesh(X, Y, Z, shading='auto')
plt.xlabel("mgl")
plt.ylabel("max_torque")
plt.colorbar()
plt.show()



# env.max_torque = 8.0
# # env.l = 0.42
# # env.m = 0.8
# env.l = 2.0
# env.m = 2.0


model = SAC.load("sac_pendulum", env=env)

vec_env = model.get_env()
obs_array = []
state_array = []
ep_reward = 0

n_steps = 200

while True:
    obs = vec_env.reset()
    for i in range(n_steps):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")