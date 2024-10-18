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

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        # add noise to the action
        u += np.random.normal(0, 0.03)
        # add noise to the state
        # thdot += np.random.normal(0, 0.01)
        th += np.random.normal(0, 0.0001)

        self.last_u = u  # for rendering

        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        I = self.m * self.l ** 2  # inertia
        # I = 0.5 * self.m1 * self.r1**2 + self.m2  * (self.l1**3 + self.l2**3) / (3 * (self.l1 + self.l2)) + self.m3 * self.l1**2 # inertia 

        newthdot = thdot + ( g / l * np.sin(th) + self.prev_u / I ) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.prev_u = u

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()

        truncation = abs(th) > np.pi*6
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, False, truncation, {}
    
    def dimensionless_cost(self, u):
        th, thdot = self.state

        g = self.g
        m = self.m
        l = self.l

        u = np.clip(u, -self.max_torque, self.max_torque)[0]

        th_s = th
        thdot_s = thdot / np.sqrt(g / l)
        u_s = u / (m * g * l)
        q_s = self.q / (m * g * l)

        costs = 1 - (q_s **2 * angle_normalize(th_s) ** 2 + (u_s**2)) / (q_s **2 * np.pi ** 2)
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

env.max_speed = 10
env.max_torque = 1.0
# env.l = 0.42
# env.m = 0.8
env.l = 0.45
env.m = 2*0.05 + 0.368 + 0.07
env.dt = 0.05
print(env.m, env.l)

model = SAC("MlpPolicy", env, verbose=1)
# model = SAC.load("sac_pendulum", env=env)
model.learn(total_timesteps=10000)

model.save("sac_pendulum")

model = SAC.load("sac_pendulum", env=env)

#load npy file
U = np.load("vi_policy_real.npy", allow_pickle=True)

vec_env = model.get_env()
obs = vec_env.reset()

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


obs = vec_env.reset()
obs_array = []
state_array = []
ep_reward = 0

n_steps = 250

theta_max = 10
theta_dot_max = 10
theta_min = -10
theta_dot_min = -10

while True:
    for _ in range(n_steps):
        action, _state = model.predict(obs, deterministic=True)
        # select action from the value iteration policy and interpolate on the action space
        th_normalized = angle_normalize(np.arctan2(obs[0][1], obs[0][0]) + np.pi) 
        th_idx = np.argmin(np.abs(theta - th_normalized))
        ths_idx = np.argmin(np.abs(theta_dot - obs[0][2]))
        # print(th_normalized, obs[0][2])
        # print(th_idx, ths_idx)

        # action = np.array([[U[ths_idx, th_idx]]])
        # print(action)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")

        reward = env.dimensionless_cost(action)
        # VecEnv resets automatically
        ep_reward += reward
        obs_array.append(obs)
        state_array.append([np.arctan2(obs[0][1], obs[0][0]), obs[0][2]])
        # state_array = np.append(state_array, [np.arctan2(obs[0][1], obs[0][0]), obs[0][2]].T)


    # plot coordinate trajectory
    # obs_array = np.array(obs_array).T
    # plt.figure()
    # plt.plot(obs_array[0], obs_array[1], "r.")
    # plt.pause(0.01)
    # plt.show()

    # # plot state space trajectory
    # state_array = np.array(state_array).T
    # plt.figure()
    # plt.plot(state_array[0], state_array[1], "r.")
    # plt.pause(0.01)
    # plt.show()

    # print reward of the episode in percentage
    print("Episode reward:", ep_reward/n_steps * 100)

    obs_array = []
    state_array = []
    obs = vec_env.reset()
    ep_reward = 0