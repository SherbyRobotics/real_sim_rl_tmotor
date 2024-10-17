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

    def step(self, u_s):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        th_s = th
        thdot_s = thdot #/ np.sqrt(1 / l)
        u = u_s * (m * l)
        q_s = self.q / (m * l)
        max_torque_s = self.max_torque * (m * l)
        # dt_s = dt / np.sqrt(m * l **2 / self.max_torque)
        dt_s = dt * np.sqrt(1 / l)
        
        u = np.clip(u, -max_torque_s, max_torque_s)[0]

        self.last_u = u  # for rendering
        costs = (angle_normalize(th_s) ** 2 + 0.1 * thdot_s**2 + 0.001 * (u_s**2) ) #* dt_s

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
        truncation = abs(th) > np.pi*10
        # truncation = False

        obs  = np.array([np.cos(newth_s), np.sin(newth_s), newthdot_s], dtype=np.float32)
    
        return obs, -costs, False, truncation, {}
    
    def dimensionless_cost(self, u):
        th, thdot = self.state

        g = self.g
        m = self.m
        l = self.l

        th_s = th
        thdot_s = thdot / np.sqrt(g / l)
        u_s = u / (m * g * l)
        q_s = self.q / (m * g * l)

        costs = 1 - (q_s **2 * angle_normalize(th_s) ** 2 + (u_s**2)) / (q_s **2 * np.pi ** 2)
        return costs

    

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.state = np.array([np.pi, 0])

        obs = self._get_obs()
        return obs, {}
    


# create the environment

env = modified_pendulum()
env.action_space = gym.spaces.Box(low=-2, high=2, shape=(1,))

env.max_speed = 10
env.max_torque = 2.0
env.l = 1.0
env.m = 2.0
env.dt = 0.05

# env = gym.wrappers.TimeLimit(env, max_episode_steps=300) #200

# model = SAC("MlpPolicy", env, verbose=1)
# # model = SAC.load("sac_pendulum_adim", env=env)
# model.learn(total_timesteps=30000)

# model.save("sac_pendulum_adim")

model = SAC.load("sac_pendulum_adim", env=env)

#load npy file
U = np.load("vi_policy.npy", allow_pickle=True)
theta = np.linspace(-np.pi, np.pi, 100)
theta_dot = np.linspace(-10, 10, 100)

vec_env = model.get_env()
obs = vec_env.reset()

obs = vec_env.reset()
obs_array = []
state_array = []
ep_reward = 0

n_steps = 300

theta_max = 10
theta_dot_max = 10
theta_min = -10
theta_dot_min = -10

while True:
    for _ in range(n_steps):
        # select action from the value iteration policy and interpolate on the action space
        th_normalized = angle_normalize(np.arctan2(obs[0][1], obs[0][0]) + np.pi) 
        th_idx = np.argmin(np.abs(theta - th_normalized))
        ths_idx = np.argmin(np.abs(theta_dot - obs[0][2]))

        action = np.array([[U[ths_idx, th_idx]]])
        # action, _state = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")

        reward = env.dimensionless_cost(action)
        # VecEnv resets automatically
        ep_reward += reward
        obs_array.append(obs)
        state_array.append([np.arctan2(obs[0][1], obs[0][0]), obs[0][2]])
        # state_array = np.append(state_array, [np.arctan2(obs[0][1], obs[0][0]), obs[0][2]].T)

    print("Episode reward:", ep_reward/n_steps * 100)

    obs_array = []
    state_array = []
    obs = vec_env.reset()
    ep_reward = 0