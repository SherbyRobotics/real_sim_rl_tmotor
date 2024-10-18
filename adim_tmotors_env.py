import time
from threading import Thread

import gymnasium as gym
import numpy as np
import rclpy

from tmotors_node import TMotorsNode, TorqueCmd, PositionCmd, EnableCmd


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class TMotorsEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None):
        self.observation_space = gym.spaces.Box(np.array([-1, -1, -8]), np.array([1, 1, 8]))
        self.action_space = gym.spaces.Box(-4, 4)
        self._target = np.array([1, 0])

        rclpy.init()
        self._is_spinning = True
        self._node = TMotorsNode()
        self._node_thread = Thread(target=self._node_spin, args=(self._node,))
        self._node_thread.start()
        self.step_count = 0

        self.render_mode = render_mode

        self.m = 1.0
        self.l = 1.0
        self.max_torque = 2.0

    def _get_obs(self):
        y = np.cos(self._node.joint_state[0])
        x = np.sin(self._node.joint_state[0])
        return np.array([y, x, self._node.joint_state[1]])

    def _get_info(self):
        state = self._node.joint_state.copy()
        return {"position": np.cos(state[0]), "velocity": state[1], "torque": state[2]}
        # return {"position": state[0], "velocity": state[1], "torque": state[2]}

    def _enable_motors(self):
        self._node.send_cmd(EnableCmd())

    def _set_joints(self, position):
        self._node.send_cmd(PositionCmd(position))

    def _apply_torque(self, tau):
        self._node.send_cmd(TorqueCmd(tau))

    def _node_spin(self, node):
        self._enable_motors()

        while self._is_spinning:
            rclpy.spin_once(node)

    def reset(self, seed=None):
        super().reset(seed=seed)

        if self.render_mode == "ansi":
            print("Reset")

        self._apply_torque(0.0)
        time.sleep(5)  # TODO: remove hardcode
        self._enable_motors()
        time.sleep(5) 

        # self._set_joints(np.random.uniform(0, 2*np.pi))
        # time.sleep(3) 

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):

        g = 10.0
        m = self.m
        l = self.l

        tau = action[0] * m * l                            # scale torque
        max_torque_s = self.max_torque * m * l             # scale max torque
        tau = np.clip(tau, -max_torque_s, max_torque_s)

        self._apply_torque(float(tau))
        time.sleep(0.05)  # TODO: remove hardcode
        obs = self._get_obs()
        info = self._get_info()

        obs[2] = obs[2] / np.sqrt(1 / l)                   # normalize theta_dot


        th = self._node.joint_state[0]
        thdot = self._node.joint_state[1]

        # reward = -( (self._target[0]-obs[0])**2 + 0.1*(self._target[1]-obs[2])**2 + 0.001*tau**2)  # TODO: better cost

        reward = -(  angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (tau**2)  )/100

        terminated = truncated = False

        if self.render_mode == "ansi":
            print("Step:", self.step_count)
            for key, value in info.items():
                print(f"{key}\t: {value:.2f}")
            print(f"Reward\t: {reward:.2f}")
            print()
            self.step_count += 1

        return obs, reward, terminated, truncated, info
    
    def dimensionless_cost(self, u):
        th = self._node.joint_state[0]
        thdot = self._node.joint_state[1]

        g = 10.0
        m = self.m
        l = self.l
        max_torque = self.max_torque
        q = 30.0

        u = np.clip(u, -max_torque, max_torque)

        th_s = th
        thdot_s = thdot / np.sqrt(g / l)
        u_s = u / (m * g * l)
        q_s = q / (m * g * l)

        costs = 1 - (q_s **2 * angle_normalize(th_s) ** 2 + (u_s**2)) / (q_s **2 * np.pi ** 2)
        return costs

    def render(self):
        pass


    def close(self):
        self._is_spinning = False
        self._node_thread.join()
        rclpy.shutdown()


if __name__ == "__main__":
    #from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3 import PPO, SAC, TD3
    import matplotlib.pyplot as plt
    from stable_baselines3.common.logger import configure

    env = TMotorsEnv(render_mode="ansi")
    env.max_torque = 4.0
    env.l = 0.45
    env.m = 2*0.05 + 0.368 + 0.07 + 0.380

    env = gym.wrappers.TimeLimit(env, max_episode_steps=200) #200

    model = SAC.load("sac_pendulum_adim", env=env)

    # print("Training...")
    # model = SAC("MlpPolicy", env, verbose=1, use_sde=True)
    # log training data to csv

    # # current working directory is used for logging
    # path = "/home/ian/tmotor_ws/src/real_sim_rl_tmotor"    

    # # set up logger
    # new_logger = configure(path, ["stdout", "csv", "tensorboard"])
    # # Set new logger
    # model.set_logger(new_logger)

    # model.learn(total_timesteps=2000) #2500   

    # model.save("sac_pendulum_v41")

    theta = np.linspace(-np.pi, np.pi, 100)
    theta_dot = np.linspace(-10, 10, 100)

    X, Y = np.meshgrid(theta, theta_dot)

    Z = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            obs = np.array([np.cos(X[i, j]), np.sin(X[i,j]) , Y[i, j]])
            Z[i, j] = model.predict(obs, deterministic=True)[0]

    
    plt.figure()
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.xlabel("theta")
    plt.ylabel("theta_dot")
    plt.colorbar()
    plt.savefig("sac_pendulum_v21.png")
    plt.show()

    ep_reward = 0
    n_steps = 200

    while True:
        obs , info = env.reset()
        for _ in range(n_steps):
            action, _states = model.predict(obs, deterministic=True)
            print("action", action * env.m * env.l )
            obs, rewards, terminated, truncated, info = env.step(action)

            reward = env.dimensionless_cost(action)
            # VecEnv resets automatically
            ep_reward += reward
            #vec_env.render("human")

        print("Episode reward:", ep_reward/n_steps * 100)
        ep_reward = 0
