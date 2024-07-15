import time
from threading import Thread

import gymnasium as gym
import numpy as np
import rclpy

from tmotors_node import TMotorsNode, TorqueCmd, PositionCmd, EnableCmd


class TMotorsEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None):
        self.observation_space = gym.spaces.Box(np.array([-2*np.pi, -10]), np.array([2*np.pi, 10]))
        self.action_space = gym.spaces.Box(-1, 1)
        self._target = np.array([0, 0])

        rclpy.init()
        self._is_spinning = True
        self._node = TMotorsNode()
        self._node_thread = Thread(target=self._node_spin, args=(self._node,))
        self._node_thread.start()

        self.render_mode = render_mode

    def _get_obs(self):
        return self._node.joint_state[:2]

    def _get_info(self):
        state = self._node.joint_state.copy()
        return {"position": state[0], "velocity": state[1], "torque": state[2]}

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

        self._set_joints(0.)
        time.sleep(1)  # TODO: remove hardcode

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "ansi":
            print("Reset")

        return obs, info

    def step(self, action):
        tau = action[0]
        self._apply_torque(float(tau))
        time.sleep(0.1)  # TODO: remove hardcode
        obs = self._get_obs()
        info = self._get_info()

        reward = -np.sum( (self._target-obs)**2 )  # TODO: better cost

        terminated = truncated = False

        if self.render_mode == "ansi":
            print("Step:")
            for key, value in info.items():
                print(f"{key}\t: {value:.2f}")
            print()

        return obs, reward, terminated, truncated, info

    def render(self):
        pass


    def close(self):
        self._is_spinning = False
        self._node_thread.join()
        rclpy.shutdown()


if __name__ == "__main__":
    #from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3 import PPO

    env = TMotorsEnv(render_mode="ansi")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)

    print("Training...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2500)

    print("Test")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #vec_env.render("human")
