# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:31:21 2024

@author: i_lal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import pendulum
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer
import matplotlib.pyplot as plt

plt.close('all')


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

# Cost Function
class CustomCostFunction( costfunction.CostFunction ):
    """ 
    Quadratic cost functions of continuous dynamical systems
    ----------------------------------------------
    n : number of states
    m : number of control inputs
    ---------------------------------------
    J = int( g(x,u,t) * dt ) + h( x(T) , T )
    
    g = xQx + uRu 
    h = xSx
    
    """
    
    ############################
    def __init__(self, n, m):
        
        costfunction.CostFunction.__init__(self)
        
        # dimensions
        self.n = n
        self.m = m
        
        # nominal values
        self.xbar = np.zeros(self.n)
        self.ubar = np.zeros(self.m)

        # Quadratic cost weights
        self.Q = np.diag( np.ones(n)  )
        self.R = np.diag( np.ones(m)  )
        self.S = np.diag( np.zeros(n) )
        
        # Optionnal zone of zero cost if ||x - xbar || < EPS 
        self.ontarget_check = True
        
    
    ############################
    @classmethod
    def from_sys(cls, sys):
        """ From ContinuousDynamicSystem instance """
        
        instance = cls( sys.n , sys.m )
        
        instance.xbar = sys.xbar
        instance.ubar = sys.ubar
        
        return instance
    

    #############################
    def h(self, x , t = 0):
        """ Final cost function with zero value """
        
        # Delta values with respect to nominal values
        dx = x - self.xbar

        dx[0] = angle_normalize(dx[0]) #+ np.pi
        
        # Quadratic terminal cost
        J_f = np.dot( dx.T , np.dot(  self.S , dx ) )
                     
        # Set cost to zero if on target
        if self.ontarget_check:
            if ( np.linalg.norm( dx ) < self.EPS ):
                J_f = 0
        
        return 0
    
    
    #############################
    def g(self, x, u, t):
        """ Quadratic additive cost """
        
        """
        TODO: Add check in init
        # Check dimensions
        if not x.shape[0] == self.Q.shape[0]:
            raise ValueError(
            "Array x of shape %s does not match weights Q with %d components" \
            % (x.shape, self.Q.shape[0])
            )
        if not u.shape[0] == self.R.shape[0]:
            raise ValueError(
            "Array u of shape %s does not match weights R with %d components" \
            % (u.shape, self.R.shape[0])
            )
        if not y.shape[0] == self.V.shape[0]:
            raise ValueError(
            "Array y of shape %s does not match weights V with %d components" \
            % (y.shape, self.V.shape[0])
            )
        """
            
        # Delta values with respect to nominal values
        dx = x - self.xbar
        du = u - self.ubar
        
        dx[0] = angle_normalize(dx[0])
        
        dJ = ( np.dot( dx.T , np.dot(  self.Q , dx ) ) +
               np.dot( du.T , np.dot(  self.R , du ) ) )
        
        # Set cost to zero if on target
        if self.ontarget_check:
            if ( np.linalg.norm( dx ) < self.EPS ):
                dJ = 0
        
        return dJ




import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.envs.classic_control.pendulum import PendulumEnv

class modifiedPendulum(PendulumEnv):

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = u*self.max_torque

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (g / l * np.sin(th) + u / (m * l**2)) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, False, False, self.state
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.state = np.array([np.pi, 0])

        obs = self._get_obs(), self.state
        return obs, {}



def step_sys(self, x, u, t=0):
        th = x[0]
        thdot = x[1]
        
        u = u*self.max_torque

        g = self.g
        m = self.m1
        l = self.lc1
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]

        dx = np.zeros(len(x))

        dx[1] = (g / l * np.sin(th) + u / (m * l**2))
        dx[0] = thdot + dx[1] * dt

        return dx


def trig(self, q ):
    """ Compute cos and sin """
    q = q + np.pi
    
    c1  = np.cos( q )
    s1  = np.sin( q )

    return [c1,s1]

pendulum.SinglePendulum.trig = trig
pendulum.SinglePendulum.f = step_sys

sys = pendulum.SinglePendulum()

m = 1.0
l = 1.0
max_torque = 4.0
dt = 0.05

# m_array = np.linspace(0.5, 4.0, 3)
# max_torque_array = m_array * 4.0

# for max_torque in max_torque_array:
#     for m in m_array:
        
#         print('m = ', m, ' max_torque = ', max_torque)

sys.x_ub = np.array([+4*np.pi, +10])
sys.x_lb = np.array([-4*np.pi,  -10])
sys.u_ub = np.array([1.0])
sys.u_lb = np.array([-1.0])
sys.m1 = m
sys.lc1 = l
sys.l1 = 1.0
sys.I1 = 0.0

sys.g = 9.81
sys.dt = dt
sys.max_torque = max_torque


# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [201,201] , [41] )
grid_sys.dt = dt


# Cost Function 
qcf = CustomCostFunction(2, 1)

qcf.xbar = np.array([ 0., 0. ]) # target

qcf.Q[0,0] = 1.0
qcf.Q[1,1] = 0.1
qcf.R[0,0] = 0.001 

qcf.S[0,0] = 10.0
qcf.S[1,1] = 1.0



# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

dp.solve_bellman_equation( animate_cost2go = False, tol = 0.4)


dp.clean_infeasible_set()
# dp.plot_cost2go_3D()
# dp.plot_policy()

ctl = dp.get_lookup_table_controller()
cl_sys = ctl + sys


env = modifiedPendulum(render_mode="human")
env.unwrapped.m = m
env.unwrapped.l = l
# env.unwrapped.max_speed = 1000
env.unwrapped.g = 9.81
env.unwrapped.max_torque = max_torque*l
env.unwrapped.dt = dt #* np.sqrt(l)
env.unwrapped.observation_space = Box(np.array([-1., -1., -10.]), np.array([1., 1., 10.]), (3,), dtype=np.float64)
env.unwrapped.action_space = Box(low=-max_torque, high=max_torque, shape=(1,), dtype=np.float64)

states_array = []
actions_array = []

sys_states_array = np.array([np.pi, 0.])
sys_actions_array = []


# while True:
obs, states = env.reset()[0]
x = np.array([np.pi, 0.0])

J = 0

for i in range(200):

    # obs = np.array(obs).squeeze()
    # states = np.array([np.arctan2(obs[1],obs[0]), obs[2]])
    # action = np.array([[2.0]])
    action = ctl.lookup_table_selection(states)
    # print(action, ' -- ', states)
    # states[1] = states[1] / np.sqrt(l)
    obs, _, _, _, states = env.step(action)
    # states[1] = states[1] * np.sqrt(l)  
    states_array.append(states)
    actions_array.append(action)

    J += 1.0 * angle_normalize(states[0])**2 + 0.1 * states[1]**2 + 0.001*(action*max_torque)**2

    # dx = cl_sys.plant.f(x, action, t=0)
    # x[0] = x[0] + dx[0]*dt
    # x[1] = dx[0]
    # # print(x)
    # sys_states_array = np.vstack((sys_states_array, x))

print(J/200)


# theta = np.linspace(-2*np.pi, 2*np.pi, 100)
# theta_dot = np.linspace(-4, 4, 100)

# X, Y = np.meshgrid(theta, theta_dot)

# Z = np.zeros_like(X)
# for i in range(len(X)):
#     for j in range(len(Y)):
#         states = np.array([X[i, j], Y[i, j]])
#         Z[i, j] = ctl.lookup_table_selection(states)

# # plot heatmap of the value function
# plt.figure()
# plt.pcolormesh(X, Y, Z, shading='auto')
# plt.xlabel("theta")
# plt.ylabel("theta_dot")
# plt.colorbar()
# plt.show()


plt.figure()
states_array = np.array(states_array)
ax = plt.subplot(311)
ax.plot(states_array[:,0])
ax.set_title('theta')
ax = plt.subplot(312)
ax.plot(states_array[:,1])
ax.set_title('theta_dot')
ax = plt.subplot(313)
ax.plot(actions_array)
ax.set_title('actions')
plt.suptitle('m = ' + str(m) + ' max_torque = ' + str(max_torque))
# plt.show()

# plt.figure()
# sys_states_array = np.array(sys_states_array).T
# print(sys_states_array.shape)
# ax = plt.subplot(311)
# ax.plot(sys_states_array[0])
# ax.set_title('theta')
# ax = plt.subplot(312)
# ax.plot(sys_states_array[1])
# ax.set_title('theta_dot')
# ax = plt.subplot(313)
# ax.plot(actions_array)
# ax.set_title('actions')



#asign controller
# cl_sys = ctl + sys
cl_sys.x0   = np.array([np.pi, 0.])
cl_sys.compute_trajectory( 10, 201, 'euler')
cl_sys.plot_trajectory('xu')
# cl_sys.plot_phase_plane_trajectory()
# cl_sys.animate_simulation()
