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
        dx[0] = angle_normalize(dx[0] + np.pi)
        
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
        
        dx[0] = angle_normalize(dx[0] + np.pi)
        
        dJ = ( np.dot( dx.T , np.dot(  self.Q , dx ) ) +
               np.dot( du.T , np.dot(  self.R , du ) ) )
        
        # Set cost to zero if on target
        if self.ontarget_check:
            if ( np.linalg.norm( dx ) < self.EPS ):
                dJ = 0
        
        return dJ

sys  = pendulum.SinglePendulum()

sys.x_ub = np.array([+np.pi, +10])
sys.x_lb = np.array([-np.pi,  -10])
sys.u_ub = np.array([1])
sys.u_lb = np.array([-1])
sys.m1 = 1.0
sys.lc1 = 0.4
sys.I1 = 0

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [201,201] , [21] )



# Cost Function
qcf = CustomCostFunction(2, 1)

qcf.xbar = np.array([ 0 , 0 ]) # target
qcf.INF  = 500

qcf.Q[0,0] = 1.0
qcf.Q[1,1] = 0.1
qcf.R[0,0] = 0.001

qcf.S[0,0] = 10.0
qcf.S[1,1] = 10.0



# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)
#dp = dprog.DynamicProgramming2DRectBivariateSpline(grid_sys, qcf)

#dp.solve_bellman_equation( animate_cost2go = True )

dp.compute_steps(200)
# dp.plot_policy()

#dp.solve_bellman_equation( tol = 1)
# dp.solve_bellman_equation( tol = 0.1 , animate_cost2go = True )
# dp.solve_bellman_equation( tol = 1 , animate_policy = True )
#dp.plot_cost2go(150)

#dp.animate_cost2go( show = False , save = True )
#dp.animate_policy( show = False , save = True )

dp.clean_infeasible_set()
dp.plot_cost2go_3D()
dp.plot_policy()

ctl = dp.get_lookup_table_controller()
# print(ctl)


U = ctl.plot_control_law( sys = sys , n = 100)
np.save('vi_policy_real', U)

#asign controller
cl_sys = ctl + sys
cl_sys.x0   = np.array([0., 0.])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()
