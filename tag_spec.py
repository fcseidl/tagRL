"""
Specifies a tag simulation.
"""

import numpy as np
from numpy.linalg import norm
from sklearn.metrics import pairwise_distances
import pygame


# game constants
# NOTE: if there is too much drag, greedy control will be nearly unbeatable,
# and the game won't be interesting.
wrap_dist = 500                 # game occurs on torus
control_force_coef = 1e3        # determines agents' acceleration capability
drag_coef = 5e-3                # determines drag strength, caps max speed
grace_period = 1                # during which there are no tagbacks
tag_radius = 20                 # how far 'it' player can reach

# time discretization
euler_step = 1e-3
framerate = 10

# discretize angles into (inter)cardinal directions
intercardinals = ['e','ne','n','nw','w','sw','s','se']

# control discretization
control_direction = {
    'coast':    np.array([0,0]),
    'n':        np.array([0,-1]),
    'ne':       np.array([1,-1]) / np.sqrt(2),
    'e':        np.array([1,0]),
    'se':       np.array([1,1]) / np.sqrt(2),
    's':        np.array([0,1]),
    'sw':       np.array([-1,1]) / np.sqrt(2),
    'w':        np.array([-1,0]),
    'nw':       np.array([-1,-1]) / np.sqrt(2)
}
N_ACTIONS = len(control_direction.keys())   # 9

# where to find each agent's position and velocity, 
# and who is it (1 -- red, -1 -- blue), in state vector
R_POS = slice(0, 2)
R_VEL = slice(2, 4)
B_POS = slice(4, 6)
B_VEL = slice(6, 8)
IT = 8
STATE_DIM = 9

# return a copy of the state with red and blue switched
def switchColors(state):
    result = np.empty(STATE_DIM)
    result[R_POS] = state[B_POS]
    result[R_VEL] = state[B_VEL]
    result[B_POS] = state[R_POS]
    result[B_VEL] = state[R_VEL]
    result[IT] = -state[IT]
    return result

# get nine equivalent poses due to world wrapping
def equivalentPoses(x):
    result = []
    steps = np.array([-1, 0, 1])
    for up in steps:
        for right in steps:
            step = np.array([up, right])
            result.append(x + wrap_dist * step)
    return np.array(result)