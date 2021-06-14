"""
Contains tag simulator and related functions and constants.

TODO: agent time discretization needn't depend on framerate
"""

import numpy as np
from numpy.random import randint, rand
from numpy.linalg import norm
from sklearn.metrics import pairwise_distances
import pygame
from pygame.constants import K_q
from torch import tensor


#### graphics settings ####

_color_red = (255,0,0)
_color_blue = (0,0,255)
_color_tag_radius = (150,150,150)
_color_background = (255,255,255)


#### spatial and temporal game constants ####

# NOTE: if there is too much drag, greedy control will be nearly unbeatable,
# and the game won't be interesting.

_wrap_dist = 500                  # game occurs on torus
_control_force_coef = 1e3        # determines agents' acceleration capability
_drag_coef = 5e-3                # determines drag strength, caps max speed
_grace_period = 1                # during which there are no tagbacks
_tag_radius = 20                  # how far 'it' player can reach
_tick = 0.1                       # period of game clock
_euler_step = 2e-2               # determines fidelity of euler's method


#### action space discretization ####

_action_directions = {
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

ACTIONS = list(_action_directions.keys())
N_ACTIONS = len(ACTIONS)   # 9

INTERCARDINALS = ['e','ne','n','nw','w','sw','s','se']


#### indexing into state vector ####

# where to find each agent's position and velocity, 
# and who is it (1 -- red, -1 -- blue), in state vector
R_POS = slice(0, 2)
B_POS = slice(2, 4)
R_VEL = slice(4, 6)
B_VEL = slice(6, 8)
IT = 8

STATE_DIM = 9


def switchColors(state) -> np.ndarray:
    """Return a copy of the state with red and blue switched."""
    result = state.copy()
    result[R_POS] = state[B_POS]
    result[R_VEL] = state[B_VEL]
    result[B_POS] = state[R_POS]
    result[B_VEL] = state[R_VEL]
    result[IT] = -state[IT]
    return result


# public for greedy agent's use
def equivalentPoses(pose) -> np.ndarray:
    """Get nine equivalent poses due to world wrapping."""
    result = []
    steps = np.array([-1, 0, 1])
    for up in steps:
        for right in steps:
            step = np.array([up, right])
            result.append(pose + _wrap_dist * step)
    return np.array(result)


#### game engine ####

class Game:
    """
    Maintain and evolve state of tag game according to agents' actions.
    Initial positions and who is it are chosen randomly.

    If animation_title is a string, animate the game in a pygame window with the given
    title.
    """

    def __init__(self, animation_title=None) -> None:
        print('[GAME] starting new game')
        self._state = np.zeros(STATE_DIM)
        self._state[R_POS] = randint(_wrap_dist, size=2)
        self._state[B_POS] = randint(_wrap_dist, size=2)
        self._state[IT] = 2 * (rand() > 0.5) - 1
        self._time = 0
        self._last_tag = -1 - _grace_period

        self._animate = False
        if animation_title is not None:
            pygame.init()
            self._screen = pygame.display.set_mode((_wrap_dist, _wrap_dist))
            pygame.display.set_caption(animation_title)
            self._animate = True
            self._clock = pygame.time.Clock()

    
    def getState(self) -> np.ndarray:
        """Return the current game state."""
        return self._state

    def timestep(self, r_action, b_action) -> bool:
        """
        Progress the game by one tick of time. Print a message if a player is tagged.
        Update the pygame window if this game is animated.

        Return boolean indicating whether or not to halt the program.
        """
        # action force directions for both agents
        r_u = _action_directions[r_action]
        b_u = _action_directions[b_action]

        # numerically simulate one tick
        stop_time = self._time + _tick
        while self._time < stop_time:

            # update red
            r_speed = norm(self._state[R_VEL])
            self._state[R_POS] += _euler_step * self._state[R_VEL]
            self._state[R_VEL] += _euler_step * (
                _control_force_coef * r_u                    # control
                - _drag_coef * r_speed * self._state[R_VEL]  # drag
            )

            # update blue
            b_speed = norm(self._state[B_VEL])
            self._state[B_POS] += _euler_step * self._state[B_VEL]
            self._state[B_VEL] += _euler_step * (
                _control_force_coef * b_u                    # control
                - _drag_coef * b_speed * self._state[B_VEL]  # drag
            )

            # update who is it
            r_equiv_poses = equivalentPoses(self._state[R_POS])
            b_equiv_poses = equivalentPoses(self._state[B_POS])
            if self._time > self._last_tag + _grace_period:
                dist = pairwise_distances(r_equiv_poses, b_equiv_poses).min()
                if dist < _tag_radius:
                    tagger, taggee = ('red', 'blue') if self._state[IT] == 1 else ('blue', 'red')
                    print('[GAME] %s tagged %s at time %f' % (tagger, taggee, self._time))
                    self._state[IT] *= -1
                    self._last_tag = self._time

            # world wrapping
            self._state[R_POS] %= _wrap_dist
            self._state[B_POS] %= _wrap_dist

            self._time += _euler_step
        
        # potentially animate new frame
        if self._animate:
            surf = pygame.Surface((_wrap_dist, _wrap_dist))
            surf.fill(_color_background)
            if self._state[IT] > 0:   # red is it
                for rp, bp in zip(r_equiv_poses, b_equiv_poses):
                    pygame.draw.circle(surf, _color_tag_radius, rp, _tag_radius)
                    pygame.draw.circle(surf, _color_blue, bp, _tag_radius / 2)
                    pygame.draw.circle(surf, _color_red, rp, _tag_radius / 2)
            else:
                for rp, bp in zip(r_equiv_poses, b_equiv_poses):
                    pygame.draw.circle(surf, _color_tag_radius, bp, _tag_radius)
                    pygame.draw.circle(surf, _color_red, rp, _tag_radius / 2)
                    pygame.draw.circle(surf, _color_blue, bp, _tag_radius / 2)
            self._screen.blit(surf, (0,0))
            self._clock.tick(1 / _tick)
            pygame.display.update()
            # quit when window is closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
        
        return True
    
    def __del__(self) -> None:
        print('[GAME] terminating game')


#### smooth representation of states for neural networks ####

def _trigCoord(x):
    x *= 2 * np.pi / _wrap_dist
    return np.cos(x), np.sin(x)
def _trigPose(pose):
    c0, s0 = _trigCoord(pose[0])
    c1, s1 = _trigCoord(pose[1])
    return np.array([c0, s0, c1, s1])

# Use these for indexing into trig represented states
# R_VEL, B_VEL, and IT work for smooth and non-smooth representations
SMOOTH_STATE_DIM = STATE_DIM + 4
SMOOTH_R_POS = slice(0, 4)
SMOOTH_B_POS = slice(STATE_DIM, SMOOTH_STATE_DIM)

def smoothStateEncoding(state):
    """
    Replace discontinuous rectangular 2D coordinates with smooth 4D 
    trigonometric coordinates. The state space is a torus is parameterized 
    by two angles, so the smooth 4D coordinates consist of two sine, cosine
    pairs.
    """
    result = np.empty(SMOOTH_STATE_DIM)
    result[:STATE_DIM] = state
    result[SMOOTH_R_POS] = _trigPose(state[R_POS])
    result[SMOOTH_B_POS] = _trigPose(state[B_POS])
    return tensor(result).float()

