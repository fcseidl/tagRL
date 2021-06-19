"""
Contains tag simulator and related functions and constants.
"""

import numpy as np
from numpy.random import randint, rand
from numpy.linalg import norm
from sklearn.metrics import pairwise_distances
import pygame


#### graphics settings ####

_color_red = (255,0,0)
_color_blue = (0,0,255)
_color_tag_radius = (150,150,150)
_color_background = (255,255,255)


#### spatial and temporal game constants ####

# NOTE: if there is too much drag, simple control algorithms will be nearly unbeatable,
# and the game won't be interesting.

_wrap_dist = 500                 # game occurs on torus
_control_force_coef = 1e3        # determines agents' acceleration capability
_drag_coef = 5e-3                # determines drag strength, caps max speed
_grace_period = 1                # during which there are no tagbacks
_tag_radius = 20                 # how far 'it' player can reach
_euler_step = 2e-2               # determines fidelity of euler's method

TICK = 0.1                       # period of game clock. Public for RL agents.

if TICK / _euler_step != int(TICK / _euler_step):
    raise ValueError('Euler method steps should divide tick length.')


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

DISP = slice(0, 2)              # shortest displacement vector from red to blue
R_VEL = slice(2, 4)             # red velocity
B_VEL = slice(4, 6)             # blue velocity
IT = 6                          # who's it (1 => red, -1 => blue)

STATE_DIM = 7

# for internal state representation
_r_pos = slice(0, 2)
_b_pos = slice(7, 9)
_internal_dim = 9


def switchColors(state) -> np.ndarray:
    """Return a copy of the state with red and blue switched."""
    result = np.empty(STATE_DIM)
    result[DISP] = -state[DISP]
    result[R_VEL] = state[B_VEL]
    result[B_VEL] = state[R_VEL]
    result[IT] = -state[IT]
    return result


# get nine equivalent poses due to world wrapping
def _equivalentPoses(pose) -> np.ndarray:
    result = []
    steps = [-1, 0, 1]
    for up in steps:
        for right in steps:
            step = np.array([up, right])
            result.append(pose + _wrap_dist * step)
    return np.array(result)

# get displacement from one pose to another
def _displacement(p1, p2) -> np.ndarray:
    equiv_p1 = _equivalentPoses(p1)
    equiv_p2 = _equivalentPoses(p2)
    D = pairwise_distances(equiv_p1, equiv_p2)
    idx = D.argmin()
    idx = np.unravel_index(idx, (9,9))
    p1 = equiv_p1[idx[0]]
    p2 = equiv_p2[idx[1]]
    return p2 - p1


#### game engine ####

class Game:
    """
    Maintain and evolve state of tag game according to agents' actions.
    Initial positions and who is it are chosen randomly.

    If animation_title is a string, animate the game in a pygame window with the given
    title.
    """

    def __init__(self, animation_title=None) -> None:
        print('[GAME] new game with random starting positions')
        # internal state has absolute rather than relative postional data
        self._state = np.zeros(_internal_dim)
        self._state[_r_pos] = randint(_wrap_dist, size=2)
        self._state[_b_pos] = randint(_wrap_dist, size=2)
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
        # trim state to include only relative positional data
        result = np.empty(STATE_DIM)
        result[DISP] = _displacement(self._state[_r_pos], self._state[_b_pos])
        result[R_VEL] = self._state[R_VEL]
        result[B_VEL] = self._state[B_VEL]
        result[IT] = self._state[IT]
        return result

    def getTime(self) -> float:
        """Return the current game time."""
        return self._time

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
        stop_time = self._time + TICK
        while stop_time - self._time > _euler_step / 2:     # this stop condition avoids tick lengthening due to imprecision
            self._time = _euler_step * round(self._time / _euler_step + 1)

            # update red
            r_speed = norm(self._state[R_VEL])
            self._state[_r_pos] += _euler_step * self._state[R_VEL]
            self._state[R_VEL] += _euler_step * (
                _control_force_coef * r_u                    # control
                - _drag_coef * r_speed * self._state[R_VEL]  # drag
            )

            # update blue
            b_speed = norm(self._state[B_VEL])
            self._state[_b_pos] += _euler_step * self._state[B_VEL]
            self._state[B_VEL] += _euler_step * (
                _control_force_coef * b_u                    # control
                - _drag_coef * b_speed * self._state[B_VEL]  # drag
            )

            # update who is it
            r_equiv_poses = _equivalentPoses(self._state[_r_pos])
            b_equiv_poses = _equivalentPoses(self._state[_b_pos])
            if self._time > self._last_tag + _grace_period:
                dist = pairwise_distances(r_equiv_poses, b_equiv_poses).min()
                if dist < _tag_radius:
                    tagger, taggee = ('red', 'blue') if self._state[IT] == 1 else ('blue', 'red')
                    print('[GAME] %s tagged %s at time %f' % (tagger, taggee, self._time))
                    self._state[IT] *= -1
                    self._last_tag = self._time

            # world wrapping
            self._state[_r_pos] %= _wrap_dist
            self._state[_b_pos] %= _wrap_dist
        
        # potentially animate new frame
        # TODO: frame rate needn't be the same as agent time discretization
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
            self._clock.tick(1 / TICK)
            pygame.display.update()
            # quit when window is closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
        
        return True
    
    def __del__(self) -> None:
        print('[GAME] terminating game; %s is it' % ('red' if self._state[IT] == 1 else 'blue'))
        pygame.quit()

