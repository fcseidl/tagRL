"""
This module contains agents which play by keyboard input or by a simple 
hardcoded rule.
"""

from pygame.constants import K_w, K_a, K_s, K_d, K_DOWN, K_LEFT, K_RIGHT, K_UP

from tag import *


def roundToIntercardinal(theta):
    theta %= (2 * np.pi)
    idx = int(4 * theta / np.pi + 0.5)
    return INTERCARDINALS[idx % 8]


class KeyboardAgent:
    """
    Control actions are determined by arrow keys (default)
    or WASD.
    """

    def __init__(self, keys):
        if keys == 'wasd':
            self._key_n = K_w
            self._key_e = K_d
            self._key_s = K_s
            self._key_w = K_a
        else:
            self._key_n = K_UP
            self._key_e = K_RIGHT
            self._key_s = K_DOWN
            self._key_w = K_LEFT
    
    def action(self, state):
        try:
            n = e = 0
            keys = pygame.key.get_pressed()
            if keys[self._key_n]:
                n += 1
            if keys[self._key_w]:
                e -= 1
            if keys[self._key_s]:
                n -= 1
            if keys[self._key_e]:
                e += 1
            if n == 0 and e == 0:
                return 'coast'
            theta = np.arctan2(n, e) % (2 * np.pi)
            return roundToIntercardinal(theta)
        except pygame.error:
            print('[ERROR] MANUALLY CONTROLLED AGENT WITH NO ANIMATION.')
            exit()


class MagneticAgent:
    """
    Always move parallel to shortest line segment connecting 
    agent and opponent. Direction depends on who is it.
    """

    def action(self, state):
        r_equiv_poses = equivalentPoses(state[R_POS])
        b_equiv_poses = equivalentPoses(state[B_POS])
        D = pairwise_distances(r_equiv_poses, b_equiv_poses)
        idx = D.argmin()
        idx = np.unravel_index(idx, (9,9))
        rp = r_equiv_poses[idx[0]]
        bp = b_equiv_poses[idx[1]]
        direction = (bp - rp) * state[IT]
        theta = np.arctan2(-direction[1], direction[0]) % (2 * np.pi)
        return roundToIntercardinal(theta)


class RandomAgent:
    """Take actions at uniform random."""

    def action(self, state) -> str:
        return np.random.choice(ACTIONS)

class StillAgent:
    """Just sit still."""

    def action(self, state) -> str:
        return 'coast'


class CompositeAgent:
    """Give control to a random member of a set of agents each round."""

    def __init__(self, agents, weights) -> None:
        self._agents = agents
        self._p = np.array(weights) / sum(weights)

    def action(self, state) -> str:
        agent = np.random.choice(self._agents, p=self._p)
        return agent.action(state)
