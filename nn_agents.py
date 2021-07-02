"""
This module contains code for training neural networks to play tag.

TODO: use CUDA?
TODO: try rewarding variance in network output
"""

from numpy.lib.shape_base import tile
from torch import nn, optim
import torch

from tag import *


#### handling pytorch modules ####

def singleLayerTagNet(hidden_dim=100) -> nn.Module:
    """Create a randomly weighted network with one hidden layer."""
    print('[NETWORK] randomly initialized single-layer network with %d hidden neurons' % hidden_dim)
    return nn.Sequential(
        nn.Linear(STATE_DIM, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, N_ACTIONS),
        nn.Tanh()
    )

def deepTagNet(hidden_dim=100, hidden_layers=3) -> nn.Module:
    """Create a randomly weighted deep network."""
    print('[NETWORK] randomly initialized network with %d hidden layers of dimension %d' % (hidden_layers, hidden_dim))
    layers = [
        nn.Linear(STATE_DIM, hidden_dim),    # first hidden layer
        nn.Tanh()
    ]
    for _ in range(hidden_layers - 1):
        layers += [
            nn.Linear(hidden_dim, hidden_dim),      # subsequent hidden layers
            nn.Tanh()
        ]
    layers += [
        nn.Linear(hidden_dim, N_ACTIONS),           # output layer
        nn.Tanh()
    ]
    return nn.Sequential(*layers)

def unpickleTagNet(location) -> nn.Module:
    """Load a prestored network from a file."""
    print('[NETWORK] reloaded network from %s' % location)
    return torch.load(location)

def pickleTagNet(net, location) -> None:
    """Dump a trained network to a .pt file."""
    print('[NETWORK] dumped network to %s' % location)
    torch.save(net, location)


#### training constants #####

_game_duration = 40                         # seconds
_discount_rate = 0.15                        # per second
_reward_cutoff = 1e-2                       # rewards smaller than this are ignored
_learning_rate = 1e-3
_l2_penalty = 1e-4

# derived constants
_lam = _discount_rate**TICK                 # effective discount rate (per tick)
_tick_horizon = 1 + int(                    # time horizon in ticks
        np.log(_reward_cutoff) / np.log(_lam)
    )
_discount_factors = np.array([
    _lam**t for t in range(_tick_horizon)
])
# ugly normalization to account for imprecision
_discount_factors /= (_discount_factors * TICK * np.ones_like(_discount_factors)).sum()


#### SGD trainer ####

class TagNetTrainer:
    """This class updates a policy network by backprop as a game progresses."""

    _loss_func = nn.MSELoss()

    def __init__(self, net, pickle_loc) -> None:
        self._net = net
        self._optimizer = optim.Adam(self._net.parameters(), lr=_learning_rate, weight_decay=_l2_penalty)
        self._pickle_loc = pickle_loc
        self._reset()
    
    def _reset(self) -> None:
        self._ticks = 0
        self._state_history = []
        self._actions_taken = {'r': [], 'b': []}
        self._r_rewards = []

    def update(self) -> dict:
        """
        Update weights by backprop, returning average loss for predictions of 
        both players' rewards. Don't update based on decisions whose time 
        horizons are beyond end of game
        """
        r_tot_loss = b_tot_loss = 0
        n_updates = max(0, self._ticks - _tick_horizon)
        for it in range(1):   # TODO: magic num here
            for t in range(n_updates):
                rewards = np.array(self._r_rewards[t:t+_tick_horizon])
                rewards *= _discount_factors
                value = rewards.sum()
                rl, bl = self._updateBothPlayers(
                    state=self._state_history[t],
                    r_action=self._actions_taken['r'][t],
                    b_action=self._actions_taken['b'][t],
                    r_value=value   # lol r_value is the lvalue in this assignment
                )
                if it == 0:
                    r_tot_loss += rl
                    b_tot_loss += bl
        r_avg_loss = r_tot_loss / n_updates
        b_avg_loss = b_tot_loss / n_updates
        print('[TRAINING] update based on %d timesteps' % n_updates)
        print('[TRAINING] average red loss = %f' % r_avg_loss)
        print('[TRAINING] average blue loss = %f' % b_avg_loss)
        # discard remembered history
        self._reset()
        # return avg losses
        return {'r': r_avg_loss, 'b': b_avg_loss}

    def recordTimestep(self, state, r_action, b_action, r_reward) -> None:
        """
        Inform the trainer of the state and actions at each timestep, along with 
        reward to red player. (Blue reward is the negative.)
        """
        # update records
        self._ticks += 1
        self._state_history.append(state)
        self._actions_taken['r'].append(r_action)
        self._actions_taken['b'].append(b_action)
        self._r_rewards.append(r_reward)
    
    def _updateBothPlayers(self, state, r_action, b_action, r_value) -> tuple:
        # internally, we always represent the player whose action is being considered as red
        red_loss = self._updateRed(state, r_action, r_value)
        switched = switchColors(state)
        blue_loss = self._updateRed(switched, b_action, -r_value)
        return red_loss, blue_loss

    def _updateRed(self, state, r_action, r_value) -> float:
        # single stochastic update
        self._optimizer.zero_grad()
        action_idx = ACTIONS.index(r_action)
        state = torch.tensor(state).float()
        pred_values = self._net(state)
        pred_v = pred_values[action_idx]
        true_v = torch.tensor(r_value).float()
        loss = self._loss_func(pred_v, true_v) #- 1e-1 * pred_v          # TODO: should we encourage high predictions?
        loss.backward()
        self._optimizer.step()
        return float(loss)
    
    def __del__(self) -> None:
        print('[TRAINING] pickling trained network')
        pickleTagNet(self._net, self._pickle_loc)


class NeuralAgent:
    """This class plays tag using a policy network."""

    def __init__(self, network, color='r') -> None:
        self._color = color
        self._net = network
    
    def set_color(self, color) -> None:
        """Change what color we control."""
        self._color = color
    
    def action(self, state) -> str:
        """
        Given a game state, predict the marginal rewards associated with each action, and
        return the most attractive action.
        """
        # internally always represent ourself as red
        if self._color[0] == 'b':
            state = switchColors(state)
        
        # use model to choose next action
        state = torch.tensor(state).float()
        predict_rewards = self._net(state)
        idx = predict_rewards.argmax()

        #print(self._color, predict_rewards)

        return ACTIONS[idx]


def trainingGame(trainer, red_agent, blue_agent, animation_title=None) -> bool:
    """
    Play a game between the two agents, and update a policy network with 
    a trainer. Return whether red won the game.
    """
    game = Game(animation_title=animation_title)
    continuing = True
    r_cum_reward = 0
    state = game.observableState()
    while game.getTime() < _game_duration and continuing:
        r_action = red_agent.action(state)
        b_action = blue_agent.action(state)
        continuing = game.timestep(r_action, b_action)
        new_state = game.observableState()
        r_reward = -new_state[IT] * norm(new_state[DISP]) * TICK * np.sqrt(2)       # TODO: this reward func knows too much
        trainer.recordTimestep(state, r_action, b_action, r_reward)
        r_cum_reward += r_reward
        state = new_state
    del game
    trainer.update()
    return r_cum_reward > 0


def train(network, red_agent, blue_agent, pickle_loc, animate_every=100) -> None:
    """Train a policy network by playing many games."""
    trainer = TagNetTrainer(network, pickle_loc)

    # play games until interupted or user kills window
    try:
        game_num = 0
        while True:
            game_num += 1
            title = None
            if game_num % animate_every == 0:
                title = 'game %d' % game_num
            red_wins = trainingGame(trainer, red_agent, blue_agent, animation_title=title)
            print('[TRAINING] %s won last game!' % 'red' if red_wins else 'blue')
    
    except (pygame.error, KeyboardInterrupt) as stop:
        print('[TRAINING] received stop signal of type', type(stop))

    print('[TRAINING] completed %d games' % game_num)


if __name__ == '__main__':
    import simple_agents

    # play vs self
    if 1:
        greedy_eps = 0.07
        print('[TRAINING] by self-play with epsilon-greedy parameter %f' % greedy_eps)

        net = deepTagNet()
        #net = unpickleTagNet('deep.pt')
        red_agent = simple_agents.CompositeAgent(
            [NeuralAgent(net, color='red'), simple_agents.RandomAgent()], 
            [1 - greedy_eps, greedy_eps]
        )
        blue_agent = simple_agents.CompositeAgent(
            [NeuralAgent(net, color='blue'), simple_agents.RandomAgent()], 
            [1 - greedy_eps, greedy_eps]
        )

        train(net, red_agent, blue_agent, pickle_loc='deep.pt', animate_every=75)

