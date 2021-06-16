"""
This module contains code for training neural networks to play tag.

TODO: use CUDA?
"""

from numpy.core.defchararray import title
from numpy.lib.shape_base import tile
from torch import nn, optim
import torch

from tag import *


#### handling pytorch modules ####

def singleLayerTagNet(hidden_dim=100) -> nn.Module:
    """Create a randomly weighted network with one hidden layer."""
    print('[NETWORK] randomly initialized single-layer network with %d hidden neurons.' % hidden_dim)
    return nn.Sequential(
        nn.Linear(SMOOTH_STATE_DIM, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, N_ACTIONS),
        nn.Tanh()
    )

def unpickleTagNet(location) -> nn.Module:
    """Load a prestored network from a file."""
    print('[NETWORK] reloaded network from %s' % location)
    return torch.load(location)

def pickleTagNet(net, location) -> None:
    """Dump a trained network to a .pt file."""
    print('[NETWORK] dumped network to %s' % location)
    torch.save(net, location)


#### training constants #####

_game_duration = 30                         # seconds
_discount_rate = 0.4                        # per second
_reward_cutoff = 5e-3                       # rewards smaller than this are ignored
_learning_rate = 1e-3
_l2_penalty = 1e-5

# derived constants
_lam = _discount_rate**TICK                 # effective discount rate (per tick)
_tick_horizon = 1 + int(                    # time horizon in ticks
        np.log(_reward_cutoff) / np.log(_lam)
    )
_discount_factors = -np.log(_discount_rate) * np.array([
    _lam**t for t in range(_tick_horizon)
])


#### SGD trainer ####

class TagNetTrainer:
    """This class updates a policy network by SGD as a game progresses."""

    _loss_func = nn.MSELoss()

    def __init__(self, net, pickle_loc) -> None:
        self._net = net
        self._optimizer = optim.SGD(self._net.parameters(), lr=_learning_rate, weight_decay=_l2_penalty)
        self._pickle_loc = pickle_loc
        self._reset()
    
    def _reset(self) -> None:
        self._ticks = 0
        self._state_history = []
        self._actions_taken = {'r': [], 'b': []}
        self._r_rewards = []

    def update(self) -> None:
        # update weights by SGD
        # don't update based on decisions whose time horizons are beyond end of game
        total_loss = 0
        n_updates = max(0, self._ticks - _tick_horizon)
        for t in range(n_updates):
            rewards = np.array(self._r_rewards[t:t+_tick_horizon])
            rewards *= _discount_factors
            value = rewards.sum()
            total_loss += self._updateBothPlayers(
                state=self._state_history[t],
                r_action=self._actions_taken['r'][t],
                b_action=self._actions_taken['b'][t],
                r_value=value   # lol r_value is the lvalue in this assignment
            )
        print('[TRAINING] update based on %d timesteps with average loss of %f' % (n_updates, total_loss / n_updates))
        # discard remembered history
        self._reset()

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
    
    def _updateBothPlayers(self, state, r_action, b_action, r_value) -> torch.Tensor:
        # internally, we always represent the player whose action is being considered as red
        red_loss = self._updateRed(state, r_action, r_value)
        switched = switchColors(state)
        blue_loss = self._updateRed(switched, b_action, -r_value)
        return (red_loss + blue_loss) / 2

    def _updateRed(self, state, r_action, r_value) -> torch.Tensor:
        # single stochastic update
        self._optimizer.zero_grad()
        action_idx = ACTIONS.index(r_action)
        smooth_state = smoothStateEncoding(state)
        pred_values = self._net(smooth_state)
        pred_v = pred_values[action_idx]
        true_v = torch.tensor(r_value).float()
        loss = self._loss_func(pred_v, true_v)
        loss.backward()
        self._optimizer.step()
        return loss
    
    def __del__(self) -> None:
        print('[TRAINING] pickling trained network')
        pickleTagNet(self._net, self._pickle_loc)


class NeuralAgent:
    """This class plays tag using a policy network."""

    def __init__(self, color, network) -> None:
        self._color = color
        self._net = network
    
    def action(self, state) -> str:
        """
        Given a game state, predict the marginal rewards associated with each action, and
        return the most attractive action. Predictions are stored for future backprop. If 
        state[IT] has changed since the last decision point, update the network by backprop.
        
        This function should be called every timestep in the game.
        """
        # internally always represent ourself as red
        if self._color[0] == 'b':
            state = switchColors(state)
        
        # use model to choose next action
        smooth_state = smoothStateEncoding(state)
        predict_rewards = self._net(smooth_state)
        idx = predict_rewards.argmax()

        #print(self._color, predict_rewards)

        return ACTIONS[idx]


def train(network, red_agent, blue_agent, pickle_loc, animate_every=100) -> None:
    """Train a policy network by playing many games."""
    trainer = TagNetTrainer(network, pickle_loc)

    # play games until interupted or user kills window
    try:
        game_num = 0
        continuing = True
        while continuing:
            # start new game
            title = None
            if game_num % animate_every == 0:
                if game_num > 0:
                    print('[TRAINING] red was it at the end of %d out of the last %d games' % (red_losses, animate_every))
                title = 'game %d' % game_num
                red_losses = 0
            game = Game(animation_title=title)
            # game loop
            state = game.getState()
            while game.getTime() < _game_duration and continuing:
                r_action = red_agent.action(state)
                b_action = blue_agent.action(state)
                continuing = game.timestep(r_action, b_action)
                state = game.getState()
                r_reward = -state[IT] * TICK
                trainer.recordTimestep(state, r_action, b_action, r_reward)
            del game
            game_num += 1
            if state[IT] == 1:  # was red it at the end?
                red_losses += 1
            # backprop network
            trainer.update()
        
        print('[TRAINING] stopped because user killed game window')
    
    except (pygame.error, KeyboardInterrupt) as stop:
        print('[TRAINING] received stop signal of type', type(stop))

    print('[TRAINING] completed %d games of self-play' % game_num)



if __name__ == '__main__':
    import simple_agents

    # self-play
    if 0:
        greedy_eps = 0.05
        print('[TRAINING] starting self-play with epsilon-greedy parameter %f' % greedy_eps)

        net = singleLayerTagNet()
        red_agent = simple_agents.CompositeAgent(
            [NeuralAgent('red', net), simple_agents.RandomAgent()], 
            [1 - greedy_eps, greedy_eps]
        )
        blue_agent = simple_agents.CompositeAgent(
            [NeuralAgent('blue', net), simple_agents.RandomAgent()],
            [1 - greedy_eps, greedy_eps]
        )

        train(net, red_agent, blue_agent, pickle_loc='singleLayer.pt')
    
    # play vs greedy
    if 0:
        print('[TRAINING] network (red) vs magnetic agent (blue)')

        net = singleLayerTagNet()
        red_agent = NeuralAgent('red', net)
        blue_agent = simple_agents.MagneticAgent()

        train(net, red_agent, blue_agent, pickle_loc='singleLayer.pt')
    
    # play vs greedy
    if 1:
        greedy_eps = 0.1
        print('[TRAINING] network (red) with eps-greedy parameter %f vs random agent (blue)' % greedy_eps)

        #net = singleLayerTagNet()
        net = unpickleTagNet('singleLayer.pt')
        red_agent = simple_agents.CompositeAgent(
            [NeuralAgent('red', net), simple_agents.RandomAgent()], 
            [1 - greedy_eps, greedy_eps]
        )
        blue_agent = simple_agents.RandomAgent()

        train(net, red_agent, blue_agent, pickle_loc='singleLayer.pt', animate_every=100)

