"""
This module contains code for training neural networks to play tag.

TODO: use CUDA?
"""

from torch import nn, optim
import torch

from tag import *


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
    print('[NETWORK] reloaded network from %s.' % location)
    return torch.load(location)

def pickleTagNet(net, location) -> None:
    """Dump a trained network to a .pt file."""
    print('[NETWORK] dumped network to %s.' % location)
    torch.save(net, location)


class TagNetTrainer:
    """This class updates a policy network by SGD as a game progresses."""

    _loss_func = nn.MSELoss()

    # TODO: magic numbers here
    _lengthscale = 50
    _time_cutoff = int(_lengthscale * np.arctanh(0.99))

    def __init__(self, net, pickle_loc) -> None:
        self._net = net
        self._optimizer = optim.SGD(self._net.parameters(), lr=5e-2)
        self._steps_recorded = 0
        self._steps = 0
        self._state_history = []
        self._actions_taken = {'r': [], 'b': []}
        self._pickle_loc = pickle_loc

    def reset(self) -> None:
        """Call when who is it changes."""
        # update weights by SGD
        print('[TRAINING] update based on %d timesteps.' % self._steps)
        for t in range(self._steps_recorded):
            self._updateBothPlayers(
                state=self._state_history[t],
                r_action=self._actions_taken['r'][t],
                b_action=self._actions_taken['b'][t],
                chase_duration=self._steps_recorded - t
            )
        # discard remembered game history
        self._steps_recorded = 0
        self._steps = 0
        self._state_history = []
        self._actions_taken = {'r': [], 'b': []}

    def recordTimestep(self, state, r_action, b_action) -> None:
        """
        Inform the trainer of the state and actions at each timestep. If no one has 
        been tagged for sufficiently long, update the network based on the 
        earliest recorded state and then discard this state.
        """
        # update records
        self._state_history.append(state)
        self._actions_taken['r'].append(r_action)
        self._actions_taken['b'].append(b_action)

        # has it been long enough to say that the it player failed?
        if self._steps == self._time_cutoff:
            print('[TRAINING] chase duration exceeds threshold of %d timesteps.' % self._time_cutoff)
        self._steps += 1
        if self._steps > self._time_cutoff:
            self._updateBothPlayers(
                state=self._state_history[0],
                r_action=self._actions_taken['r'][0],
                b_action=self._actions_taken['b'][0],
                chase_duration=self._steps_recorded
            )
            self._state_history.pop(0)
            self._actions_taken['r'].pop(0)
            self._actions_taken['b'].pop(0)
        else:
            self._steps_recorded += 1
    
    def _updateBothPlayers(self, state, r_action, b_action, chase_duration) -> None:
        # internally, we always represent the player whose action is being considered as red
        self._updateRed(state, r_action, chase_duration)
        switched = switchColors(state)
        self._updateRed(state, b_action, chase_duration)

    def _updateRed(self, state, r_action, chase_duration) -> None:
        # single stochastic update
        self._optimizer.zero_grad()
        action_idx = ACTIONS.index(r_action)
        smooth_state = smoothStateEncoding(state)
        pred_reward = self._net(smooth_state)[action_idx]
        true_reward = np.tanh(chase_duration / self._lengthscale) * -state[IT]
        true_reward = torch.tensor(true_reward).float()
        loss = self._loss_func(pred_reward, true_reward)
        loss.backward()
        self._optimizer.step()
    
    def __del__(self) -> None:
        print('[TRAINING] pickling trained network.')
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
        return ACTIONS[idx]


def trainBySelfPlay(network, pickle_loc, animate=False) -> None:
    """Train a policy network by repeated self play."""
    # initialization
    game = Game(animation_title='self-play' if animate else None)
    red_agent = NeuralAgent('red', network)
    #blue_agent = NeuralAgent('blue', network)
    import simple_agents
    blue_agent = simple_agents.KeyboardAgent('arrows')
    trainer = TagNetTrainer(network, pickle_loc)

    # play game
    continuing = True
    prev_it = game.getState()[IT]
    while continuing:
        state = game.getState()

        # did someone get tagged?
        if state[IT] != prev_it:
            trainer.reset()
        prev_it = state[IT]

        # get actions and do a timestep
        r_action = red_agent.action(state)
        b_action = blue_agent.action(state)
        continuing = game.timestep(r_action, b_action)
        
        # update trainer
        trainer.recordTimestep(state, r_action, b_action)


def trainVsSimple(network, pickle_loc, animate=False) -> None:
    import simple_agents
    simple = simple_agents.CompositeAgent(
        [simple_agents.RandomAgent(), simple_agents.MagneticAgent()],
        [1, 10]
    )

    # initialization
    game = Game(animation_title='net vs greedy' if animate else None)
    red_agent = NeuralAgent('red', network)
    blue_agent = simple
    trainer = TagNetTrainer(network, pickle_loc)

    continuing = True
    prev_it = -1
    while continuing:
        state = game.getState()

        # did someone get tagged?
        if state[IT] != prev_it:
            trainer.reset()
            if state[IT] == 1:
                red_agent = simple
                blue_agent = NeuralAgent('blue', network)
            else:
                red_agent = NeuralAgent('red', network)
                blue_agent = simple
        prev_it = state[IT]

        # get actions and do a timestep
        r_action = red_agent.action(state)
        b_action = blue_agent.action(state)
        continuing = game.timestep(r_action, b_action)
        
        # update trainer
        trainer.recordTimestep(state, r_action, b_action)


if __name__ == '__main__':
    net = singleLayerTagNet()
    trainVsSimple(net, pickle_loc='singleLayer.pt', animate=False)
    #trainBySelfPlay(net, pickle_loc='singleLayer.pt', animate=True)

