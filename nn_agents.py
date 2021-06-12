"""
Use a neural network to predict the time until next tag given 
every possible next action; this informs actions.

TODO: time discounting
TODO: learn from another agent for bootstrapping
"""

from torch import nn, optim
import torch

from tag import *


# TODO: engage cuda?
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: the class that trains the network should be different from the one that 
# uses it for policy!


class NeuralAgent:
    """
    Base class for an agent which chooses actions using a neural network to predict
    the marginal reward associated with each action. TODO: subclasses for different 
    network structures. Currently, the network will have a single hidden layer.

    Params
    ------
    color : str
        'r' or 'b'
    hidden_dim : int, optional
        Dimension of single hidden layer. Default is 100.
    load_from : str, optional
        Name of file from which to load pretrained model. Model is initialized randomly
        if load_from is not specified.
    """

    def __init__(self, color, hidden_dim=100, load_from=None) -> None:
        self._color = color
        if load_from is not None:
            self._model = torch.load(load_from)
        else:
            self._model = nn.Sequential(
                nn.Linear(SMOOTH_STATE_DIM, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, N_ACTIONS)
            )
        self._loss_func = nn.MSELoss()
        self._optimizer = optim.SGD(self._model.parameters(), lr=1e-2)

    def dumpModel(self, dump_to) -> None:
        """Save the agent's trained network to a .pt file."""
        print('Saving learned model to %s' % dump_to)
        torch.save(self._model, dump_to)

    def startGame(self, init_state, _switch_if_blue=True) -> None:
        """Call this function when the agent starts a game."""
        # internally always represent ourself as red
        if _switch_if_blue and self._color[0] == 'b':
            init_state = switchColors(init_state)
        self._chase_duration = 0
        self._it = init_state[IT]
        self._smooth_state_history = []
        self._actions_taken = []
    
    def control(self, state) -> str:
        """
        Given a game state, predict the marginal rewards associated with each action, and
        return the most attractive action. Predictions are stored for future backprop. If 
        state[IT] has changed since the last decision point, update the network by backprop.

        This function should be called every timestep in the game.
        """
        # internally always represent ourself as red
        if self._color[0] == 'b':
            state = switchColors(state)

        # update weights if someone has been tagged
        try:
            if self._it != state[IT]:
                # 'it' player has changed, backprop step and lose remembered history
                self._updateWeights()
                self.startGame(state, _switch_if_blue=False)
            self._it = state[IT]
        except AttributeError:
            raise Exception('NeuralAgent control() function called before startGame()')

        # use model to choose next action
        smooth_state = trigStateEncoding(state)
        smooth_state = torch.tensor(smooth_state).float()
        predict_rewards = self._model(smooth_state)
        idx = predict_rewards.argmax()
        action = list(_control_directions.keys())[idx]

        # remember state and action idx for backprop
        self._smooth_state_history.append(smooth_state)
        self._actions_taken.append(idx)
        self._chase_duration += 1

        return action

    def _updateWeights(self) -> None:
        # Use prev state and chase duration to determine true reward.
        # Update weights using SGD.
        print('[AGENT] %s updates based on %d timesteps' % (self._color, self._chase_duration))
        for t in range(self._chase_duration):
            self._optimizer.zero_grad()
            a_t = self._actions_taken[t]
            s_t = self._smooth_state_history[t]
            pred = self._model(s_t)[a_t]
            truth = (t - self._chase_duration) * self._it
            truth = torch.tensor(truth).float()
            loss = self._loss_func(pred, truth)
            loss.backward()
            self._optimizer.step()

