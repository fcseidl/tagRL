"""
Use a neural network to predict the time until next tag given 
every possible next action; this informs actions.
"""

from torch import nn
import torch
from torch.nn.modules.activation import ReLU

from tag_spec import *


# TODO: engage cuda?
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# create and (un)pickle models

def singleLayerModel(hidden_dim=100, load_from=None):
    """Either load or randomly initialize a single layer model."""
    if load_from is not None:
        model = torch.load(load_from)
    else:
        model = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS)
        )
    return model

def dumpModel(model, dump_to):
    """Dump a model to a destination file."""
    print('Saving learned model to %s' % dump_to)
    torch.save(model, dump_to)



class NeuralAgent:

    def __init__(self, color, model, loss, optim):
        self._color = color
        self._model = model
        self._loss = loss
        self._optim = optim

    def startChase(self, state, _switch_if_blue=True):
        # internally always represent ourself as red
        if _switch_if_blue and self._color[0] == 'b':
            state = switchColors(state)
        self._chase_duration = 0
        self._prev_state = state
        self._predictions = []
        self._action_idxs = []
    
    def updateWeights(self):
        # use prev state and chase duration to determine reward
        pred = torch.tensor(self._predictions)
        truth = pred.copy()
        for t in range(self._chase_duration):
            truth[t, self._action_idxs[t]] = self._chase_duration - t
        # backprop
        L = self._loss(pred, truth)
        L.backward()
        self._optim.step()
    
    def control(self, state):
        # internally always represent ourself as red
        if self._color[0] == 'b':
            state = switchColors(state)
        # update weights if someone has been tagged
        try:
            if self._prev_state[IT] != state[IT]:
                self.updateWeights()
                self.startChase(state, _switch_if_blue=False)
        except:
            raise Exception('NeuralAgent control() function called before startChase()')
        # use model to choose next action
        state = torch.tensor(state).float()
        predict_rewards = self._model(state)
        idx = predict_rewards.argmax()
        action = list(control_direction.keys())[idx]
        # remember predict_rewards and idx for backprop
        self._predictions.append(predict_rewards)
        self._action_idxs.append(idx)
        self._chase_duration += 1
        return action

