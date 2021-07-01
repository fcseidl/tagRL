"""
Use league play to train a policy network.
"""

from tag import *
import simple_agents
import nn_agents

net = nn_agents.deepTagNet()
agents = {
    'still': simple_agents.StillAgent(),
    'random': simple_agents.RandomAgent(),
    'magnet': simple_agents.MagneticAgent(),
    'network': nn_agents.NeuralAgent(net)
}


