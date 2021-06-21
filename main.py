"""
Driver which plays a tag game.
"""

import simple_agents
import nn_agents
import tag


# initialize game
game = tag.Game(animation_title='tag')
#game = tag.Game()

# initialize agents

red_agent = simple_agents.KeyboardAgent('wasd')
'''blue_agent = simple_agents.CompositeAgent(
    [simple_agents.RandomAgent(), simple_agents.MagneticAgent()],
    [1, 10]
)'''
net = nn_agents.unpickleTagNet('deep.pt')
blue_agent = nn_agents.NeuralAgent('blue', net)
#blue_agent = simple_agents.KeyboardAgent('arrows')

# main loop
continuing = True
while continuing:
    state = game.observableState()
    r_action = red_agent.action(state)
    b_action = blue_agent.action(state)
    continuing = game.timestep(r_action, b_action)
    
