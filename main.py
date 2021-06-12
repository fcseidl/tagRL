"""
Driver which plays a tag game.
"""

import simple_agents
from nn_agents import NeuralAgent
import tag


# initialize game
game = tag.Game(animation_title='tag')
#game = tag.Game()

# initialize agents

#red_agent = simple_agents.KeyboardAgent('wasd')
red_agent = simple_agents.GreedyAgent()
#red_agent = NeuralAgent('red')
#red_agent.startGame(state)
blue_agent = simple_agents.KeyboardAgent('arrows')
#blue_agent = simple_agents.GreedyAgent()
#blue_agent = NeuralAgent('blue')
#blue_agent.startGame(state)

# main loop
continuing = True
while continuing:
    state = game.getState()
    r_control = red_agent.control(state)
    b_control = blue_agent.control(state)
    continuing = game.timestep(r_control, b_control)
    
