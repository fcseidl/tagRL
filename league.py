"""
Use league play to train a policy network.
"""

from tag import *
import simple_agents
import nn_agents

# initialize learning setup
net = nn_agents.deepTagNet()
player = nn_agents.NeuralAgent(net, color='red')
trainer = nn_agents.TagNetTrainer(net)

# initialize league opponents
opponents = {
    'still': simple_agents.StillAgent(),
    'random': simple_agents.RandomAgent(),
    'magnet': simple_agents.MagneticAgent(),
    'network': nn_agents.NeuralAgent(net, color='blue')
}
win_counts = {}
for opp in opponents.keys():
    win_counts[opp] = 1


# training loop: play games until interupted or user kills window
animate_every = 75
try:
    game_num = 1
    while True:
        names = list(win_counts.keys()) 
        weights = np.array(list(win_counts.values()))
        opp_name = np.random.choice(names, p=weights/weights.sum())
        title = None
        if game_num % animate_every == 0:
            title = 'game %d vs %s' % (game_num, opp_name)
        red_wins = nn_agents.trainingGame(trainer, player, opponents[opp_name], animation_title=title)
        print('[TRAINING] %s won last game!' % ('red' if red_wins else 'blue (%s)' % opp_name))
        if not red_wins:
            win_counts[opp_name] += 1
        game_num += 1

except (pygame.error, KeyboardInterrupt) as stop:
    print('[TRAINING] received stop signal of type', type(stop))

print('[TRAINING] completed %d games' % game_num)

nn_agents.pickleTagNet(net, 'deep.pt')



