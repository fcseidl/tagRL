"""
Driver which simulates tag game.

TODO: main loop is too long. More functions!
TODO: agent time discretization needn't depend on framerate
"""

import simple_agents
import nn_agents
from tag_spec import *


# animation settings
animate = True
COLOR_RED = (255,0,0)
COLOR_BLUE = (0,0,255)
COlOR_TAG_RADIUS = (150,150,150)
COLOR_WHITE = (255,255,255)

# initialize state
state = np.empty(STATE_DIM)
state[R_POS] = [25, 25]
state[R_VEL] = [0, 0]
state[B_POS] = [75, 75]
state[B_VEL] = [0, 0]
state[IT] = -1      # Blue
time = 0
last_tag = -1 - grace_period

# initialize agents

model = nn_agents.singleLayerModel()#load_from='singleLayer.pt')
import torch
loss = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

#red_agent = simple_agents.KeyboardAgent('wasd')
red_agent = simple_agents.GreedyAgent()
#red_agent = nn_agents.NeuralAgent('r', model, loss, optim)
#red_agent.startChase(state)
blue_agent = simple_agents.KeyboardAgent('arrows')
#blue_agent = simple_agents.GreedyAgent()
#blue_agent = nn_agents.NeuralAgent('b', model, loss, optim)
#blue_agent.startChase(state)

# initialize display
pygame.init()
if animate:
    screen = pygame.display.set_mode((wrap_dist, wrap_dist))
    clock = pygame.time.Clock()

# main loop
quitting = False
while not quitting:
    r_control = red_agent.control(state)
    b_control = blue_agent.control(state)
    r_u = control_direction[r_control]
    b_u = control_direction[b_control]

    # simulate one tick of time
    stop_time = time + 1. / framerate
    while time < stop_time:
        # update red
        r_speed = norm(state[R_VEL])
        state[R_POS] += euler_step * state[R_VEL]
        state[R_VEL] += euler_step * (
            control_force_coef * r_u                    # control
            - drag_coef * r_speed * state[R_VEL]        # drag
        )

        # update blue
        b_speed = norm(state[B_VEL])
        state[B_POS] += euler_step * state[B_VEL]
        state[B_VEL] += euler_step * (
            control_force_coef * b_u                    # control
            - drag_coef * b_speed * state[B_VEL]        # drag
        )

        # update who is it
        r_equiv_poses = equivalentPoses(state[R_POS])
        b_equiv_poses = equivalentPoses(state[B_POS])
        if time > last_tag + grace_period:
            dist = pairwise_distances(r_equiv_poses, b_equiv_poses).min()
            if dist < tag_radius:
                tagger, taggee = ('red', 'blue') if state[IT] == 1 else ('blue', 'red')
                print('%s tagged %s at time %f' % (tagger, taggee, time))
                state[IT] *= -1
                last_tag = time

        # world wrapping
        state[R_POS] %= wrap_dist
        state[B_POS] %= wrap_dist

        time += euler_step
    # end main loop

    # potentially render graphics
    if animate:
        surf = pygame.Surface((wrap_dist, wrap_dist))
        surf.fill(COLOR_WHITE)
        if state[IT] > 0:   # red is it
            for rp, bp in zip(r_equiv_poses, b_equiv_poses):
                pygame.draw.circle(surf, COlOR_TAG_RADIUS, rp, tag_radius)
                pygame.draw.circle(surf, COLOR_BLUE, bp, tag_radius / 2)
                pygame.draw.circle(surf, COLOR_RED, rp, tag_radius / 2)
        else:
            for rp, bp in zip(r_equiv_poses, b_equiv_poses):
                pygame.draw.circle(surf, COlOR_TAG_RADIUS, bp, tag_radius)
                pygame.draw.circle(surf, COLOR_RED, rp, tag_radius / 2)
                pygame.draw.circle(surf, COLOR_BLUE, bp, tag_radius / 2)
        screen.blit(surf, (0,0))
        clock.tick(framerate)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quitting = True

nn_agents.dumpModel(model, 'singleLayer.pt')
    
