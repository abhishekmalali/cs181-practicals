# Imports.
import numpy as np
import numpy.random as npr
import matplotlib.pylab as plt

from SwingyMonkey import SwingyMonkey


# tree_centers = []

class Learner(object):
    '''
    This agent does the right thing___________
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.Qtable = {} #'gravity_high':np.zeros((10,10,10,10,2)), 'gravity_low':np.zeros((10,10,10,10,2))}
        self.gravity = 0
        self.iter_counter = 0
        self.alpha = 1
        self.gamma = 0.9
        # Distance to tree, Monkey_center, Tree_center, Vertical_velocity, Jump(yes,no)

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.iter_counter = 0

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # 
        self.iter_counter += 1
        self.alpha = 1./self.iter_counter

        if self.iter_counter == 2:
            self.gravity = state['monkey']['vel'] - self.last_state['monkey']['vel']
            if not self.Qtable.has_key(int(self.gravity)):
                self.Qtable.update({int(self.gravity):np.zeros((10,10,10,10,2))})

        # Constants
        vel_bounds = (-50, 30)
        dist_bounds = (-150, 500)
        m_c_bounds = (0, 400)
        t_c_bounds = (100, 240)


        vel = state['monkey']['vel']
        distance = state['tree']['dist']
        monkey_center = state['monkey']['top']-state['monkey']['bot'])/2. + state['monkey']['bot']
        tree_center = state['tree']['top']-state['tree']['bot'])/2. + state['tree']['bot']

        for i in 
        if vel < vel_bounds[0]:
            vel_idx = 0
        else if vel < vel_bounds[1]:
            vel_idx = (vel - vel_bounds[0])/(vel_bounds[1] - vel_bounds[0])*9
        else:
            vel_idx = 9

        # tree_centers.append((state['tree']['top']-state['tree']['bot'])/2. + state['tree']['bot'])





        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        new_action = npr.rand() < 0.1
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        # print state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. 
    run_games(agent, hist, 100, 1)

    # Save history. 
    np.save('hist',np.array(hist))

    # plt.hist(tree_centers)
    # plt.show()


