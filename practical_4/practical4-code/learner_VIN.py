# Imports.
import numpy as np
# import numpy.random as npr
# import matplotlib.pylab as plt

from SwingyMonkey import SwingyMonkey


class Learner(object):
    """
    This agent does the right thing___________
    """

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        # 'gravity_high':np.zeros((10, 10, 10, 10, 2)),
        # 'gravity_low':np.zeros((10, 10, 10, 10, 2))}
        self.Qtable = np.zeros((10, 10, 10, 10, 2))
        self.gravity = 0
        self.iter_counter = 0
        self.alpha = 0.4
        self.gamma = 0.9
        # Distance to tree, Monkey_center, Tree_center, Vertical_velocity, Jump(no, yes)

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.iter_counter = 0

    def _compute_indices(self, state):
        # Constants
        vel_bounds = (-50, 30)
        dist_bounds = (-150, 500)
        m_c_bounds = (0, 400)
        t_c_bounds = (100, 240)

        vel = state['monkey']['vel']
        distance = state['tree']['dist']
        monkey_center = (state['monkey']['top'] - state['monkey']['bot'])/2. + state['monkey']['bot']
        tree_center = (state['tree']['top'] - state['tree']['bot'])/2. + state['tree']['bot']

        vel_idx = -1
        dist_idx = -1
        m_c_idx = -1
        t_c_idx = -1

        # compute indices for given state
        for i in [[vel, vel_bounds, vel_idx], [distance, dist_bounds, dist_idx],
                  [monkey_center, m_c_bounds, m_c_idx],  [tree_center, t_c_bounds, t_c_idx]]:
            if i[0] < i[1][0]:
                i[2] = 0
            elif i[0] < i[1][1]:
                i[2] = (i[0] - i[1][0])/(i[1][1] - i[1][0])*9
            else:
                i[2] = 9

        return vel_idx, dist_idx, m_c_idx, t_c_idx

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """
        self.iter_counter += 1
        # self.alpha = 1./self.iter_counter

        if self.iter_counter < 2:
            new_action = 0
            new_state = state
            self.last_action = new_action
            self.last_state = new_state

            return self.last_action

        # if self.iter_counter == 2:
        #     gravity = state['monkey']['vel'] - self.last_state['monkey']['vel']
        #     if int(gravity) != int(self.gravity):
        #
        #     if not self.Qtable.has_key(int(self.gravity)):
        #         self.Qtable.update({int(self.gravity):np.zeros((10,10,10,10,2))})

        # get indices for old state
        vel_idx, dist_idx, m_c_idx, t_c_idx = self._compute_indices(self.last_state)
        # get indices for new state
        vel_idx_new, dist_idx_new, m_c_idx_new, t_c_idx_new = self._compute_indices(state)

        if self.Qtable[dist_idx_new, m_c_idx_new, t_c_idx_new, vel_idx_new, 0] \
         > self.Qtable[dist_idx_new, m_c_idx_new, t_c_idx_new, vel_idx_new, 1]:
            new_action = 0
        else:
            new_action = 1

        # Q learning update equation
        self.Qtable[dist_idx, m_c_idx,
                    t_c_idx, vel_idx,
                    self.last_action] = ((1 - self.alpha) * self.Qtable[dist_idx,
                                        m_c_idx, t_c_idx, vel_idx, self.last_action]) \
                                        + (self.alpha * (self.last_reward +
                                        self.gamma * self.Qtable[dist_idx_new,
                                        m_c_idx_new, t_c_idx_new, vel_idx_new, new_action]))

        # You might do some learning here based on the current state and the last state.
        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        # new_action = npr.rand() < 0.1
        new_state = state

        self.last_action = new_action
        self.last_state = new_state

        return self.last_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length=t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

    print(np.amin(learner.Qtable))
    print(np.amax(learner.Qtable))

    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games.
    run_games(agent, hist, 100, 1)

    print max(hist)
    # Save history.
    np.save('hist', np.array(hist))

    # plt.hist(tree_centers)
    # plt.show()
