import numpy as np
from SwingyMonkey import SwingyMonkey


class Learner(object):
    """
    This agent jumps randomly.
    """

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.qtable = {}
        self.vertical_distance = np.arange(0, 400, step=40)
        self.horiz_dist = np.arange(-300, 600, step=90)
        self.vel_bin = np.arange(-30, 50, step=8)
        self.gravity = 0
        self.know_gravity = False
        self.actions = [0, 1]
        self.discount = 0.9
        self.alpha = 1  # learning rate
        self.count = 1  # parameter used to decay the learning rate over time

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.gravity = 0
        self.know_gravity = False
        self.count += 0.2  # decaying learning rate

    def learn_gravity(self, new_state, old_state):
        self.gravity = old_state['monkey']['vel'] - new_state['monkey']['vel']
        self.know_gravity = True

    def convert_state_to_reqstates(self, state, action):
        """
        discretize our states using binning
        """
        hdist = state['tree']['dist']
        vdist = state['monkey']['bot'] - state['tree']['bot']
        vel = state['monkey']['vel']

        hbin = np.digitize(hdist, self.horiz_dist)
        vbin = np.digitize(vdist, self.vertical_distance)
        vel_bin = np.digitize(vel, self.vel_bin)

        reqstate = [hbin, vbin, vel_bin, self.gravity, action]

        return "-".join(map(str, reqstate))

    def getQscore(self, state, action):
        """ Retrieves Q score for a given state, action """
        key = self.convert_state_to_reqstates(state, action)

        if self.qtable.has_key(key):
            return self.qtable[key]
        else:
            return 0

    def setQscore(self, state, action, value):
        """ updates Q score for a given action, state """
        key = self.convert_state_to_reqstates(state, action)
        self.qtable[key] = value

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        # You might do some learning here based on the current state and the last state.
        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        # initial case
        if state is None or self.last_action is None:
            self.last_state = state
            new_action = 0
        else:
            if not self.know_gravity:
                new_action = 0
                # Need to have two non jumps in a row to measure gravity correctly
                if self.last_action == 0 and new_action == 0:
                    self.learn_gravity(state, self.last_state)

            # Implementing Q learning
            prevQ = self.getQscore(self.last_state, self.last_action)
            newQval = [(prevQ + (self.alpha/self.count) * (self.last_reward +
                       self.discount * self.getQscore(state, act) - prevQ)) for
                       act in self.actions]
            new_action = np.argmax(newQval)
            new_Q = np.max(newQval)
            self.setQscore(self.last_state, self.last_action, new_Q)

        self.last_action = new_action
        self.last_state = state

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
                             tick_length=t_len,            # Make game ticks super fast.
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
    run_games(agent, hist, 1000, 1)
    # Save history.
    np.save('hist', np.array(hist))
