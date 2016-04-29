# Imports.
import numpy as np
import numpy.random as npr
from SwingyMonkey import SwingyMonkey
from sklearn.ensemble import RandomForestRegressor

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.regressor = None
        self.epsilon = 0.5
        self.alpha = 0.8
        self.gamma = 0.5
        self.know_gravity = False
        self.gravity = 4
        self.model_trained = False
        self.actions = [False, True]
        self.model = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.regressor = None
        self.know_gravity = False
        self.gravity = 4
        self.model_trained = True

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        
        new_state  = state
        state_array = np.array(state['tree'].values()+ state['monkey'].values()+[self.gravity])
        if self.model_trained == False:
            new_action = npr.rand() < 0.1
        else:
            if np.random.uniform() < self.epsilon:
                new_action = npr.rand() < 0.1
            else:
                new_action = np.argmax([self.model.predict(np.append(state_array, int(action))) for action in self.actions])

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

    def learn_gravity(self, state_data, actions):
        #take last two actions are not jumps, we can infer gravity
        if np.sum(actions[-2:]) == 0:
            self.gravity = state_data[-2][3] - state_data[-1][3]
            self.know_gravity = True


def build_training_set(state_data, actions):
    ret_arr =  np.array([np.append(state_data[k], actions[k]) \
            for k in range(len(actions))])
    return ret_arr

def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    net_states = []
    net_rewards = []
    net_actions = []
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,       # Don't play sounds.
                             text="Epoch %d" % (ii),# Display the epoch on screen.
                             tick_length = t_len,# Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        iter_states = []
        iter_rewards = []
        iter_actions = []
        iter_count = 0
        while swing.game_loop():
            state = swing.get_state()
            iter_states.append(np.array(state['tree'].values()+\
                    state['monkey'].values()+[learner.gravity]))
            iter_rewards.append(learner.last_reward)
            iter_actions.append(int(learner.last_action))
            iter_count += 1
            if iter_count > 1 and learner.know_gravity == False:
                learner.learn_gravity(iter_states, iter_actions)
                if learner.know_gravity == True:
                    for num in range(len(iter_states)):
                        iter_states[num][-1] = learner.gravity
        #To get the state after the 
        state = swing.get_state()
        iter_states.append(state['tree'].values()+\
                    state['monkey'].values()+[learner.gravity])
        iter_rewards.append(learner.last_reward)
        iter_actions.append(int(learner.last_action))
        
        #Adding to the net training set
        net_states += iter_states
        net_rewards += iter_rewards
        net_actions += iter_actions
        
        if ii == 0:
            xtrain = build_training_set(net_states, net_actions)
            ytrain = np.array(net_rewards)
            RF = ExtraTreesRegressor(n_estimators = 50)  
            RF.fit(xtrain, ytrain)

        else:
            xtrain = build_training_set(net_states[:-1], net_actions[:-1])
            #Building the q_state update.
            ytrain = np.array([learner.model.predict(np.append(net_states[k], net_actions[k])) + \
                    learner.alpha*(net_rewards[k] + learner.gamma* np.max([learner.model.predict(np.append(net_states[k+1], int(action)))\
                            for action in learner.actions]) - \
                    learner.model.predict(np.append(net_states[k], net_actions[k]))) for k in range(len(net_states)-1)]) 
            RF = ExtraTreesRegressor(n_estimators = 50)
            RF.fit(xtrain, ytrain)
            
        learner.model = RF
        learner.model_trained = True


        if ii%10 == 0:
            learner.epsilon -= 0.05
        
         
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
	run_games(agent, hist, 100, 10)

	# Save history. 
	np.save('hist',np.array(hist))


