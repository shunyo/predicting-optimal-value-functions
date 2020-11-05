""" Create gridworld """
import numpy as np
from matplotlib import pyplot as plt
class GridWorld:
    def __init__(self,wind, living_reward,positive_scalar_factor,negative_scalar_factor):
        self.gamma = 0.99
        self.size = (3,4)
        self.padded_size = (5,6) # padded in all directions
        self.wind = wind
        self.reward = np.zeros(self.padded_size)
        self.reward[1:1+self.size[0],1:1+self.size[1]] = np.ones(self.size)*living_reward # all states have reward provided by user
        self.reward[1,4] = +1*positive_scalar_factor
        self.reward[2,4] = -1*negative_scalar_factor
        self.wall = (2,2) # a wall in position
        self.reward[self.wall] = 0
        self.actions = 4 # N, S, E, W
        self.possible_states = np.ones(self.padded_size)*-1
        self.possible_states[1:1+self.size[0],1:1+self.size[1]] = np.zeros(self.size)
        self.possible_states[self.wall] = -1
        self.state_value_function = np.zeros(self.padded_size)
        self.action_value_function = np.zeros((5,6,4))
        self.policy = np.ones(self.padded_size)*-1

    def value_iteration(self):
        old_values = np.copy(self.state_value_function)
        old_values[0,:] = np.copy(old_values[1,:])
        old_values[:,0] = np.copy(old_values[:,1])
        old_values[4,:] = np.copy(old_values[3,:])
        old_values[:,5] = np.copy(old_values[:,4])
        for i in range(1,1+self.size[0]):
            for j in range(1,1+self.size[1]):
                if (i, j) == (1, 4) or (i, j) == (2, 4) or (i,j) == self.wall:
                    self.state_value_function[i,j] = self.reward[i,j]
                else:
                    value_actions = np.zeros((self.actions,))
                    for k in range(self.actions):
                        # not legal actions
                        if k == 0:
                        #    if self.possible_states[i,j+1] == -1:
                        #        continue
                            if (i,j+1) == self.wall:
                                value_actions[k] = (1-2*self.wind)*old_values[i,j] + self.wind*old_values[i-1,j] + self.wind*old_values[i+1,j]
                            elif (i-1,j) == self.wall:
                                value_actions[k] = (1-2*self.wind)*old_values[i,j+1] + self.wind*old_values[i,j] + self.wind*old_values[i+1,j]
                            elif (i+1,j) == self.wall:
                                value_actions[k] = (1-2*self.wind)*old_values[i,j+1] + self.wind*old_values[i-1,j] + self.wind*old_values[i,j]
                            else:
                                value_actions[k] = (1-2*self.wind)*old_values[i,j+1] + self.wind*old_values[i-1,j] + self.wind*old_values[i+1,j]
                        elif k == 1:
                        #    if self.possible_states[i+1,j] == -1:
                        #        continue
                            if (i+1,j) == self.wall:
                                value_actions[k] = (1-2*self.wind)*old_values[i,j] + self.wind*old_values[i,j-1] + self.wind*old_values[i, j+1]
                            elif (i,j-1) == self.wall:
                                value_actions[k] = (1-2 * self.wind) * old_values[i+1, j] + self.wind*old_values[i,j] + self.wind*old_values[i,j+1]
                            elif (i,j+1) == self.wall:
                                value_actions[k] = (1-2 * self.wind) * old_values[i+1, j] + self.wind*old_values[i,j-1] + self.wind*old_values[i,j]
                            else:
                                value_actions[k] = (1-2 * self.wind) * old_values[i+1, j] + self.wind*old_values[i,j-1] + self.wind *old_values[i,j+1]
                        elif k == 2:
                        #    if self.possible_states[i,j-1] == -1:
                        #        continue
                            if (i,j-1) == self.wall:
                                value_actions[k] = (1-2*self.wind)*old_values[i,j] + self.wind*old_values[i-1,j] + self.wind*old_values[i+1,j]
                            elif (i+1,j) == self.wall:
                                value_actions[k] = (1-2*self.wind)*old_values[i,j-1] + self.wind*old_values[i-1,j] + self.wind*old_values[i,j]
                            elif (i-1,j) == self.wall:
                                value_actions[k] = (1 - 2 * self.wind) * old_values[i, j - 1] + self.wind * old_values[
                                i, j] + self.wind * old_values[i+1, j]
                            else:
                                value_actions[k] = (1-2*self.wind)*old_values[i,j-1] + self.wind*old_values[i-1,j] + self.wind*old_values[i+1,j]
                        elif k == 3:
                        #    if self.possible_states[i-1,j] == -1:
                        #        continue
                            if (i-1,j) == self.wall:
                                value_actions[k] = (1-2*self.wind)*old_values[i,j] + self.wind*old_values[i,j-1] + self.wind *old_values[i,j+1]
                            elif (i,j+1) == self.wall:
                                value_actions[k] = (1-2*self.wind)*old_values[i-1,j] + self.wind * \
                                                                         old_values[
                                                                             i, j - 1] + self.wind * \
                                                                                         old_values[
                                                                                             i, j]
                            elif (i,j-1) == self.wall:
                                value_actions[k] = (1 - 2 * self.wind) * old_values[i - 1, j] + self.wind * \
                                                                         old_values[
                                                                             i, j] + self.wind * \
                                                                                         old_values[
                                                                                             i, j + 1]
                            else:
                                value_actions[k] = (1 - 2 * self.wind) * old_values[i-1, j] + self.wind * \
                                                                                                           old_values[
                                                                                                               i, j-1] + self.wind * \
                                                                                                                           old_values[
                                                                                                                               i, j+1]
                        self.action_value_function[i,j,k] = self.reward[i,j] + self.gamma*value_actions[k]
                    self.state_value_function[i,j] = max(self.action_value_function[i,j,:])
                    self.policy[i,j] = np.argmax(value_actions)

    def run_value_iteration(self):
        old_value_function = np.copy(self.state_value_function)
        delta = 100 # start with high value
        iteration = 0
        while delta > 0.00001 or iteration < 2:
            self.value_iteration()
            # print(self.state_value_function[1:1+self.size[0],1:1+self.size[1]])
            # print(self.action_value_function[1:1+self.size[0],1:1+self.size[1],:])
            value_del = np.abs(old_value_function - self.state_value_function)
            delta = np.max(np.max(value_del))
            # print("delta = ",delta)
            old_value_function = np.copy(self.state_value_function)
            iteration += 1


if __name__ == "__main__":
    wind = 0.1
    living_reward = -0.23
    grid = GridWorld(wind=wind,living_reward=living_reward,positive_scalar_factor=1,negative_scalar_factor=1)
    grid.run_value_iteration()
    reward = grid.reward[1:4, 1:5]
    value = grid.state_value_function[1:4, 1:5]
    policy = grid.policy[1:4, 1:5]
    print("State Value function = \n", value)
    print("Action Value function = \n", grid.action_value_function[1:4, 1:5, :])
    print("Policy = \n", policy)
    fig1 = plt.figure(1, figsize=(16, 6))
    ax1 = fig1.add_subplot(121)
    im1 = ax1.imshow(policy, cmap='rainbow')
    for (j, i), label in np.ndenumerate(policy):
        if label == -1:
            label_text = np.format_float_positional(reward[j, i], precision=3)
        elif label == 0:
            label_text = '\u27a1'
        elif label == 1:
            label_text = '\u2b07'
        elif label == 2:
            label_text = '\u2b05'
        else:
            label_text = '\u2b06'
        ax1.text(i, j, label_text, ha='center', va='center', fontsize=28)
    ax2 = fig1.add_subplot(122)
    im2 = ax2.imshow(value, cmap='rainbow')
    for (j, i), label in np.ndenumerate(value):
        label_text = np.format_float_positional(label, precision=3)
        ax2.text(i, j, label_text, ha='center', va='center', fontsize=28)
    # plt.title('Reward and state value function', x=0, y=1.1, fontdict = {'fontsize' : 15})
    fig1.subplots_adjust(right=0.8)
    cbar_ax = fig1.add_axes([0.85, 0.1, 0.02, 0.7])
    fig1.colorbar(im2, cax=cbar_ax)
    plt.show()
