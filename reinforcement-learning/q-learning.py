import random

import numpy as np
import math


class QLearning:
    def __init__(self, states: int, actions: int, model=None):
        # Init matrix
        if model is None:
            self.q_matrix = np.zeros((states, actions))
        else:
            self.q_matrix = np.loadtxt(model, delimiter=',')

        # Hyperparameters
        self.alpha = 0.1  # Learning rate (0 - 1)
        self.gamma = 0.6  # Discount factor (0 - 1)
        self.eps = 0.1    # Exploration vs Exploitation factor

    def action(self, state: int):
        # Action to be performed to maximize Q and ignoring nan values
        try:
            best = np.nanargmax(self.q_matrix[state, :])  # Best according to q-value
            if random.uniform(0, 1) < self.eps:  # Explore
                possible = [i for i, el in enumerate(self.q_matrix[state, :]) if not math.isnan(el) and i != best]
                if possible:
                    rnd = random.randint(0, len(possible) - 1)
                    return possible[rnd]
                else:
                    return best
            else:  # Exploit
                return best
        except ValueError:
            return None

    def update(self, state: int, next_states: list, action: int, reward: int):
        # Next states worst-case scenario
        if next_states:
            try:
                next_states_mat = self.q_matrix[next_states, :]
                worst_next_q = np.nanmin(next_states_mat)
            except RuntimeWarning:
                worst_next_q = 0
        else:
            worst_next_q = 0

        # Update Q value
        self.q_matrix[state, action] = \
            (1 - self.alpha) * self.q_matrix[state, action] + \
            self.alpha * (reward + self.gamma*worst_next_q)

        if math.isnan(self.q_matrix[state, action]):
            print("check")

    def save_model(self, name):
        np.savetxt(name+".csv", self.q_matrix, delimiter=",")
