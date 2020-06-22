import numpy as np

DEFAULT_ACTION = (0, 0, 0)

class RandomAgent():
    def __init__(self, n_hospitals, n_severity_levels, n_actions):
        self.n_hospitals = n_hospitals
        self.n_severity_levels = n_severity_levels
        self.n_actions = n_actions

    def __call__(self, observation):
        severities = np.random.randint(self.n_severity_levels+1, size=self.n_actions)
        start_hospitals = np.random.randint(self.n_hospitals+1, size=self.n_actions)
        end_hospitals = np.random.randint(self.n_hospitals+1, size=self.n_actions)
        
        to_return = [(severities[i], start_hospitals[i], end_hospitals[i]) for i in range(self.n_actions)]

        return to_return