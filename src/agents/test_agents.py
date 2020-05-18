
import numpy as np


class RandomAgent():
    def __init__(self, n_hospitals, n_severity_levels):
        self.n_hospitals = n_hospitals
        self.n_severity_levels = n_severity_levels

    def __call__(self, observation):
        start_hospitals = np.random.randint(self.n_hospitals+1, size=self.n_severity_levels+1)
        end_hospitals = np.random.randint(self.n_hospitals+1, size=self.n_severity_levels+1)

        return start_hospitals, end_hospitals


