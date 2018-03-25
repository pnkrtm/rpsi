import numpy as np


class OptimizeHelper:
    def __init__(self, nerrors=20, in_use=True, error_to_stop=0.01, std_to_stop=0.001):
        self.nerrors = nerrors
        self.errors_list = []
        self.iter = 0
        self.in_use = in_use
        self.error_to_stop = error_to_stop
        self.std_to_stop = std_to_stop

    def increment(self):
        self.iter += 1

    def add_error(self, error):
        self.errors_list.append(error)
        self.increment()

        if len(self.errors_list) > self.nerrors:
            self.errors_list = self.errors_list[-self.nerrors::]

    def get_mean(self):
        return np.mean(self.errors_list)

    def get_std(self):
        return np.std(self.errors_list)

    def need_to_stop(self):
        if not len(self.errors_list) < self.nerrors and self.in_use:
            mean = self.get_mean()
            std = self.get_std()

            if mean < self.error_to_stop and std < self.std_to_stop:
                return True

        return False
