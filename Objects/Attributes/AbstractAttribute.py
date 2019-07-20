import abc


class AbstarctAttribute:
    @abc.abstractmethod
    def get_params_to_optimize(self):
        ...

    @abc.abstractmethod
    def get_min_optimize_bound(self):
        ...

    @abc.abstractmethod
    def get_max_optimize_bound(self):
        ...

    @abc.abstractmethod
    def set_optimized_params(self, values):
        ...
