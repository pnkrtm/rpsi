import numpy as np


class Trace:
    def __init__(self, values=[], dt=-1, offset=-1, start_time=0):
        self.values = values
        self.dt = dt
        self.offset = offset
        self.start_time = start_time

    @property
    def tracelength(self):
        return len(self.values)

    @property
    def times(self):
        return [self.start_time + i*self.dt for i in range(self.tracelength)]

    @property
    def min_time(self):
        return self.start_time

    @property
    def max_time(self):
        return self.start_time + self.tracelength * self.dt


class Seismogram:
    def __init__(self, traces=None):
        self.traces = traces or []

    def add_trace(self, trace):
        if self.traces:
            self.traces.append(trace)

        else:
            self.traces = [trace]

    def get_time_range(self):
        return self.traces[0].times

    def get_offsets(self):
        return [t.offset for t in self.traces]

    def get_values_matrix(self):
        return np.array([t.values for t in self.traces])

    @property
    def ntraces(self):
        return len(self.traces)

    @classmethod
    def from_segy(cls, segy):
        traces=[]

        for trace in segy.traces:
            traces.append(
                Trace()
            )
