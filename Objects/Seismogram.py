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

    # ВСЕГДА ПРОВЕРЯТЬ, ЕСЛИ В ТРАССЫ ДОБАВЛЯЮТСЯ НОВЫЕ ПОЛЯ!!!
    def _check_equality(self, other):
        for trace_self, trace_other in zip(self.traces, other.traces):
            attributes = list(trace_self.__dict__.keys())
            attributes.remove('values')

            for attr in attributes:
                if trace_self.__getattribute__(attr) != trace_other.__getattribute__(attr):
                    return False

            if trace_self.tracelength != trace_other.tracelength:
                return False

        return True

    def _trace_values_math_operation(self, other, operation):
        """

        :param other:
        :param operation: lambda function
        :return:
        """
        if self._check_equality(other):
            traces_result = []
            for trace_self, trace_other in zip(self.traces, other.traces):
                trace_params = trace_self.__dict__
                trace_params['values'] = operation(trace_self.values, trace_other.values)

                traces_result.append(Trace(**trace_params))

            return Seismogram(traces_result)

        else:
            raise ValueError("Seismograms are not equal!")

    def __add__(self, other):
        return self._trace_values_math_operation(other, lambda x1, x2: x1 + x2)

    def __sub__(self, other):
        return self._trace_values_math_operation(other, lambda x1, x2: x1 - x2)

    def __mul__(self, other):
        return self._trace_values_math_operation(other, lambda x1, x2: x1 * x2)

    def __truediv__(self, other):
        return self._trace_values_math_operation(other, lambda x1, x2: x1 / x2)

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
        return np.array([np.array(t.values) for t in self.traces])

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
