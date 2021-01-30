from collections import OrderedDict
from objects.Attributes.AbstractAttribute import AbstarctAttribute
from exceptions.bad_calcs import ParamOutOfBoundException


class RockPhysicsAttribute(AbstarctAttribute):
    def __init__(self, vals_dict: OrderedDict, model_name):
        self.model_name = model_name
        self.vals_dict = vals_dict.copy()

    def __getitem__(self, item):
        return self.vals_dict[item]["value"]

    def __setitem__(self, key, value):
        if self.vals_dict[key]["min"] <= value <= self.vals_dict[key]["max"]:
            self.vals_dict[key]["value"] = value

        else:
            raise ParamOutOfBoundException()

    def get_params_to_optimize(self):
        return {key: value["value"] for key, value in self.vals_dict.items() if value["optimize"]}

    def get_min_optimize_bound(self):
        return {key: value["min"] for key, value in self.vals_dict.items() if value["optimize"]}

    def get_max_optimize_bound(self):
        return {key: value["max"] for key, value in self.vals_dict.items() if value["optimize"]}

    def get_input_params(self):
        return {key: value["value"] for key, value in self.vals_dict.items()}

    def set_optimized_params(self, values):
        i = 0
        for key in self.vals_dict.keys():
            if self.vals_dict[key]["optimize"]:
                self.vals_dict[key]["value"] = values[i]

                i += 1
