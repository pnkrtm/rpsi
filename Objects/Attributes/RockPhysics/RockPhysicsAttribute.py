from collections import OrderedDict
from Objects.Attributes.AbstractAttribute import AbstarctAttribute


class RockPhysicsAttribute(AbstarctAttribute):
    def __init__(self, vals_dict: OrderedDict, model_name):
        self.model_name = model_name
        self.vals_dict = vals_dict.copy()

    def __getitem__(self, item):
        return self.vals_dict[item]["value"]

    def __setitem__(self, key, value):
        self.vals_dict[key]["value"] = value

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
