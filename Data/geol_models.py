from collections import OrderedDict


def get_model_3layered_fluid_rp_dicts():
    layer_1_dict = OrderedDict({
                  "name": "xu-payne",
                  "components": {
                    "Km": {
                      "value": 7.3,
                      "optimize": False,
                      "min": 5,
                      "max": 10
                    },
					"Gm": {
                      "value": 2.71,
                      "optimize": False,
                      "min": 0.1,
                      "max": 5
                    },
					"rho_m": {
                      "value": 2.71,
                      "optimize": False,
                      "min": 2.5,
                      "max": 3
                    },
                    "Vm": {
                      "value": 1,
                      "optimize": False,
                      "min": 0.7,
                      "max": 1
                    },
                    "Ks": {
                      "value": 0,
                      "optimize": False,
                      "min": 7,
                      "max": 14
                    },
					"Gs": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.8
                    },
					"rho_s": {
                      "value": 0,
                      "optimize": False,
                      "min": 2.2,
                      "max": 2.6
                    },
                    "phi_s": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.1
                    },
                    "Kf": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.001,
                      "max": 3
                    },
                    "rho_f": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.01,
                      "max": 3
                    },
                    "phi": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.2
                    }
                  }
                })
    layer_2_dict = OrderedDict({
                  "name": "xu-payne",
                  "components": {
                    "Km": {
                      "value": 21.5,
                      "optimize": False,
                      "min": 20,
                      "max": 25
                    },
					"Gm": {
                      "value": 17.5,
                      "optimize": False,
                      "min": 15,
                      "max": 20
                    },
					"rho_m": {
                      "value": 2.8,
                      "optimize": False,
                      "min": 2.5,
                      "max": 3
                    },
                    "Vm": {
                      "value": 0.85,
                      "optimize": False,
                      "min": 0.7,
                      "max": 1
                    },
                    "Ks": {
                      "value": 9.2,
                      "optimize": False,
                      "min": 7,
                      "max": 14
                    },
					"Gs": {
                      "value": 0.4,
                      "optimize": False,
                      "min": 0.1,
                      "max": 0.8
                    },
					"rho_s": {
                      "value": 2.43,
                      "optimize": False,
                      "min": 2.2,
                      "max": 2.6
                    },
                    "phi_s": {
                      "value": 0.05,
                      "optimize": True,
                      "min": 0.001,
                      "max": 0.1
                    },
                    "Kf": {
                      "value": 1.8,
                      "optimize": True,
                      "min": 1,
                      "max": 2
                    },
                    "rho_f": {
                      "value": 0.95,
                      "optimize": True,
                      "min": 0.01,
                      "max": 1
                    },
                    "phi": {
                      "value": 0.1,
                      "optimize": True,
                      "min": 0.001,
                      "max": 0.2
                    }
                  }
                })

    layer_3_dict = OrderedDict({
        "name": "xu-payne",
                "components":
        {"Km": {
                      "value": 22,
                      "optimize": True,
                      "min": 20,
                      "max": 25
                    },
					"Gm": {
                      "value": 10.7,
                      "optimize": False,
                      "min": 5,
                      "max": 15
                    },
					"rho_m": {
                      "value": 2.85,
                      "optimize": False,
                      "min": 2.5,
                      "max": 3
                    },
                    "Vm": {
                      "value": 1,
                      "optimize": False,
                      "min": 0.7,
                      "max": 1
                    },
                    "Ks": {
                      "value": 9.2,
                      "optimize": False,
                      "min": 7,
                      "max": 14
                    },
					"Gs": {
                      "value": 0.4,
                      "optimize": False,
                      "min": 0.1,
                      "max": 0.8
                    },
					"rho_s": {
                      "value": 2.43,
                      "optimize": False,
                      "min": 2.2,
                      "max": 2.6
                    },
                    "phi_s": {
                      "value": 0.0,
                      "optimize": False,
                      "min": 0.00,
                      "max": 0.1
                    },
                    "Kf": {
                      "value": 1.8,
                      "optimize": False,
                      "min": 1,
                      "max": 2
                    },
                    "rho_f": {
                      "value": 0.95,
                      "optimize": False,
                      "min": 0.01,
                      "max": 1
                    },
                    "phi": {
                      "value": 0.1,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.2
                    }}})

    return layer_1_dict, layer_2_dict, layer_3_dict