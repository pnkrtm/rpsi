import time
import numpy as np

from fmodeling.ForwardProcessing1D import forward_with_trace_calcing
from Inversion.Strategies.SeismDiffInversion1D import inverse
from Inversion.Optimizators.Optimizations import LBFGSBOptimization, DifferentialEvolution
from Tests.test_ForwardProcessing1D import get_model_2layered, get_model_3layered
from objects.Data.WavePlaceholder import WaveDataPlaceholder
from objects.seismic.waves import OWT
from objects.Models.Models import SeismicModel1D
from objects.Attributes.RockPhysics.RockPhysicsAttribute import RockPhysicsAttribute
from objects.Models.Layer1D import Layer1D, LayerOPT
from Inversion.Strategies.SeismDiffInversion1D import func_to_optimize


def process_3_layer():
    layer_1_dict, layer_2_dict, layer_3_dict = get_model_3layered()
    h = 500

    layer_1 = Layer1D(h,
                      rp_attribute=RockPhysicsAttribute(layer_1_dict["components"], layer_1_dict["name"]),
                      seism_attribute=None,
                      opt=LayerOPT.RP)

    layer_2 = Layer1D(h,
                      rp_attribute=RockPhysicsAttribute(layer_2_dict["components"], layer_2_dict["name"]),
                      seism_attribute=None,
                      opt=LayerOPT.RP)

    layer_3 = Layer1D(-1,
                      rp_attribute=RockPhysicsAttribute(layer_3_dict["components"], layer_2_dict["name"]),
                      seism_attribute=None,
                      opt=LayerOPT.RP)

    model = SeismicModel1D([layer_1, layer_2, layer_3])
    # model.layers[0]["Km"] =
    model.layers[1].opt = LayerOPT.NO
    model.layers[2].opt = LayerOPT.NO


    dx = 300
    nx = 5
    x_rec = [i*dx for i in range(1, nx+1)]
    wave_types = [OWT.PdPu]

    observe, test_seismic = \
        forward_with_trace_calcing(model, x_rec,
                                   dt=3e-03, trace_len=1500, wavetypes=wave_types, display_stat=True,
                                   visualize_res=False)

    placeholders = {}
    for wt in wave_types:
        placeholders[wt] = WaveDataPlaceholder(
            wt,
            test_seismic[wt]["rays"],
            test_seismic[wt]["seismogram"]
        )

    forward_params = {
        "model": model,
        "x_rec": x_rec,
        "dt": 3e-03,
        "trace_len": 1500,
        "wavetypes": wave_types,
        "display_stat": False,
        "visualize_res": False
    }
    experiment_val_Km1 = 5.89

    optimizers = [
        LBFGSBOptimization(
            maxiter=15000,
            maxfun=15000,
            factr=10000,
            maxls=50,
            epsilon=0.000001
        )
    ]

    inversed_model = inverse(optimizers, error=0.01, placeholders=placeholders, forward_params=forward_params, scale="minmax")
    err = func_to_optimize([experiment_val_Km1], placeholders, forward_params, helper=None, show_tol=False)
    print(1)


if __name__ == '__main__':
    process_3_layer()