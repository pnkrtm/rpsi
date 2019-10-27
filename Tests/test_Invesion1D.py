import sys
sys.path.append('../')

from fmodeling.ForwardProcessing1D import forward_with_trace_calcing
from Inversion.Strategies.SeismDiffInversion1D import inverse
from Inversion.Optimizators.Optimizations import LBFGSBOptimization
from Tests.test_ForwardProcessing1D import get_model_2layered
from objects.Data.WavePlaceholder import WaveDataPlaceholder
from objects.seismic.waves import OWT
from objects.Models.Models import SeismicModel1D
from objects.Attributes.RockPhysics.RockPhysicsAttribute import RockPhysicsAttribute
from objects.Models.Layer1D import Layer1D, LayerOPT

import time


def main():
    # Km, Gm, rho_m, Ks, Gs, rho_s, Kf, rho_f, phi, phi_s, h = get_model_1()
    layer_1_dict, layer_2_dict = get_model_2layered()
    h = 500

    layer_1 = Layer1D(h,
                      rp_attribute=RockPhysicsAttribute(layer_1_dict["components"], layer_1_dict["name"]),
                      seism_attribute=None,
                      opt=LayerOPT.RP)

    layer_2 = Layer1D(-1,
                      rp_attribute=RockPhysicsAttribute(layer_2_dict["components"], layer_2_dict["name"]),
                      seism_attribute=None,
                      opt=LayerOPT.RP)

    dx = 100
    nx = 20
    x_rec = [i * dx for i in range(1, nx)]
    wave_types = [OWT.PdPu]
    model = SeismicModel1D([layer_1, layer_2])

    print('Calculating DEM modeling...')
    observe, test_seismic = \
        forward_with_trace_calcing(model, x_rec,
                                   dt=3e-03, trace_len=1500, wavetypes=wave_types, display_stat=True,
            visualize_res=False)
    print('Forward calculated!')

    print('Calculating inversion...')

    inversion_start_time = time.time()

    forward_params = {
        "model": model,
        "x_rec": x_rec,
        "dt": 3e-03,
        "trace_len": 1500,
        "wavetypes": wave_types,
        "display_stat": False,
        "visualize_res": False
    }

    placeholders = {}
    for wt in wave_types:
        placeholders[wt] = WaveDataPlaceholder(
            wt,
            test_seismic[wt]["rays"],
            test_seismic[wt]["seismogram"]
        )

    optimizers = [
        LBFGSBOptimization(
            maxiter=15000,
            maxfun=15000,
            factr=10000,
            maxls=50,
            epsilon=0.0001
        )
    ]

    model.layers[0]['Km'] = 5
    model.layers[1]['Km'] = 20

    inversed_model = inverse(optimizers, error=0.01, placeholders=placeholders, forward_params=forward_params)

    print('Inversion calculated!')
    inversion_end_time = time.time()

    print('Inversion duration in minutes: {}'.format((inversion_end_time - inversion_start_time)/60))

    print(inversed_model)


if __name__ == '__main__':
    main()