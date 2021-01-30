import sys
sys.path.append('../')

from fmodelling.forward_proc_1D import forward_with_trace_calcing
from inversion.Strategies.SeismDiffInversion1D import inverse
from inversion.optimizators.optimizations import LBFGSBOptimization, AxOptimizer, DifferentialEvolution
from tests.test_ForwardProcessing1D import get_model_2layered
from objects.data.WavePlaceholder import WaveDataPlaceholder
from objects.seismic.waves import OWT
from objects.models.Models import SeismicModel1D
from objects.attributes.rock_physics.RockPhysicsAttribute import RockPhysicsAttribute
from objects.models.Layer1D import Layer1D, LayerOPT
from inversion.Strategies.SeismDiffInversion1D import func_to_optimize

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
    nx = 2
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

    optimizers_de = [
        DifferentialEvolution(
            popsize=5,
            maxiter=50000,
            init="latinhypercube",
            strategy="best1bin",
            disp=True,
            polish=False,
            tol=0.00001,
            mutation=1.5,
            recombination=0.6,
            workers=8
        )
    ]

    # optimizers = [
    #     AxOptimizer(num_evals=20)
    # ]

    model.layers[0]['Km'] = 5
    model.layers[1]['Km'] = 20

    # true values: 7.3 and 21.5

    # from inversion.Strategies.SeismDiffInversion1D import func_to_optimize
    # assert func_to_optimize(forward_params['model'].get_optimization_option('val', vectorize=True), placeholders,
    #                  forward_params, helper=None, show_tol=False) < 0.01

    inversed_model, stats = inverse(optimizers_de, error=0.0001, placeholders=placeholders, forward_params=forward_params)

    print('inversion calculated!')
    inversion_end_time = time.time()

    print('inversion duration: {} seconds'.format((inversion_end_time - inversion_start_time)))

    print(inversed_model)

    func_to_opt_start_time = time.time()
    func_to_optimize(forward_params['model'].get_optimization_option('val', vectorize=True), placeholders,
                     forward_params, helper=None, show_tol=False)
    print(f'Func to optimize evaluation time: {time.time() - func_to_opt_start_time} seconds')


if __name__ == '__main__':
    main()