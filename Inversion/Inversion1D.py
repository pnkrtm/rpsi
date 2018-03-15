import numpy as np

from ForwardModeling.ForwardProcessing1D import forward
from Inversion.Optimizations import DifferentialEvolution


def get_time_differences(rays_observed, rays_synthetic, weights=None):
    times_observed = np.array([r.time for r in rays_observed])
    times_synthetic = np.array([r.time for r in rays_synthetic])

    if weights is None:
        weights = [1]*len(times_observed)

    diffs = np.abs(times_observed - times_synthetic) / times_observed
    err = np.average(diffs, weights=weights)

    return err

def func_to_optimize(model_opt, nlayers, Km, Gm, Ks, Gs, Kf, h, x_rec, rays_observed_p, rays_observed_s):
    observe, model, rays_p, rays_s = forward(nlayers, Km, Gm, Ks, Gs, Kf, model_opt[0], model_opt[1], model_opt[2],
                                             model_opt[3], model_opt[4], h, x_rec, False
                                             )

    depths = model.get_depths()

    errs = []

    for d in depths[1:]:
        rays_p_ = [r for r in rays_p if r.get_reflection_depth() == d]
        rays_p_o = [r for r in rays_observed_p if r.get_reflection_depth() == d]

        errs.append(get_time_differences(rays_p_o, rays_p_))

        rays_s_ = [r for r in rays_s if r.get_reflection_depth() == d]
        rays_s_o = [r for r in rays_observed_s if r.get_reflection_depth() == d]

        errs.append(get_time_differences(rays_s_o, rays_s_))

    print(np.average(errs))

    return np.average(errs)


def inverse(nlayers, Km, Gm, Ks, Gs, Kf, h, x_rec, rays_observed_p, rays_observed_s, data_start=None, opt_type='de'):
    if opt_type == 'de':
        optimizer = DifferentialEvolution()

        args = (nlayers, Km, Gm, Ks, Gs, Kf, h, x_rec, rays_observed_p, rays_observed_s)

        result_model = optimizer.optimize(func_to_optimize, data_start, args)

        return result_model


