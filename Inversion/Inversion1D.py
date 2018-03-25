import numpy as np
import time

from ForwardModeling.ForwardProcessing1D import forward
from Inversion.Optimizations import DifferentialEvolution
from Inversion.Tools import OptimizeHelper
from Exceptions.exceptions import ErrorAchievedException


def get_time_differences(rays_observed, rays_synthetic, weights=None):
    times_observed = np.array([r.time for r in rays_observed])
    times_synthetic = np.array([r.time for r in rays_synthetic])

    if weights is None:
        weights = [1]*len(times_observed)

    diffs = np.abs(times_observed - times_synthetic) / times_observed
    err = np.average(diffs, weights=weights)

    return err


def get_reflection_differences(reflection_observed, reflection_synthetic, weights=None):
    ampl_obs_real = np.array([ampl.real for ampl in reflection_observed.amplitudes])
    ampl_obs_imag = np.array([ampl.imag for ampl in reflection_observed.amplitudes])

    ampl_syn_real = np.array([ampl.real for ampl in reflection_synthetic.amplitudes])
    ampl_syn_imag = np.array([ampl.imag for ampl in reflection_synthetic.amplitudes])

    # if weights is None:
    #     weights = [1]*len(ampl_obs_real)

    diffs_1 = [abs((r - s) / r) for r, s in zip(ampl_obs_real, ampl_syn_real) if not np.isnan(r) and not np.isnan(s)]

    diffs_1 = [d for d in diffs_1 if not np.isnan(d)]

    # diffs_2 = [abs(r - s) / r for r, s in zip(ampl_obs_imag, ampl_syn_imag) if not np.isnan(r) and not np.isnan(s)]
    #
    # diffs_2 = [d for d in diffs_2 if not np.isnan(d)]

    # return (np.average(diffs_1) + np.average(diffs_2)) / 2

    return  np.average(diffs_1)


def func_to_optimize_universal(model_opt, params_all, params_to_optimize,
                rays_observed_p, rays_observed_s,
                reflection_observed_p, reflection_observed_s,
                use_rays_p, use_rays_s,
                use_reflection_p, use_reflection_s, helper):

    params_all_ = {}

    for key in list(params_all.keys()):
        if type(params_all[key]) == type([]):
            params_all_[key] = params_all[key].copy()

        else:
            params_all_[key] = params_all[key]

    for m, p in zip(model_opt, params_to_optimize):
        params_all_[list(p.keys())[0]][list(p.values())[0]] = m

    observe, model, rays_p, rays_s, reflection_p, reflection_s = forward(**params_all_)

    depths = model.get_depths()

    error_start_time = time.time()

    errs = []

    for d in depths[1:]:
        if use_rays_p:
            rays_p_ = [r for r in rays_p if r.get_reflection_z() == d]
            rays_p_o = [r for r in rays_observed_p if r.get_reflection_z() == d]

            errs.append(get_time_differences(rays_p_o, rays_p_))

        if use_rays_s:
            rays_s_ = [r for r in rays_s if r.get_reflection_z() == d]
            rays_s_o = [r for r in rays_observed_s if r.get_reflection_z() == d]

            errs.append(get_time_differences(rays_s_o, rays_s_))

    if use_reflection_p:
        for rp, rop in zip(reflection_p, reflection_observed_p):
            errs.append(get_reflection_differences(rop, rp))

    if use_reflection_s:
        for rs, ros in zip(reflection_s, reflection_observed_s):
            errs.append(get_reflection_differences(ros, rs))

    error = np.average(errs)

    print(np.average(errs))

    helper.add_error(error)

    if helper.need_to_stop():
        raise ErrorAchievedException(model_opt)

    return error


def func_to_optimize(model_opt, nlayers, Km, Gm, Ks, Gs, h, x_rec,
                     rays_observed_p, rays_observed_s,
                     reflection_observed_p, reflection_observed_s,
                     use_rays_p, use_rays_s, use_reflection_p=False, use_reflection_s=False):

    forward_start_time = time.time()

    observe, model, rays_p, rays_s, reflection_p, reflection_s = forward(nlayers, Km, Gm, Ks, Gs,
                                             model_opt[0:nlayers], # Kf
                                             model_opt[nlayers:2*nlayers], # phi
                                             model_opt[2*nlayers:3*nlayers], # phi_s
                                             model_opt[3*nlayers:4*nlayers], # rho_s
                                             model_opt[4*nlayers:5*nlayers], # rho_f
                                             model_opt[5*nlayers:6*nlayers], # rho_m
                                             h, x_rec,
                                             display_stat=False, visualize_res=False,
                                             calc_rays_p=use_rays_p, calc_rays_s=use_rays_s,
                                             calc_reflection_p=use_reflection_p, calc_reflection_s=use_reflection_s
                                             )

    depths = model.get_depths()

    error_start_time = time.time()

    errs = []

    for d in depths[1:]:
        if use_rays_p:
            rays_p_ = [r for r in rays_p if r.get_reflection_z() == d]
            rays_p_o = [r for r in rays_observed_p if r.get_reflection_z() == d]

            errs.append(get_time_differences(rays_p_o, rays_p_))

        if use_rays_s:
            rays_s_ = [r for r in rays_s if r.get_reflection_z() == d]
            rays_s_o = [r for r in rays_observed_s if r.get_reflection_z() == d]

            errs.append(get_time_differences(rays_s_o, rays_s_))

    if use_reflection_p:
        for rp, rop in zip(reflection_p, reflection_observed_p):
            errs.append(get_reflection_differences(rop, rp))

    if use_reflection_s:
        for rs, ros in zip(reflection_s, reflection_observed_s):
            errs.append(get_reflection_differences(ros, rs))


    error_end_time = time.time()

    # print('forward time = {}'.format(error_start_time - forward_start_time))
    # print('error time = {}'.format(error_end_time - error_start_time))
    # print('all_time = {}'.format(error_end_time - forward_start_time))

    print(np.average(errs))

    return np.average(errs)


def inverse(nlayers, Km, Gm, Ks, Gs, h, x_rec,
            rays_observed_p, rays_observed_s,
            reflection_observed_p, reflection_observed_s,
            data_start=None, opt_type='de',
            use_rays_p=True, use_rays_s=True, use_reflection_p=False, use_reflection_s=False):


    if opt_type == 'de':
        optimizer = DifferentialEvolution(popsize=2, maxiter=10, atol=1000, init='random', polish=True)

        args = (nlayers, Km, Gm, Ks, Gs, h, x_rec,
                rays_observed_p, rays_observed_s,
                reflection_observed_p, reflection_observed_s,
                use_rays_p, use_rays_s,
                use_reflection_p, use_reflection_s)

        result_model = optimizer.optimize(func_to_optimize, data_start, args)

    return result_model


def inverse_universal(optimizers, error, params_all, params_to_optimize, data_start,
                    rays_observed_p, rays_observed_s,
                    reflection_observed_p, reflection_observed_s,
                    opt_type='de',
                    use_rays_p=True, use_rays_s=True, use_reflection_p=False, use_reflection_s=False):

    if opt_type == 'de':
        helper = OptimizeHelper(nerrors=len(data_start), error_to_stop=error)

        args = (params_all, params_to_optimize,
                rays_observed_p, rays_observed_s,
                reflection_observed_p, reflection_observed_s,
                use_rays_p, use_rays_s,
                use_reflection_p, use_reflection_s, helper)

        try:
            result_model = optimizers[0].optimize(func_to_optimize_universal, data_start, args)

        except ErrorAchievedException as e:
            print('======  LBFGS optimization started! =======')
            helper.in_use = False
            start_model = e.model

            result_model = optimizers[1].optimize(func_to_optimize_universal, start_model, data_start, args)

    return result_model


