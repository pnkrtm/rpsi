import numpy as np
import multiprocessing as mp
import gc
from numpy import linalg
import time

from ForwardModeling.ForwardProcessing1D import forward, forward_with_trace_calcing
from Inversion.Optimizations import DifferentialEvolution, DifferentialEvolution_parallel
from Inversion.Tools import OptimizeHelper
from Exceptions.exceptions import ErrorAchievedException


def get_time_differences(rays_observed, rays_synthetic, weights=None, metrics='rmse'):
    times_observed = np.array([r.time for r in rays_observed])
    times_synthetic = np.array([r.time for r in rays_synthetic])

    if weights is None:
        weights = 1

    if weights and not isinstance(weights, list) and not isinstance(weights, np.ndarray):
        weights = [weights]*len(times_observed)

    if metrics == 'mae':
        diffs = np.abs(times_observed - times_synthetic) / times_observed
        err = np.average(diffs, weights=weights)

    elif metrics == 'rmse':
        diffs = (times_observed - times_synthetic)**2 / times_observed**2
        err = np.average(diffs)
        err = np.sqrt(err)

    return err


def get_reflection_differences(reflection_observed, reflection_synthetic, weights=None, metrics='rmse'):
    ampl_obs_real = np.array([ampl.real for ampl in reflection_observed.amplitudes])
    ampl_obs_imag = np.array([ampl.imag for ampl in reflection_observed.amplitudes])

    ampl_syn_real = np.array([ampl.real for ampl in reflection_synthetic.amplitudes])
    ampl_syn_imag = np.array([ampl.imag for ampl in reflection_synthetic.amplitudes])

    if weights is None:
        weights = 1

    if weights and not isinstance(weights, list) and not isinstance(weights, np.ndarray):
        weights = [weights]*len(ampl_obs_real)


    # if weights is None:
    #     weights = [1]*len(ampl_obs_real)

    if metrics == 'mae':
        diffs_1 = [abs((r - s) / r) for r, s in zip(ampl_obs_real, ampl_syn_real) if not np.isnan(r) and not np.isnan(s)]
        diffs_1 = [d for d in diffs_1 if not np.isnan(d)]
        err = np.average(diffs_1)

    elif metrics == 'rmse':
        diffs_1 = [(r - s)**2 / r**2 for r, s in zip(ampl_obs_real, ampl_syn_real) if not np.isnan(r) and not np.isnan(s)]
        diffs_1 = np.average(diffs_1)
        err = np.sqrt(diffs_1)



    # diffs_2 = [abs(r - s) / r for r, s in zip(ampl_obs_imag, ampl_syn_imag) if not np.isnan(r) and not np.isnan(s)]
    #
    # diffs_2 = [d for d in diffs_2 if not np.isnan(d)]

    # return (np.average(diffs_1) + np.average(diffs_2)) / 2

    return  err


def func_to_optimize_universal_mp_helper(args):

    return func_to_optimize_universal(**args)


def func_to_optimize_universal(model_opt, params_all, params_to_optimize, params_bounds,
                rays_observed_p, rays_observed_s,
                reflection_observed_p, reflection_observed_s,
                use_rays_p, use_rays_s,
                use_reflection_p, use_reflection_s, helper, parallel, pool, layer_weights=None):

    if parallel:
        # определние величины популяции
        njobs = int(len(model_opt) / len(model_opt[0]))

        input_args = []

        for i in range(njobs):
            model_opt_ = model_opt[i, :]
            # model_opt_ = model_opt_ * (params_bounds[:,1] - params_bounds[:,0]) + params_bounds[:,0]

            input_args.append(
                {
                    'model_opt': model_opt_,
                    'params_all': params_all,
                    'params_to_optimize': params_to_optimize,
                    'params_bounds': params_bounds,
                    'rays_observed_p': rays_observed_p,
                    'rays_observed_s': rays_observed_s,
                    'reflection_observed_p': reflection_observed_p,
                    'reflection_observed_s': reflection_observed_s,
                    'use_rays_p': use_rays_p,
                    'use_rays_s': use_rays_s,
                    'use_reflection_p': use_reflection_p,
                    'use_reflection_s': use_reflection_s,
                    'helper': helper,
                    'parallel': False,
                    'pool': None
                }
            )



        # with mp.Pool(processes=nproc) as pool:
        result_errors = pool.map(func_to_optimize_universal_mp_helper, input_args)

        gc.collect()

        return result_errors

    # model_opt_2 = np.array(model_opt) * (params_bounds[:,1] - params_bounds[:,0]) + params_bounds[:,0]
    model_opt_2 = model_opt

    params_all_ = {}

    for key in list(params_all.keys()):
        if type(params_all[key]) == type([]):
            params_all_[key] = params_all[key].copy()

        else:
            params_all_[key] = params_all[key]

    for m, p in zip(model_opt_2, params_to_optimize):
        params_all_[list(p.keys())[0]][list(p.values())[0]] = m

    observe, model, rays_p, rays_s, reflection_p, reflection_s = forward(**params_all_)

    depths = model.get_depths()

    error_start_time = time.time()

    errs = []

    # weights for depths params
    xp = [0, len(depths) - 1]
    fp = [0.6, 1]

    errs_times_p = []
    weights_times_p = []

    errs_times_s = []
    weights_times_s = []

    errs_refl_p = []
    weights_refl_p = []

    errs_refl_s = []
    weights_refl_s = []

    i = 0
    for d in depths[1:]:
        if use_rays_p:
            rays_p_ = [r for r in rays_p if r.get_reflection_z() == d]
            rays_p_o = [r for r in rays_observed_p if r.get_reflection_z() == d]

            errs_times_p.append(get_time_differences(rays_p_o, rays_p_))
            weights_times_p.append(np.interp(i, xp, fp))

        if use_rays_s:
            rays_s_ = [r for r in rays_s if r.get_reflection_z() == d]
            rays_s_o = [r for r in rays_observed_s if r.get_reflection_z() == d]

            errs_times_s.append(get_time_differences(rays_s_o, rays_s_))
            weights_times_s.append(np.interp(i, xp, fp))

        i += 1

    i = 0
    if use_reflection_p:
        for rp, rop in zip(reflection_p, reflection_observed_p):
            errs_refl_p.append(get_reflection_differences(rop, rp))
            weights_refl_p.append(np.interp(i, xp, fp))
            i += 1

    i = 0
    if use_reflection_s:
        for rs, ros in zip(reflection_s, reflection_observed_s):
            errs_refl_s.append(get_reflection_differences(ros, rs))
            weights_refl_s.append(np.interp(i, xp, fp))
            i += 1

    if layer_weights:
        # weights_times_p = layer_weights
        # weights_times_s = layer_weights
        weights_refl_p = layer_weights
        weights_refl_s = layer_weights

    errs_times_p = np.average(errs_times_p, weights=weights_times_p)
    errs_times_s = np.average(errs_times_s, weights=weights_times_s)

    errs_refl_p = np.average(errs_refl_p, weights=weights_refl_p)
    errs_refl_s = np.average(errs_refl_s, weights=weights_refl_s)

    errs = [errs_times_p, errs_times_s, errs_refl_p, errs_refl_s]
    weights = [1, 1, 0.65, 0.65]

    error = np.average(errs, weights=weights)

    print(np.average(errs))

    helper.add_error(error)

    if helper.need_to_stop():
        raise ErrorAchievedException(model_opt)

    return error


def func_to_optimize_seismogram_universal(model_opt, params_all, params_to_optimize,
                seismogram_observed_p, seismogram_observed_s,
                use_p_waves, use_s_waves, helper):

    params_all_ = {}

    for key in list(params_all.keys()):
        if type(params_all[key]) == type([]):
            params_all_[key] = params_all[key].copy()

        else:
            params_all_[key] = params_all[key]

    for m, p in zip(model_opt, params_to_optimize):
        params_all_[list(p.keys())[0]][list(p.values())[0]] = m

    seismogram_p, seismogram_s = forward_with_trace_calcing(**params_all_)

    errs = []

    if use_p_waves:
        p1 = seismogram_p.get_values_matrix()
        p2 = seismogram_observed_p.get_values_matrix()
        diff_p = p1 - p2
        np.nan_to_num(diff_p, False)
        # diff_p = (seismogram_p.get_values_matrix() - seismogram_observed_p.get_values_matrix())
        errs.append(linalg.norm(diff_p))

    if use_s_waves:
        s1 = seismogram_s.get_values_matrix()
        s2 = seismogram_observed_s.get_values_matrix()
        diff_s = s1 - s2
        np.nan_to_num(diff_s, False)
        # diff_s = np.abs(seismogram_s.get_values_matrix() - seismogram_observed_s.get_values_matrix())
        errs.append(linalg.norm(diff_s))

    error_start_time = time.time()

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


def inverse_universal(optimizers, error, params_all, params_to_optimize, params_bounds,
                    rays_observed_p, rays_observed_s,
                    reflection_observed_p, reflection_observed_s,
                    opt_type='de',
                    use_rays_p=True, use_rays_s=True, use_reflection_p=False, use_reflection_s=False,
                      layer_weights=None):

    if opt_type == 'de':
        # data_start = [[0.000001, 1] for i in range(len(params_bounds))]
        data_start = params_bounds
        helper = OptimizeHelper(nerrors=len(data_start), error_to_stop=error)

        if type(optimizers[0]) == DifferentialEvolution_parallel:
            parallel = optimizers[0].parallel

            if parallel:

                helper.in_use = False

                ncpu = mp.cpu_count()
                nproc = int(ncpu * 2)

                pool = mp.Pool(nproc)

            else:
                parallel = False
                pool = None
        else:
            parallel = False
            pool = None

        args = [params_all, params_to_optimize, params_bounds,
                rays_observed_p, rays_observed_s,
                reflection_observed_p, reflection_observed_s,
                use_rays_p, use_rays_s,
                use_reflection_p, use_reflection_s, helper, parallel, pool, layer_weights]

        try:
            result_model = optimizers[0].optimize(func_to_optimize_universal, data_start, args)
            start_model = result_model

        except ErrorAchievedException as e:
            start_model = e.model

        finally:
            if pool:
                pool.close()
                pool.join()

        print('======  LBFGS optimization started! =======')
        helper.in_use = False

        args[-3] = False

        result_model = optimizers[1].optimize(func_to_optimize_universal, start_model, params_bounds, tuple(args))

        # result_model = np.array(result_model) * (params_bounds[:, 1] - params_bounds[:, 0]) + params_bounds[:, 0]


    return result_model


def inverse_universal_shots(optimizers, error, params_all, params_to_optimize, data_start,
                            seismogram_observed_p, seismogram_observed_s,
                            opt_type='de',
                            use_p_waves=True, use_s_waves=True):
    if opt_type == 'de':
        helper = OptimizeHelper(nerrors=len(data_start), error_to_stop=error)

        args = (params_all, params_to_optimize,
                seismogram_observed_p, seismogram_observed_s,
                use_p_waves, use_s_waves, helper)

        try:
            result_model = optimizers[0].optimize(func_to_optimize_seismogram_universal, data_start, args)
            start_model = result_model

        except ErrorAchievedException as e:
            start_model = e.model

        print('======  LBFGS optimization started! =======')
        helper.in_use = False

        result_model = optimizers[1].optimize(func_to_optimize_seismogram_universal, start_model, data_start, args)

    return result_model

