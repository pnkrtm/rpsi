import numpy as np
import random
from scipy.optimize import fmin_l_bfgs_b, differential_evolution, dual_annealing, newton, fmin_cg, fmin_bfgs, fmin, \
    minimize


from Inversion.Optimizators._differentialevolution import differential_evolution as differential_evolution_parallel


class BaseOptimization:
    def __init__(self, helper=None):
        self.helper = helper

    def optimize(self, func, x0, bounds, args=(), **kwargs):
        raise NotImplementedError()


class LBFGSBOptimization(BaseOptimization):
    def __init__(self, approx_grad=True, m=10, factr=1e10, pgtol=1e-8, epsilon=1e-6, maxiter=200, bounds=None,
                 maxfun=15000, maxls=20, helper=None):
        super().__init__(helper)
        self.approx_grad = approx_grad
        self.m = m
        self.factr = factr
        self.pgtol = pgtol
        self.epsilon = epsilon
        self.maxiter = maxiter
        if bounds is None:
            self.bounds = []

        else:
            self.bounds = bounds

        self.maxfun = maxfun
        self.maxls = maxls

    def optimize(self, func, x0, bounds, args=(), **kwargs):

        x, f, d = fmin_l_bfgs_b(func=func, x0=x0, args=args, approx_grad=self.approx_grad, m=self.m, factr=self.factr,
                               pgtol=self.pgtol, epsilon=self.epsilon, maxiter=self.maxiter, bounds=bounds,
                               maxfun=self.maxfun, maxls=self.maxls)

        if self.helper is not None:
            self.helper.log_message("L-BFGS-B optimizer finished!")
            self.helper.log_message(str(d))

        return x

class BFGSOptimization(BaseOptimization):
    def __init__(self, gtol=1e-05, norm=np.inf, epsilon=1.4901161193847656e-08, maxiter=None,
                 full_output=True, disp=1, retall=False, helper=None):
        super().__init__(helper)
        self.gtol = gtol
        self.norm = norm
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.retall = retall
        self.helper = helper

    def optimize(self, func, x0, args=(), fprime=None, callback=None, **kwargs):
        xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = fmin_bfgs(func, x0, fprime, args, gtol=self.gtol,
                                                                             norm=self.norm, epsilon=self.epsilon,
                                                                             maxiter=self.maxiter, full_output=self.full_output,
                                                                             disp=self.disp, retall=self.retall, callback=callback)

        if self.helper is not None:
            self.helper.log_message("BFGS optimizer finished!")
            self.helper.log_message(f"BFGS stats: warnflag={warnflag}, error={fopt}, gradopt={gopt}, function calls={func_calls}, "
                                    f"gradient calls={grad_calls}, ")

        return xopt


class NelderMeadOptimization:
    def __init__(self, xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=True, disp=False, retall=False,
                 initial_simplex=None, helper=None):
        self.xtol = xtol
        self.ftol = ftol
        self.maxiter = maxiter
        self.maxfun = maxfun
        self.full_output = full_output
        self.disp = disp
        self.retall = retall
        self.initial_simplex = initial_simplex
        self.helper = helper

    def optimize(self, func, x0, args=(), callback=None, **kwargs):
        xopt, fopt, iter, funcalls, warnflag = fmin(func, x0, args, xtol=self.xtol, ftol=self.ftol, maxiter=self.maxiter,
                                                    maxfun=self.maxfun, full_output=self.full_output, disp=self.disp,
                                                    callback=callback, initial_simplex=self.initial_simplex)

        if self.helper is not None:
            self.helper.log_message("Nelder mead optimizer finished!")
            self.helper.log_message(f"Nelder mead stats: warnflag={warnflag}, error={fopt}, function calls={funcalls}, "
                                    f"number of iterations={iter}")

        return xopt


class ConjugateGradient:
    def __init__(self, gtol=1e-05, norm=np.inf, epsilon=1.4901161193847656e-08, maxiter=None, full_output=True, disp=False,
                 retall=False, helper=None):
        self.gtol = gtol
        self.norm = norm
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.retall = retall
        self.helper = helper

    def optimize(self, func, x0, args=(), fprime=None, callback=None, **kwargs):
        xopt, fopt, func_calls, grad_calls, warnflag = fmin_cg(func, x0, fprime=fprime, args=args, gtol=self.gtol,
                                                                        norm=self.norm, epsilon=self.epsilon,
                                                                        maxiter=self.maxiter, full_output=self.full_output,
                                                                        disp=self.disp, retall=self.retall, callback=callback)

        if self.helper is not None:
            self.helper.log_message("CG optimizer finished!")
            self.helper.log_message(f"CG stats: warnflag={warnflag}, error={fopt}, function calls={func_calls}, "
                                    f"gradient calls={grad_calls}")

        """
        warnflag : int
            Integer value with warning status, only returned if full_output is True.
            
            0 : Success.
            1 : The maximum number of iterations was exceeded.
            2 : Gradient and/or function calls were not changing. May indicate
            that precision was lost, i.e., the routine did not converge.
            
        """

        return xopt


class TrustConstr:
    def __init__(self, helper=None, hess=None, hessp=None, tol=None, constraints=(), opt_options=None):
        self.opt_options = opt_options or {}
        self.helper = helper
        self.hess = hess
        self.hessp = hessp
        self.constraints = constraints
        self.tol = tol

    def optimize(self, func, x0, bounds, args=(), callback=None, **kwargs):
        res = minimize(func, x0, args, method="trust-constr", hess=self.hess, hessp=self.hessp, constraints=self.constraints,
                       tol=self.tol, bounds=bounds,
                       callback=callback, options=self.opt_options)

        return res.x

# TODO fix this minimizer!
class TrustKrylov:
    def __init__(self, jac=None, hess=None, hessp=None, tol=None, options={'inexact': True},
                 helper=None):
        self.jac = jac
        self.hess = hess
        self.hessp = hessp
        self.tol = tol
        self.options = options
        self.helper = helper

    def optimize(self, func, x0, args=(), fprime=None, callback=None,**kwargs):
        res = minimize(func, x0, args, 'trust-krylov',
                                                                self.jac, self.hess, self.hessp, self.tol,
                                                                self.options, callback=callback)

        """
        warnflag : int
            Integer value with warning status, only returned if full_output is True.

            0 : Success.
            1 : The maximum number of iterations was exceeded.
            2 : Gradient and/or function calls were not changing. May indicate
            that precision was lost, i.e., the routine did not converge.

        """

        return res.x


class DifferentialEvolution(BaseOptimization):
    def __init__(self, strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7,
                 seed=None, disp=False, polish=True, init='latinhypercube', atol=0, updating='immediate',
                 workers=1, helper=None):
        super().__init__(helper)
        self.strategy = strategy
        self.maxiter = maxiter
        self.popsize = popsize
        self.tol = tol
        self.mutation = mutation
        self.recombination = recombination
        self.seed = seed
        self.disp=disp
        self.polish = polish
        self.init = init
        self.atol = atol
        self.updating = updating
        self.workers = workers

        self.helper = helper

    def create_uniform_dist(self, bounds):
        subsets = [[b[0] + (b[1] - b[0]) / (self.popsize - 1) * i for i in range(self.popsize)] for b in bounds]
        subsets_meshed = np.meshgrid(*subsets)
        subsets_meshed_r = [sm.ravel() for sm in subsets_meshed]
        inits = np.column_stack(subsets_meshed_r)

        self.init = inits

    def create_random_inits(self, bounds):
        init = []
        for i in range(self.popsize):
            init.append([random.uniform(b[0], b[1]) for b in bounds])

        self.init = init

    def optimize(self, func, bounds, args=(), callback=None, **kwargs):
        if self.init == "uniformdist":
            self.create_uniform_dist(bounds)

        elif self.init == "random_min":
            self.create_random_inits(bounds)

        result = differential_evolution(func, bounds, args=args, strategy=self.strategy, maxiter=self.maxiter,
                                        popsize=self.popsize, tol=self.tol, mutation=self.mutation,
                                        recombination=self.recombination, seed=self.seed, callback=callback,
                                        disp=self.disp, polish=self.polish, init=self.init, atol=self.atol,
                                        updating=self.updating, workers=self.workers)

        if self.helper is not None:
            self.helper.log_message("DE optimizer finished!")
            self.helper.log_message(result.message)
            self.helper.log_message(f"DE statistics: error={result.fun}, niterations={result.nit}, nfuncevaluations={result.nfev}")

        return result.x


class DifferentialEvolution_parallel():
    def __init__(self, parallel=True, strategy='best1bin', maxiter=None, popsize=15, tol=0.01,
                           mutation=(0.5, 1), recombination=0.7, seed=None, callback=None, disp=False, polish=True,
                           init='latinhypercube'):
        self.parallel = parallel
        self.strategy = strategy
        self.maxiter = maxiter
        self.popsize = popsize
        self.tol = tol
        self.mutation = mutation
        self.recombination = recombination
        self.seed = seed
        self.callback = callback
        self.disp = disp
        self.polish = polish
        self.init = init

    def optimize(self, func, bounds, args=()):
        result = differential_evolution_parallel(func, bounds, args, parallel=self.parallel, strategy=self.strategy,
                                                 maxiter=self.maxiter, popsize=self.popsize, tol=self.tol, mutation=self.mutation,
                                                 recombination=self.recombination, seed=self.seed, callback=self.callback,
                                                 disp=self.disp, polish=self.polish, init=self.init)

        return result.x

class DualAnnealing:
    def __init__(self, maxiter=1000, local_search_options=None, initial_temp=5230., restart_temp_ratio=2.e-5, visit=2.62,
                 accept=-5.0, maxfun=10000000, seed=None, no_local_search=False, callback=None):
        self.maxiter = maxiter
        self.local_search_options = local_search_options or {}
        self.initial_temp = initial_temp
        self.restart_temp_ratio = restart_temp_ratio
        self.visit = visit
        self.accept = accept
        self.maxfun = maxfun
        self.seed = seed
        self.no_local_search = no_local_search
        self.callback = callback

    def optimize(self, func, bounds, args=(), x0=None, **kwargs):
        result = dual_annealing(func, bounds, args, maxiter=self.maxiter, local_search_options=self.local_search_options,
                                initial_temp=self.initial_temp,
                                restart_temp_ratio=self.restart_temp_ratio, visit=self.visit, accept=self.accept,
                                maxfun=self.maxfun, seed=self.seed,
                                no_local_search=self.no_local_search, callback=self.callback, x0=x0)

        return result.x


optimizers_dict = {
    'de_parallel': DifferentialEvolution_parallel,
    'cg': ConjugateGradient,
    'de': DifferentialEvolution,
    'anneal': DualAnnealing,
    'lbfgsb': LBFGSBOptimization,
    'bfgs': BFGSOptimization,
    'nm': NelderMeadOptimization
}
