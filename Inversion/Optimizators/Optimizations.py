from scipy.optimize import fmin_l_bfgs_b, differential_evolution, dual_annealing

from Inversion.Optimizators._differentialevolution import differential_evolution as differential_evolution_parallel


class LBFGSBOptimization:
    def __init__(self, approx_grad=True, m=10, factr=1e10, pgtol=1e-8, epsilon=1e-6, maxiter=200, bounds=None,
                 maxfun=15000, maxls=20):
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

    def optimize(self, func, x0, bounds, args=()):

        result = fmin_l_bfgs_b(func=func, x0=x0, args=args, approx_grad=self.approx_grad, m=self.m, factr=self.factr,
                               pgtol=self.pgtol, epsilon=self.epsilon, maxiter=self.maxiter, bounds=bounds,
                               maxfun=self.maxfun, maxls=self.maxls)

        return result[0]


class DifferentialEvolution:
    def __init__(self, strategy='best1bin', maxiter=None, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7,
                 seed=None, callback=None, disp=False, polish=True, init='latinhypercube', atol=0):
        self.strategy = strategy
        self.maxiter = maxiter
        self.popsize = popsize
        self.tol = tol
        self.mutation = mutation
        self.recombination = recombination
        self.seed = seed
        self.callback = callback
        self.disp=disp
        self.polish = polish
        self.init = init
        self.atol = atol

    def optimize(self, func, bounds, args=()):
        result = differential_evolution(func, bounds, args=args, strategy=self.strategy, maxiter=self.maxiter,
                                        popsize=self.popsize, tol=self.tol, mutation=self.mutation,
                                        recombination=self.recombination, seed=self.seed, callback=self.callback,
                                        disp=self.disp, polish=self.polish, init=self.init)

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
                 accept=-5.0, maxfun=1e7, seed=None, no_local_search=False, callback=None):
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

    def optimize(self, func, bounds, args=(), x0=None):
        result = dual_annealing(func, bounds, args, maxiter=self.maxiter, local_search_options=self.local_search_options,
                                initial_temp=self.initial_temp,
                                restart_temp_ratio=self.restart_temp_ratio, visit=self.visit, accept=self.accept,
                                maxfun=self.maxfun, seed=self.seed,
                                no_local_search=self.no_local_search, callback=self.callback, x0=x0)

        return result.x


optimizers_dict = {
    'de_parallel': DifferentialEvolution_parallel,
    'anneal': DualAnnealing,
    'lbfgsb': LBFGSBOptimization
}
