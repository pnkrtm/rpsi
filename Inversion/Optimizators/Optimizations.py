from scipy.optimize import fmin_l_bfgs_b, differential_evolution
from Inversion.Optimizators._differentialevolution import differential_evolution as differential_evolution_parallel


class LBFGSBOptimization:
    def __init__(self, approx_grad=True, m=10, factr=1e10, pgtol=1e-8, epsilon=1e-6, maxiter=200, bounds=[]):
        self.approx_grad = approx_grad
        self.m = m
        self.factr = factr
        self.pgtol = pgtol
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.bounds = bounds

    def optimize(self, func, x0, bounds, args=()):

        result = fmin_l_bfgs_b(func=func, x0=x0, args=args, approx_grad=self.approx_grad, m=self.m, factr=self.factr,
                               pgtol=self.pgtol, epsilon=self.epsilon, maxiter=self.maxiter, bounds=bounds)

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


optimizers_dict = {
    'de_parallel': DifferentialEvolution_parallel,
    'lbfgsb': LBFGSBOptimization
}
