from scipy.optimize import fmin_l_bfgs_b, differential_evolution


class LBFGSBOptimization:
    def __init__(self):
        self.approx_grad = True
        self.m = 10
        self.factr = 1e10
        self.pgtol = 1e-8
        self.epsilon = 1e-6
        self.maxiter = 200
        self.bounds = []

    def optimize(self, func, x0, args):

        result = fmin_l_bfgs_b(func=func, x0=x0, args=args, approx_grad=self.approx_grad, m=self.m, factr=self.factr,
                               pgtol=self.pgtol, epsilon=self.epsilon, maxiter=self.maxiter, bounds=self.bounds)

        return result[0]


class DifferentialEvolution:
    def __init__(self, strategy='best1bin', maxiter=None, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7,
                 seed=None, callback=None, disp=False, polish=True, init='latinhypercube'):
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

    def optimize(self, func, bounds, args=()):
        result = differential_evolution(func, bounds, args=args, strategy=self.strategy, maxiter=self.maxiter,
                                        popsize=self.popsize, tol=self.tol, mutation=self.mutation,
                                        recombination=self.recombination, seed=self.seed, callback=self.callback,
                                        disp=self.disp, polish=self.polish, init=self.init)

        return result