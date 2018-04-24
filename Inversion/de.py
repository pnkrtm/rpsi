"""
differential_evolution: The differential evolution global optimization algorithm
Added by Andrew Nelson 2014
Updated for parallel by Pavel Ponomarev 2015
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize.optimize import _status_message
import numbers

__all__ = ['differential_evolution']

_MACHEPS = np.finfo(np.float64).eps


def differential_evolution(func, bounds, args=(),
                           parallel=False, strategy='best1bin', 
                           maxiter=None, popsize=15, tol=0.01,
                           mutation=(0.5, 1), recombination=0.7, seed=None,
                           callback=None, disp=False, polish=True,
                           init='latinhypercube', aggressive=True, njobs=None):
    """Finds the global minimum of a multivariate function.
    Differential Evolution is stochastic in nature (does not use gradient
    methods) to find the minimium, and can search large areas of candidate
    space, but often requires larger numbers of function evaluations than
    conventional gradient based techniques. Suitable to find the minimum of 
    non-differentiable, non-linear and noizy functions.

    The algorithm is due to Storn and Price [1]_.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    parallel : bool, optional
        If True (default False), changes the objective function 
        ``f([x], *args)`` to accept vectorized arguments, where [x] is 
        the list of parameters of the whole population. 
        This means in practice that current generation can be evaluated
        externally using parallel computing packages and functions like
        multiprocessing.Pool.map or joblib.Parallel, e.g.:
            Parallel(n_jobs=2)(delayed(func)(i) for i in [x])
        Then, the alghorithm expects the objective function also to return a 
        vector of energies of the same length. 
        When triggered, this automatically sets polish=False.
        Use of 'parallel' is justified when 'func' is time-consuming and heavy,
        otherwise, the overhang of parallel execution can overwhelm the possible
        time speedup.
    strategy : str, optional
        The differential mutation strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
        The default is 'best1bin'.
    aggressive : bool, optional
        If True (default), the best candidate can be used for mutation even 
        if it is in the current generation (or subgeneration, when 
        'parallel = True' and 'njobs' is set) itself. This aggressive mutation
        scheme shows better performance in most of the cases. 
        The False value corresponds to conventional DE alghorithm. 
        Non agresive mutation strategy does not mix the generations, which
        enables the evaluation of fitness for the whole generation in parallel.
        If 'njobs' is used, the quasi-parallel mutation strategy can be used
        in parallel.
    njobs : int, optional
        Sets the number of dispatched jobs (or length of the sub-populations)
        for parallel evaluation. Enables quasi-agressive strategy when
        "parallel" is used. The best candidae is updaed after evaluation of
        'njobs', and then, a new bunch of objective function parameters of
        'njobs' is dispatched for external evaluation.
        Can speedup the convergence when njobs is at least 2-5 times lower than
        the total population size. 
    maxiter : int, optional
        The maximum number of times the entire population is evolved.
        The maximum number of function evaluations is:
        ``maxiter * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals. Default is 15.
    tol : float, optional
        When the mean of the population energies, multiplied by tol,
        divided by the standard deviation of the population energies
        is greater than 1 the solving process terminates:
        ``convergence = mean(pop) * tol / stdev(pop) > 1``
        Default value is 'tol = 0.01'
    mutation : float or tuple(float, float), optional
        The mutation constant.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        ``U[min, max)``. Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. Increasing
        this value allows a larger number of mutants to progress into the next
        generation, but at the risk of population stability.
    seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.RandomState` singleton is used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with seed.
        If `seed` is already a `np.random.RandomState instance`, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Display status messages
    callback : callable, optional
        callback(x0, convergence=val)
        A function to follow the progress of the minimization. 
        ``val`` represents the fractional value of the population convergence.
        When ``val`` is greater than one the function halts. If callback 
        returns `True`, then the minimization is halted.
        callback(x0, convergence, *args, **kwargs) to send to callback all the
        parameters of the current population as ``x`` and all the energies
        as ``f``, can be used to realize own stopping criterions.
    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best population member at the end, which
        can improve the minimization slightly. Note, that `polish` sets 
        automatically to False if parallel is True.
    init : string, optional
        Specify how the population initialization is performed. Should be
        one of:
            - 'latinhypercube'
            - 'random'
        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space. 'random'
        initializes the population randomly - this has the drawback that
        clustering can occur, preventing the whole of parameter space
        being covered.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes. If `polish`
        was employed, then OptimizeResult also contains the `jac` attribute.

    Notes
    -----
    Differential evolution is a stochastic population based method that is
    useful for global optimization problems. At each pass through the
    population the algorithm mutates each candidate solution by mixing with
    other candidate solutions to create a trial candidate. There are several
    strategies [2]_ for creating trial candidates, which suit some problems
    more than others. The 'best1bin' strategy is a good starting point for
    many systems. In this strategy two members of the population are
    randomly chosen. Their difference is used to mutate the best member
    (the `best` in `best1bin`), :math:`b_0`, so far:

    .. math::

        b' = b_0 + mutation * (population[rand0] - population[rand1])

    A trial vector is then constructed. Starting with a randomly chosen 'i'th
    parameter the trial is sequentially filled (in modulo) with parameters from
    `b'` or the original candidate. The choice of whether to use `b'` or the
    original candidate is made with a binomial distribution (the 'bin' in
    'best1bin') - a random number in [0, 1) is generated.  If this number is
    less than the `recombination` constant then the parameter is loaded from
    `b'`, otherwise it is loaded from the original candidate.  The final
    parameter is always loaded from `b'`.  Once the trial candidate is built
    its fitness is assessed. If the trial is better than the original candidate
    then it takes its place. If it is also better than the best overall
    candidate it also replaces that.
    To improve your chances of finding a global minimum use higher `popsize`
    values, with higher `mutation` and (dithering), but lower `recombination`
    values. This has the effect of widening the search radius, but slowing
    convergence.

    .. versionadded:: 0.15.0

    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function. This
    function is implemented in `rosen` in `scipy.optimize`.

    >>> from scipy.optimize import rosen, differential_evolution
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    >>> result = differential_evolution(rosen, bounds)
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)

    Next find the minimum of the Ackley function
    (http://en.wikipedia.org/wiki/Test_functions_for_optimization).

    >>> from scipy.optimize import differential_evolution
    >>> import numpy as np
    >>> def ackley(x):
    ...     arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    ...     arg2 = 0.5 * (np.cos(2.*np.pi * x[0]) + np.cos(2.*np.pi * x[1]))
    ...     return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
    >>> bounds = [(-5, 5), (-5, 5)]
    >>> result = differential_evolution(ackley, bounds)
    >>> result.x, result.fun
    (array([ 0.,  0.]), 4.4408920985006262e-16)
    
    Utilize the parallelization capability of the DE for 
    heavy objective unctions:
    
    from scipy.optimize import rosen
    from _differentialevolution import differential_evolution as de
    from joblib import Parallel, delayed, cpu_count
    import time

    bounds = [(0,2), (0, 2), (0,2)]
    itn = 10
    def objfunc(params):
        # a heavy objective function
        for it in xrange(1000000):
            it**2
        return rosen(params)

    def pareval(listcoords):
        listresults = Parallel(n_jobs=cpu_count())\
                            (delayed(objfunc)(i) for i in listcoords) 
        return listresults

    def parallel_run():
        result = de(pareval, bounds, maxiter=itn, parallel=True, 
                    aggressive=False, njobs=10, disp=False)
        return result.fun
        
    def serial_run():
        result = de(objfunc, bounds, maxiter=itn, polish=False, 
                    aggressive=False, disp=False)
        return result.fun
        
    start_time = time.time()
    fser = serial_run()
    tser = time.time() - start_time
    start_time = time.time()
    fpar = parallel_run()

    print "Parallel run took %s seconds using %s cores" %\
             ((time.time() - start_time), cpu_count())
    print fpar
    print("Serial run took %s seconds using 1 core " % tser)   
    print fser
    
    

    References
    ----------
    .. [1] Storn, R and Price, K, Differential Evolution - a Simple and
           Efficient Heuristic for Global Optimization over Continuous Spaces,
           Journal of Global Optimization, 1997, 11, 341 - 359.
    .. [2] http://www1.icsi.berkeley.edu/~storn/code.html
    .. [3] http://en.wikipedia.org/wiki/Differential_evolution
    .. [4] http://www.dii.unipd.it/~alotto/didattica/corsi/Elettrotecnica%20computazionale/DE.pdf
    """

    solver = DifferentialEvolutionSolver(func, bounds, args=args,
                                         parallel=parallel, strategy=strategy,
                                         maxiter=maxiter, popsize=popsize,
                                         tol=tol, mutation=mutation,
                                         recombination=recombination,
                                         seed=seed, polish=polish,
                                         callback=callback,
                                         disp=disp,
                                         init=init, aggressive=aggressive,
                                         njobs=njobs)
    return solver.solve()


class DifferentialEvolutionSolver(object):

    """This class implements the differential evolution solver

    Parameters same as in the differential_evolution function.
    """

    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {'best1bin': '_best1',
                 'randtobest1bin': '_randtobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2',
                 'rand1bin': '_rand1'}
    _exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2'}

    def __init__(self, func, bounds, args=(), parallel=False,
                 strategy='best1bin', maxiter=None, popsize=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 maxfun=None, callback=None, disp=False, polish=True,
                 init='latinhypercube', aggressive=True, njobs=None ):

        self.parallel = parallel

        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy

        self.callback = callback
        
        # set polish false in the parallel case
        if self.parallel:
            self.polish = False
        else:
            self.polish = polish
            
        self.aggressive = aggressive
            
        self.tol = tol

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.scale = mutation
        if (not np.all(np.isfinite(mutation)) or
                np.any(np.array(mutation) >= 2) or
                np.any(np.array(mutation) < 0)):
            raise ValueError('The mutation constant must be a float in '
                             'U[0, 2), or specified as a tuple(min, max)'
                             ' where min < max and min, max are in U[0, 2).')

        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination

        self.func = func
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        self.limits = np.array(bounds, dtype='float').T
        if (np.size(self.limits, 0) != 2 or not
                np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence containing '
                             'real valued (min, max) pairs for each value'
                             ' in x')

        self.maxiter = maxiter or 1000
        
        # Maximum number of function evaluations is determined by the 
        # maximum number of generations, or maxiter.
        # This variable is kept in order to keep compatibility as someone 
        # might be catching status messages about 'maxfev' 
        # (I would remove as it is obvious that just 'maxiter' is enough here. 
        # The 'nfev' will not correlate with 'nit' only when polish is used. 
        # And this is apparently not important, as when polish is used the 
        # 'maxfun' is not monitored at all)
        self.maxfun = (self.maxiter + 1) * popsize * np.size(self.limits, 1)

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        parameter_count = np.size(self.limits, 1)
        
        if self.parallel:
            if njobs is None:
                # standart non-aggressive parallel evaluation
                self.njobs = popsize * parameter_count
                self.aggressive = False
            elif isinstance(njobs, int) and (popsize*parameter_count > njobs):
                # Sets the sub-population size for quasi-aggressive mutation
                # strategy in parallel.
                self.njobs = njobs
            else:
                raise ValueError("The njobs must be integer and smaller than"
                                    "the total population size")
                                    
        self.random_number_generator = _make_random_gen(seed)

        # default initialization is a latin hypercube design, but there
        # are other population initializations possible.
        self.population = np.zeros((popsize * parameter_count,
                                    parameter_count))
        self.disp = disp
        if self.disp:
            print("Population size is %s" % (popsize * parameter_count))

        if init == 'latinhypercube':
            self.init_population_lhs()
        elif init == 'random':
            self.init_population_random()
        else:
            raise ValueError("The population initialization method must be one"
                             "of 'latinhypercube' or 'random'")

        self.population_energies = np.ones(
            popsize * parameter_count) * np.inf


    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling
        Latin Hypercube Sampling ensures that the sampling of parameter space
        is maximised.
        """
        samples = np.size(self.population, 0)
        N = np.size(self.population, 1)
        rng = self.random_number_generator

        # Generate the intervals
        segsize = 1.0 / samples

        # Fill points uniformly in each interval
        rdrange = rng.rand(samples, N) * segsize
        rdrange += np.atleast_2d(
            np.linspace(0., 1., samples, endpoint=False)).T

        # Make the random pairings
        self.population = np.zeros_like(rdrange)

        for j in range(N):
            order = rng.permutation(range(samples))
            self.population[:, j] = rdrange[order, j]

    def init_population_random(self):
        """
        Initialises the population at random.  This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.random_sample(self.population.shape)

    @property
    def x(self):
        """
        The best solution from the solver

        Returns
        -------
        x - ndarray
            The best solution from the solver.
        """
        return self._scale_parameters(self.population[0])

    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes. If polish
            was employed, then OptimizeResult also contains the ``hess_inv``
            and ``jac`` attributes.
        """
        
        # initialize counters of objfun evaluations and number of iterations
        nfev, nit = 0, 0
        warning_flag = False
        status_message = _status_message['success']

        # calculate energies to start with
        if self.parallel:
            # vectorized candidate evaluation
            params = []
            for candidate in self.population:
                params.append(self._scale_parameters(candidate))

                nfev += 1

            self.population_energies = self.func(params, *self.args)

        else:
            # or serial candidate evaluation needed for possible
            # agressive srategies
            for index, candidate in enumerate(self.population):
                parameters = self._scale_parameters(candidate)
                self.population_energies[index] = self.func(parameters,
                                                            *self.args)
                nfev += 1
               
               
        minval = np.argmin(self.population_energies)

        # put the lowest energy into the best solution position
        lowest_energy = self.population_energies[minval]
        self.population_energies[minval] = self.population_energies[0]
        self.population_energies[0] = lowest_energy  
        # and exchange places of previous and new best solutions
        self.population[[0, minval], :] = self.population[[minval, 0], :]

        if warning_flag:
            return OptimizeResult(
                           x=self.x,
                           fun=self.population_energies[0],
                           nfev=nfev,
                           nit=nit,
                           message=status_message,
                           success=(warning_flag is not True))

        # do the optimization.
        for nit in range(1, self.maxiter + 1):
            if self.dither is not None:
                self.scale = self.random_number_generator.rand(
                    )*(self.dither[1] - self.dither[0]) + self.dither[0]

            if self.parallel:
                # determine number of bunches of parameters 'nbun', or number of 
                # parallel evaluations, which is 'nbun + 1'
                # and the length of the reminder 'nrem'
                nbun, nrem = divmod(np.size(self.population, 0), self.njobs)
                itbun = 0
                
                # iterate among bunches
                while itbun <= nbun:
                    # determine the length of the bunch
                    lenb = self.njobs if (itbun < nbun) else nrem
                    paramsbun = []
                    # mutate parameters for the bunch
                    for candidate in xrange(lenb):
                        trial = self._mutate(candidate + self.njobs*itbun)
                        self._ensure_constraint(trial)
                        paramsbun.append(self._scale_parameters(trial))
                        nfev += 1
                        
                    # in parallel case the self.func must return a list of energies
                    # for the whole bunch
                    bunenergies = self.func(paramsbun, *self.args)
                    
                    # update population and their energies if bunch members are 
                    # better by iteration among all jobs results in the bunch
                    for itjob in range(len(bunenergies)):
                        energy = bunenergies[itjob]
                        if energy < self.population_energies[itjob + self.njobs*itbun]:
                            self.population[itjob + self.njobs*itbun] =\
                                self._unscale_parameters(paramsbun[itjob])
                            self.population_energies[itjob + self.njobs*itbun] = energy
                            
                            # update global best if there is a better in the bunch
                            # and strategy is aggressive
                            if self.aggressive:
                                if energy < self.population_energies[0]:
                                    self.population_energies[0] = energy
                                    self.population[0] =\
                                        self._unscale_parameters(paramsbun[itjob])
                                    if self.disp:
                                        print("differential_evolution step %d: f(x)= %g"
                                              % (nit, self.population_energies[0]))
                                        print self._scale_parameters(self.population[0])
                                    self.population[[0, itjob + self.njobs*itbun], :] =\
                                        self.population[[itjob + self.njobs*itbun, 0], :]
                    itbun += 1
                   
                                 
            else:
                # make serial evaluation of the func
                # serial evaluation is required for possible 
                # agressive strategies
                for candidate in range(np.size(self.population, 0)):

                    trial = self._mutate(candidate)
                    self._ensure_constraint(trial)
                    parameters = self._scale_parameters(trial)

                    energy = self.func(parameters, *self.args)
                    nfev += 1

                    if energy < self.population_energies[candidate]:
                        self.population[candidate] = trial
                        self.population_energies[candidate] = energy
                        
                        # update global best if aggressive mutation is used
                        if self.aggressive:
                            if energy < self.population_energies[0]:
                                self.population_energies[0] = energy
                                self.population[0] = trial
                                if self.disp:
                                    print("differential_evolution step %d: f(x)= %g"
                                          % (nit, self.population_energies[0]))
                                    print self._scale_parameters(self.population[0])
                                # and exchange places of previous and new best solutions
                                self.population[[0, candidate], :] = self.population[[candidate, 0], :]
            
            if not self.aggressive:
                # put the lowest energy into the best solution position                    
                minval = np.argmin(self.population_energies)

                lowest_energy = self.population_energies[minval]
                self.population_energies[minval] = self.population_energies[0]
                self.population_energies[0] = lowest_energy 
                
                if self.disp:
                    print("differential_evolution step %d: f(x)= %g"
                          % (nit, self.population_energies[0]))
                    print self._scale_parameters(self.population[0])
                    
                # and exchange places of previous and new best solutions
                self.population[[0, minval], :] = self.population[[minval, 0], :]
        
            
            
            if self.disp:
                print("differential_evolution step %d: f(x)= %g"
                      % (nit, self.population_energies[0]))
                print self._scale_parameters(self.population[0])
            
                
            # stop when the fractional s.d. of the population is less than tol
            # of the mean energy
            convergence = (np.std(self.population_energies) /
                           np.abs(np.mean(self.population_energies) +
                                  _MACHEPS))
        
            if (self.callback and
                    self.callback(self._scale_parameters(self.population[0]),
                                  convergence=self.tol / convergence,
                                  x=self._scale_parameters(self.population),
                                  f=self.population_energies) is True):

                warning_flag = True
                status_message = ('callback function requested stop early '
                                  'by returning True')
                break

            if convergence < self.tol or warning_flag:
                break

        else:
            status_message = _status_message['maxiter']
            warning_flag = True

        DE_result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is not True))

        if self.polish:
            result = minimize(self.func,
                              np.copy(DE_result.x),
                              method='L-BFGS-B',
                              bounds=self.limits.T,
                              args=self.args)

            nfev += result.nfev
            DE_result.nfev = nfev

            if result.fun < DE_result.fun:
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                # to keep internal state consistent
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)

        return DE_result

    def _scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """
        make sure the parameters lie between the limits
        """
        for index, param in enumerate(trial):
            if param > 1 or param < 0:
                trial[index] = self.random_number_generator.rand()

    def _mutate(self, candidate):
        """
        create a trial vector based on a mutation strategy
        """
        trial = np.copy(self.population[candidate])
        parameter_count = np.size(trial, 0)

        fill_point = self.random_number_generator.randint(0, parameter_count)

        if (self.strategy == 'randtobest1exp' or
                self.strategy == 'randtobest1bin'):

            bprime = self.mutation_func(candidate,
                                        self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        if self.strategy in self._binomial:
            crossovers = self.random_number_generator.rand(parameter_count)
            crossovers = crossovers < self.cross_over_probability
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        elif self.strategy in self._exponential:
            i = 0
            while (i < parameter_count and
                   self.random_number_generator.rand() <
                   self.cross_over_probability):

                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % parameter_count
                i += 1

            return trial

    def _best1(self, samples):
        """
        best1bin, best1exp
        """
        r0, r1 = samples[:2]
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        """
        rand1bin, rand1exp
        """
        r0, r1, r2 = samples[:3]
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _randtobest1(self, candidate, samples):
        """
        randtobest1bin, randtobest1exp
        """
        r0, r1 = samples[:2]
        bprime = np.copy(self.population[candidate])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r0] -
                                self.population[r1])
        return bprime

    def _best2(self, samples):
        """
        best2bin, best2exp
        """
        r0, r1, r2, r3 = samples[:4]
        bprime = (self.population[0] + self.scale*(
                    self.population[r0] + self.population[r1] -
                    self.population[r2] - self.population[r3]))

        return bprime

    def _rand2(self, samples):
        """
        rand2bin, rand2exp
        """
        r0, r1, r2, r3, r4 = samples
        bprime = (self.population[r0] + self.scale*(
                    self.population[r1] + self.population[r2] -
                    self.population[r3] - self.population[r4]))

        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(np.size(self.population, 0)),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(np.size(self.population, 0)))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs


def _make_random_gen(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)