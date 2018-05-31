from __future__ import print_function

import numpy as np
import cvxopt as cvx
from cvxopt import solvers
from scipy.special import gammaln
import scipy
import scipy.optimize as spopt
import sys, time, math, pdb
from multiprocessing import Process
from multiprocessing.queues import Queue
from pathos.multiprocessing import ProcessingPool as Pool
from numba import jit
# from multiprocessing import Pool

# suppress optimizer output
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 20
np.random.seed(10)

# defining some constants
EPS = np.finfo(np.double).tiny
MAX = np.finfo(np.double).max

# defining some simple functions
logistic = lambda x: 1./(1+np.exp(x))
insum = lambda x,axes: np.apply_over_axes(np.sum,x,axes)

@jit
def polygamma2(n, x, d):

    n_arr = np.asarray(n)
    x_arr = np.asarray(x)
    fac2 = (-1.0)**(n_arr+1) * scipy.special.gamma(n_arr+1.0) * scipy.special.zeta(n_arr+1, x_arr)
    return np.where(n_arr == 0, d, fac2)

@jit
def digamma3(x):
    return np.vectorize(digamma2)(x)

@jit
def digamma2(x):
 #  Check the input.
 #
  if ( x <= 0.0 ):
    value = 0.0
    return value
 #
 #  Initialize.
 #
  value = 0.0
 #
 #  Use approximation for small argument.
 #
  if ( x <= 0.000001 ):
    euler_mascheroni = 0.57721566490153286060
    value = - euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x
    return value
 #
 #  Reduce to DIGAMA(X + N).
 #
  while ( x < 8.5 ):
    value = value - 1.0 / x
    x = x + 1.0
 #
 #  Use Stirling's (actually de Moivre's) expansion.
 #
  r = 1.0 / x
  value = value + np.log ( x ) - 0.5 * r
  r = r * r
  value = value \
    - r * ( 1.0 / 12.0 \
    - r * ( 1.0 / 120.0 \
    - r * ( 1.0 / 252.0 \
    - r * ( 1.0 / 240.0 \
    - r * ( 1.0 / 132.0 ) ) ) ) )
  return value

def nplog(x):
    """Compute the natural logarithm, handling very
    small floats appropriately.

    """
    try:
        x[x<EPS] = EPS
    except TypeError:
        x = max([x,EPS])
    return np.log(x)

def run_parallel(f, arg_values, cores, reps, J, is_update=True):
    processes_needed = reps * J if is_update else reps
    if cores >= processes_needed:
        print("Running optimize with Process")
        results = []
        queues  = [Queue() for i in range(J)]
        arg_values = [arg + (queues[index],) for index, arg in enumerate(arg_values)]
        jobs    = [Process(target=f['Process'], args=(list(arg_values[i]))) for i in range(J)]

        for job in jobs:
            job.start()
        for queue in queues:
            results.append(queue.get())
        for job in jobs:
            job.join()
        return results
    else:
        print("Running optimize with Pool")
        arg_values = [arg + (None,) for index, arg in enumerate(arg_values)]
        my_pool = Pool(cores / reps)
        return my_pool.map(f['Pool'], arg_values)


class Data:
    """
    A data structure to store a multiscale representation of
    chromatin accessibility read counts across `N` genomic windows of
    length `L` in `R` replicates.

    Arguments
        reads : array

    """
    def __init__(self):

        self.N = 0
        self.L = 0
        self.R = 0
        self.J = 0
        self.valueA = dict()
        self.valueB = dict()
        self.total = dict()

    def transform_to_multiscale(self, reads):
        """Transform a vector of read counts
        into a multiscale representation.

        .. note::
            See msCentipede manual for more details.

        """

        self.N = reads.shape[0]
        self.L = reads.shape[1]
        print("0: %s, 1: %s, 2: %s" % (str(reads.shape[0]), str(reads.shape[1]), str(reads.shape[2])))
        self.R = reads.shape[2]
        self.J = math.frexp(self.L)[1]-1
        for j in range(self.J):
            size = self.L/(2**(j+1))
            self.total[j] = np.array([reads[:,k*size:(k+2)*size,:].sum(1) for k in range(0,2**(j+1),2)]).T
            self.valueA[j] = np.array([reads[:,k*size:(k+1)*size,:].sum(1) for k in range(0,2**(j+1),2)]).T
            self.valueB[j] = self.total[j] - self.valueA[j]

    def inverse_transform(self):
        """Transform a multiscale representation of the data or parameters,
        into vector representation.

        """

        if self.data:
            profile = np.array([val for k in range(2**self.J) \
                for val in [self.value[self.J-1][k][0],self.value[self.J-1][k][1]-self.value[self.J-1][k][0]]])
        else:
            profile = np.array([1])
            for j in range(self.J):
                profile = np.array([p for val in profile for p in [val,val]])
                vals = np.array([i for v in self.value[j] for i in [v,1-v]])
                profile = vals*profile

        return profile

    def copy(self):
        """ Create a copy of the class instance
        """

        newcopy = Data()
        newcopy.J = self.J
        newcopy.N = self.N
        newcopy.L = self.L
        newcopy.R = self.R
        for j in range(self.J):
            newcopy.valueA[j] = self.valueA[j]
            newcopy.valueB[j] = self.valueB[j]
            newcopy.total[j] = self.total[j]

        return newcopy

class Zeta():
    """
    Inference class to store and update (E-step) the posterior
    probability that a transcription factor is bound to a motif
    instance.

    Arguments
        data : Data
        totalreads : array

    """

    def __init__(self, totalreads, N, infer):

        self.N = N
        self.total = totalreads

        if infer:
            self.prior_log_odds = np.zeros((self.N,1), dtype=float)
            self.footprint_log_likelihood_ratio = np.zeros((self.N,1), dtype=float)
            self.total_log_likelihood_ratio = np.zeros((self.N,1), dtype=float)
            self.posterior_log_odds = np.zeros((self.N,1), dtype=float)
        else:
            self.estim = np.zeros((self.N, 2),dtype=float)
            order = np.argsort(self.total.sum(1))
            indices = order[:self.N/2]
            self.estim[indices,1:] = -MAX
            indices = order[self.N/2:]
            self.estim[indices,1:] = MAX
            self.estim = np.exp(self.estim - np.max(self.estim,1).reshape(self.N,1))
            self.estim = self.estim / insum(self.estim,[1])

    def update(self, data, scores, \
        pi, tau, alpha, beta, omega, \
        pi_null, tau_null, model):

        footprint_logodds = np.zeros((self.N,1), dtype=float)
        lhoodA, lhoodB = compute_footprint_likelihood(data, pi, tau, pi_null, tau_null, model)

        for j in range(data.J):
            footprint_logodds += insum(lhoodA.valueA[j] - lhoodB.valueA[j],[1])

        prior_logodds = insum(beta.estim * scores, [1])
        negbin_logodds = insum(gammaln(self.total + alpha.estim.T[1]) \
                - gammaln(self.total + alpha.estim.T[0]) \
                + gammaln(alpha.estim.T[0]) - gammaln(alpha.estim.T[1]) \
                + alpha.estim.T[1] * nplog(omega.estim.T[1]) - alpha.estim.T[0] * nplog(omega.estim.T[0]) \
                + self.total * (nplog(1 - omega.estim.T[1]) - nplog(1 - omega.estim.T[0])),[1])

        self.estim[:,1:] = prior_logodds + footprint_logodds + negbin_logodds
        self.estim[:,0] = 0.
        self.estim[self.estim==np.inf] = MAX
        self.estim = np.exp(self.estim-np.max(self.estim,1).reshape(self.N,1))
        self.estim = self.estim/insum(self.estim,[1])

    def infer(self, data, scores, \
        pi, tau, alpha, beta, omega, \
        pi_null, tau_null, model):

        lhoodA, lhoodB = compute_footprint_likelihood(data, pi, tau, pi_null, tau_null, model)

        for j in range(data.J):
            self.footprint_log_likelihood_ratio += insum(lhoodA.valueA[j] - lhoodB.valueA[j],[1])
        self.footprint_log_likelihood_ratio = self.footprint_log_likelihood_ratio / np.log(10)

        self.prior_log_odds = insum(beta.estim * scores, [1]) / np.log(10)

        self.total_log_likelihood_ratio = insum(gammaln(self.total + alpha.estim.T[1]) \
            - gammaln(self.total + alpha.estim.T[0]) \
            + gammaln(alpha.estim.T[0]) - gammaln(alpha.estim.T[1]) \
            + alpha.estim.T[1] * nplog(omega.estim.T[1]) - alpha.estim.T[0] * nplog(omega.estim.T[0]) \
            + self.total * (nplog(1 - omega.estim.T[1]) - nplog(1 - omega.estim.T[0])),[1])
        self.total_log_likelihood_ratio = self.total_log_likelihood_ratio / np.log(10)

        self.posterior_log_odds = self.prior_log_odds \
            + self.footprint_log_likelihood_ratio \
            + self.total_log_likelihood_ratio


class Pi(Data):
    """
    Class to store and update (M-step) the parameter `p` in the
    msCentipede model. It is also used for the parameter `p_o` in
    the msCentipede-flexbg model.

    Arguments
        J : int
        number of scales

    """

    def __init__(self, J):

        self.J = J
        self.value = dict()
        for j in range(self.J):
            self.value[j] = np.empty((2**j,), dtype='float')

    def __reduce__(self):
        return (rebuild_Pi, (self.J, self.value))

    def update(self, data, zeta, tau):
        """Update the estimates of parameter `p` (and `p_o`) in the model.
        """

        zetaestim = zeta.estim[:,1].sum()

        # call optimizer

        print("Number of Optimizer calls: %s" % str(self.J))

        # Set up args
        arg_vals = []
        for j in range(self.J):
            # initialize optimization variable
            xo = self.value[j].copy()
            X = xo.size

            # set constraints for optimization variable
            xmin = 1./tau.estim[j]*np.ones((X,1),dtype=float)
            xmax = (1-1./tau.estim[j])*np.ones((X,1),dtype=float)
            G = np.vstack((np.diag(-1*np.ones((X,), dtype=float)), np.diag(np.ones((X,), dtype=float))))
            h = np.vstack((-1*xmin,xmax))

            arg_vals.append(dict([('G',G),('h',h),('data',data),('zeta',zeta),('tau',tau),('zetaestim',zetaestim),('j',j)]))

        results = run_parallel({'Process': parallel_optimize_process, 'Pool': parallel_optimize_pool},
                              ((self.value[j].copy(), arg_vals[j], self.J) for j in range(self.J)), 21, data.R, self.J)

        for j in range(self.J):
            self.value[j] = results[j]

def parallel_optimize_process(xo, args, J, queue):
    return parallel_optimize_pool((xo, args, J, queue))

def parallel_optimize_pool(params):
    xo, args, J, queue = params

    my_x_final = optimizer(xo, pi_function_gradient, pi_function_gradient_hessian, args, J)

    if np.isnan(my_x_final).any():
        print("Nan in Pi")
        raise ValueError

    if np.isinf(my_x_final).any():
        print("Inf in Pi")
        raise ValueError

    if queue is not None:
        queue.put(my_x_final)
    else:
        return my_x_final

def rebuild_Pi(J, value):

    pi = Pi(J)
    pi.value = value
    return pi

def pi_gamma_calculations(val_A, val_B, alpha, beta, queue):
    data_alpha = val_A + alpha
    data_beta  = val_B + beta
    new_func = gammaln(data_alpha) + gammaln(data_beta)
    new_df   = digamma3(data_alpha) - digamma3(data_beta)
    queue.put((new_func, new_df))

def pi_function_gradient(x, args, J):

    """Computes part of the likelihood function that has
    terms containing `pi`, along with its gradient
    """

    data = args['data']
    zeta = args['zeta']
    tau = args['tau']
    zetaestim = args['zetaestim']
    j = args['j']

    J = 2**j
    func = np.zeros((data.N,J), dtype=float)
    df = np.zeros((data.N,J), dtype=float)
    alpha = x * tau.estim[j]
    beta = (1-x) * tau.estim[j]

    val_A = data.valueA[j]
    val_B = data.valueB[j]

    # LOOP TO PARALLELIZE
    results = []
    queues = [Queue() for i in range(data.R)]
    jobs   = [Process(target=pi_gamma_calculations, args=(val_A[r], val_B[r], alpha, beta, queues[r])) for r in range(data.R)]

    for job in jobs: job.start()
    for q in queues: results.append(q.get())
    for job in jobs: job.join()

    for r in range(data.R):
        this_result = results[r]
        func += this_result[0]
        df   += this_result[1]

    F  = np.sum(func,1) - np.sum(gammaln(alpha) + gammaln(beta)) * data.R
    Df = tau.estim[j] * (np.sum(zeta.estim[:,1:] * df,0) \
        - zetaestim * (digamma3(alpha) - digamma3(beta)) * data.R)

    f  = -1. * np.sum(zeta.estim[:,1] * F)
    Df = -1. * Df

    return f, Df

def pi_gamma_calculations_hess(val_A, val_B, alpha, beta, queue):
    data_alpha = val_A + alpha
    data_beta  = val_B + beta

    new_func = gammaln(data_alpha) + gammaln(data_beta)

    dg_tmp_a = digamma3(data_alpha)
    dg_tmp_b = digamma3(data_beta)

    new_df = dg_tmp_a - dg_tmp_b
    new_hf = polygamma2(1, data_alpha, dg_tmp_a) + polygamma2(1, data_beta, dg_tmp_b)

    queue.put((new_func, new_df, new_hf))

def pi_function_gradient_hessian(x, args, J):

    """Computes part of the likelihood function that has
    terms containing `pi`, along with its gradient and hessian
    """

    data = args['data']
    zeta = args['zeta']
    tau = args['tau']
    zetaestim = args['zetaestim']
    j = args['j']

    hess = np.zeros((x.size,), dtype=float)

    J = 2**j
    func = np.zeros((data.N,J), dtype=float)
    df = np.zeros((data.N,J), dtype=float)
    hf = np.zeros((data.N,J), dtype=float)
    alpha = x * tau.estim[j]
    beta = (1-x) * tau.estim[j]

    val_A = data.valueA[j]
    val_B = data.valueB[j]

    results = []
    queues = [Queue() for i in range(data.R)]
    jobs   = [Process(target=pi_gamma_calculations_hess, args=(val_A[r], val_B[r], alpha, beta, queues[r])) for r in range(data.R)]

    for job in jobs: job.start()
    for q in queues: results.append(q.get())
    for job in jobs: job.join()

    for r in range(data.R):
        this_result = results[r]
        func += this_result[0]
        df   += this_result[1]
        hf   += this_result[2]

    F = np.sum(func,1) - np.sum(gammaln(alpha) + gammaln(beta)) * data.R

    dg_alpha = digamma3(alpha)
    dg_beta = digamma3(beta)

    Df = tau.estim[j] * (np.sum(zeta.estim[:,1:] * df,0) \
        - zetaestim * (dg_alpha - dg_beta) * data.R)
    hess = tau.estim[j]**2 * (np.sum(zeta.estim[:,1:] * hf,0) \
        - zetaestim * (polygamma2(1, alpha, dg_alpha) + polygamma2(1, beta, dg_beta)) * data.R)

    f = -1. * np.sum(zeta.estim[:,1] * F)
    Df = -1. * Df
    Hf = np.diag(-1.*hess)

    return f, Df, Hf

class Tau():
    """
    Class to store and update (M-step) the parameter `tau` in the
    msCentipede model. It is also used for the parameter `tau_o` in
    the msCentipede-flexbg model.

    Arguments
        J : int
        number of scales

    """

    def __init__(self, J):

        self.J = J
        self.estim = np.empty((self.J,), dtype='float')

    def __reduce__(self):
        return (rebuild_Tau, (self.J,self.estim))

    def update(self, data, zeta, pi):
        """Update the estimates of parameter `tau` (and `tau_o`) in the model.
        """

        zetaestim = np.sum(zeta.estim[:,1])
        print("Number of Optimizer calls: %s" % str(self.J))

        arg_vals = []
        minj_vals = []
        xmin_vals = []

        for j in range(self.J):
            xo = self.estim[j:j+1]

            # set constraints for optimization variables
            minj = 1./min([np.min(pi.value[j]), np.min(1-pi.value[j])])
            minj_vals.append(minj)
            xmin = np.array([minj])
            xmin_vals.append(xmin)
            G = np.diag(-1 * np.ones((1,), dtype=float))
            h = -1*xmin.reshape(1,1)

            # additional arguments
            arg_vals.append(dict([('j',j),('G',G),('h',h),('data',data),('zeta',zeta),('pi',pi),('zetaestim',zetaestim)]))

        results = run_parallel({'Process': tau_parallel_optimize_process, 'Pool': tau_parallel_optimize_pool},
                              ((self.estim[j:j+1], xmin_vals[j], minj_vals[j], arg_vals[j], self.J) for j in xrange(self.J)), 21, data.R, self.J)

        for j in range(self.J):
            self.estim[j:j+1] = results[j]

        if np.isnan(self.estim).any():
            print("Nan in Tau")
            raise ValueError

        if np.isinf(self.estim).any():
            print("Inf in Tau")
            raise ValueError

def tau_parallel_optimize_process(xo, xmin, minj, args, J, queue):
    return tau_parallel_optimize_pool((xo, xmin, minj, args, J, queue))

def tau_parallel_optimize_pool(params):
    xo, xmin, minj, args, J, queue = params
    try:
        x_final = optimizer(xo, tau_function_gradient, tau_function_gradient_hessian, args, J)
    except ValueError:
        xo = xmin+100*np.random.rand()
        bounds = [(minj, None)]
        solution = spopt.fmin_l_bfgs_b(tau_function_gradient, xo, \
            args=(args,), bounds=bounds)
        x_final = solution[0]

    if queue is not None:
        queue.put(x_final)
    else:
        return x_final


def rebuild_Tau(J, estim):

    tau = Tau(J)
    tau.estim = estim
    return tau

def tau_gamma_calculations(val_A, val_B, val_T, alpha, beta, x, pi_val, queue):
    data_alpha = val_A + alpha
    data_beta  = val_B + beta
    data_x     = val_T + x

    new_func = np.sum(gammaln(data_alpha),1) \
             + np.sum(gammaln(data_beta),1) \
             - np.sum(gammaln(data_x),1)

    new_df   = np.sum(pi_val*digamma3(data_alpha),1) \
             + np.sum((1-pi_val)*digamma3(data_beta),1) \
             - np.sum(digamma3(data_x),1)

    queue.put((new_func, new_df))

def tau_function_gradient(x, args, J):
    """Computes part of the likelihood function that has
    terms containing `tau`, and its gradient.
    """

    data = args['data']
    zeta = args['zeta']
    pi = args['pi']
    zetaestim = args['zetaestim']
    j = args['j']

    func = np.zeros((zeta.N,), dtype=float)
    ffunc = 0
    Df = np.zeros((x.size,), dtype=float)

    pi_val = pi.value[j]
    alpha = pi_val * x
    beta = (1 - pi_val) * x
    ffunc = ffunc + data.R * np.sum(gammaln(x) - gammaln(alpha) - gammaln(beta))
    dff = data.R * np.sum(digamma3(x) - pi_val * digamma3(alpha) - (1 - pi_val) * digamma3(beta))
    df = np.zeros((zeta.N,), dtype=float)

    # loop over replicate
    # LOOP TO PARALLELIZE

    val_A = data.valueA[j]
    val_B = data.valueB[j]
    val_T = data.total[j]

    results = []
    queues = [Queue() for i in range(data.R)]
    jobs   = [Process(target=tau_gamma_calculations, args=(val_A[r], val_B[r], val_T[r], alpha, beta, x, pi_val, queues[r])) for r in range(data.R)]

    for job in jobs: job.start()
    for q in queues: results.append(q.get())
    for job in jobs: job.join()

    for r in range(data.R):
        this_result = results[r]
        func = func + this_result[0]
        df   = df + this_result[1]

    Df[0] = -1. * (np.sum(zeta.estim[:,1] * df) + zetaestim * dff)
    F = -1. * (np.sum(zeta.estim[:,1] * func) + zetaestim * ffunc)

    return F, Df

def tau_gamma_calculations_hess(val_A, val_B, val_T, alpha, beta, x, pi_val, queue):
    data_alpha = val_A + alpha
    data_beta  = val_B + beta
    data_x     = val_T + x

    new_func = np.sum(gammaln(data_alpha),1) \
             + np.sum(gammaln(data_beta),1) \
             - np.sum(gammaln(data_x),1)

    dg_data_x     = digamma3(data_x)
    dg_data_alpha = digamma3(data_alpha)
    dg_data_beta  = digamma3(data_beta)

    new_df = np.sum(pi_val*dg_data_alpha,1) \
           + np.sum((1-pi_val)*dg_data_beta,1) \
           - np.sum(dg_data_x,1)

    new_hf = np.sum(pi_val*pi_val * polygamma2(1,data_alpha, dg_data_alpha),1) \
           + np.sum((1 - pi_val)*(1 - pi_val) * polygamma2(1,data_beta, dg_data_beta),1) \
           - np.sum(polygamma2(1,data_x, dg_data_x),1)

    queue.put((new_func, new_df, new_hf))

def tau_function_gradient_hessian(x, args, J):
    """Computes part of the likelihood function that has
    terms containing `tau`, and its gradient and hessian.
    """

    data = args['data']
    zeta = args['zeta']
    pi = args['pi']
    zetaestim = args['zetaestim']
    j = args['j']

    func = np.zeros((zeta.N,), dtype=float)
    ffunc = 0
    Df = np.zeros((x.size,), dtype=float)
    hess = np.zeros((x.size,), dtype=float)
    # loop over each scale

    pi_val = pi.value[j]
    alpha = pi_val * x
    beta = (1 - pi_val) * x
    ffunc = ffunc + data.R * np.sum(gammaln(x) - gammaln(alpha) - gammaln(beta))
    dg_x = digamma3(x)
    dg_alpha = digamma3(alpha)
    dg_beta = digamma3(beta)
    dff = data.R * np.sum(dg_x - pi_val * dg_alpha - (1 - pi_val) * dg_beta)
    hff = data.R * np.sum(polygamma2(1, x, dg_x) - pi_val**2 * polygamma2(1, alpha, dg_alpha) \
        - (1-pi_val)**2 * polygamma2(1, beta, dg_beta))
    df = np.zeros((zeta.N,), dtype=float)
    hf = np.zeros((zeta.N,), dtype=float)

    val_A = data.valueA[j]
    val_B = data.valueB[j]
    val_T = data.total[j]

    results = []
    queues = [Queue() for i in range(data.R)]
    jobs   = [Process(target=tau_gamma_calculations_hess, args=(val_A[r], val_B[r], val_T[r], alpha, beta, x, pi_val, queues[r])) for r in range(data.R)]

    for job in jobs: job.start()
    for q in queues: results.append(q.get())
    for job in jobs: job.join()

    for r in range(data.R):
        this_result = results[r]
        func = func + this_result[0]
        df   = df + this_result[1]
        hf   = hf + this_result[2]
    # loop over replicates
    # LOOP TO PARALLELIZE

    Df[0]   = -1 * (np.sum(zeta.estim[:,1] * df) + zetaestim * dff)
    hess[0] = -1 * (np.sum(zeta.estim[:,1] * hf) + zetaestim * hff)
    F       = -1. * (np.sum(zeta.estim[:,1] * func) + zetaestim * ffunc)
    Hf      = np.diag(hess)

    return F, Df, Hf

class Alpha():
    """
    Class to store and update (M-step) the parameter `alpha` in negative
    binomial part of the msCentipede model. There is a separate parameter
    for bound and unbound states, for each replicate.

    Arguments
        R : int
        number of replicate measurements

    """

    def __init__(self, R):

        self.R = R
        self.estim = np.random.rand(self.R,2)*10

    def __reduce__(self):
        return (rebuild_Alpha, (self.R,self.estim))

    def update(self, zeta, omega):
        """Update the estimates of parameter `alpha` in the model.
        """

        zetaestim = np.sum(zeta.estim,0)
        constant = zetaestim*nplog(omega.estim)

        # initialize optimization variables
        xo = self.estim.ravel()

        # set constraints for optimization variables
        G = np.diag(-1 * np.ones((2*self.R,), dtype=float))
        h = np.zeros((2*self.R,1), dtype=float)

        args = dict([('G',G),('h',h),('omega',omega),('zeta',zeta),('constant',constant),('zetaestim',zetaestim)])

        # call optimizer
        x_final = optimizer(xo, alpha_function_gradient, alpha_function_gradient_hessian, args, 1)
        self.estim = x_final.reshape(self.R,2)

        if np.isnan(self.estim).any():
            print("Nan in Alpha")
            raise ValueError

        if np.isinf(self.estim).any():
            print("Inf in Alpha")
            raise ValueError

def rebuild_Alpha(R, estim):

    alpha = Alpha(R)
    alpha.estim = estim
    return alpha

def alpha_function_gradient(x, args, J):
    """Computes part of the likelihood function that has
    terms containing `alpha`, and its gradient
    """

    zeta = args['zeta']
    omega = args['omega']
    constant = args['constant']
    zetaestim = args['zetaestim']

    func = 0
    df = np.zeros((2*omega.R,), dtype='float')

    # LOOP TO PARALLELIZE
    for r in range(omega.R):
        xzeta = zeta.total[:,r:r+1] + x[2*r:2*r+2]
        func = func + np.sum(np.sum(gammaln(xzeta) * zeta.estim, 0) \
                    - gammaln(x[2*r:2*r+2]) * zetaestim + constant[r] * x[2*r:2*r+2])
        df[2*r:2*r+2] = np.sum(digamma3(xzeta) * zeta.estim, 0) \
            - digamma3(x[2*r:2*r+2]) * zetaestim + constant[r]

    f  = -1.*func
    Df = -1. * df

    return f, Df

def alpha_function_gradient_hessian(x, args, J):
    """Computes part of the likelihood function that has
    terms containing `alpha`, and its gradient and hessian
    """

    zeta = args['zeta']
    omega = args['omega']
    zetaestim = args['zetaestim']
    constant = args['constant']

    func = 0
    df = np.zeros((2*omega.R,), dtype='float')
    hess = np.zeros((2*omega.R,), dtype='float')

    # LOOP TO PARALLELIZE
    for r in range(omega.R):
        xzeta = zeta.total[:,r:r+1] + x[2*r:2*r+2]
        func = func + np.sum(np.sum(gammaln(xzeta) * zeta.estim, 0) \
            - gammaln(x[2*r:2*r+2]) * zetaestim + constant[r] * x[2*r:2*r+2])

        dg_xzeta = digamma3(xzeta)
        dg_weird = digamma3(x[2*r:2*r+2])

        df[2*r:2*r+2] = np.sum(dg_xzeta * zeta.estim, 0) \
            - dg_weird * zetaestim + constant[r]

        hess[2*r:2*r+2] = np.sum(polygamma2(1, xzeta, dg_xzeta) * zeta.estim, 0) \
            - polygamma2(1, x[2*r:2*r+2], dg_weird) * zetaestim

    f  = -1. * func
    Df = -1. * df
    Hf = -1. * np.diag(hess)

    return f, Df, Hf

class Omega():
    """
    Class to store and update (M-step) the parameter `omega` in negative
    binomial part of the msCentipede model. There is a separate parameter
    for bound and unbound states, for each replicate.

    Arguments
        R : int
        number of replicate measurements

    """

    def __init__(self, R):

        self.R = R
        self.estim = np.random.rand(self.R,2)
        self.estim[:,1] = self.estim[:,1]/100

    def __reduce__(self):
        return (rebuild_Omega, (self.R,self.estim))

    def update(self, zeta, alpha):
        """Update the estimates of parameter `omega` in the model.
        """

        numerator = np.sum(zeta.estim,0) * alpha.estim
        denominator = np.array([np.sum(zeta.estim * (estim + zeta.total[:,r:r+1]), 0) \
            for r,estim in enumerate(alpha.estim)])
        self.estim = numerator / denominator

        if np.isnan(self.estim).any():
            print("Nan in Omega")
            raise ValueError

        if np.isinf(self.estim).any():
            print("Inf in Omega")
            raise ValueError

def rebuild_Omega(R, estim):

    omega = Omega(R)
    omega.estim = estim
    return omega

class Beta():
    """
    Class to store and update (M-step) the parameter `beta` in the logistic
    function in the prior of the msCentipede model.

    Arguments
        scores : array
        an array of scores for each motif instance. these could include
        PWM score, conservation score, a measure of various histone
        modifications, outputs from other algorithms, etc.

    """

    def __init__(self, S):

        self.S = S
        self.estim = np.random.rand(self.S)

    def __reduce__(self):
        return (rebuild_Beta, (self.S,self.estim))

    def update(self, scores, zeta):
        """Update the estimates of parameter `beta` in the model.
        """

        xo = self.estim.copy()
        args = dict([('scores',scores),('zeta',zeta)])

        try:
            self.estim = optimizer(xo, beta_function_gradient, beta_function_gradient_hessian, args, 1)
        except (ValueError, OverflowError):
            pass

        if np.isnan(self.estim).any():
            print("Nan in Beta")
            raise ValueError

        if np.isinf(self.estim).any():
            print("Inf in Beta")
            raise ValueError

def rebuild_Beta(S, estim):

    beta = Beta(S)
    beta.estim = estim
    return beta

def beta_function_gradient(x, args, J):
    """Computes part of the likelihood function that has
    terms containing `beta`, and its gradient.
    """

    scores = args['scores']
    zeta = args['zeta']

    arg = insum(x * scores,[1])

    func = arg * zeta.estim[:,1:] - nplog(1 + np.exp(arg))
    f = -1. * func.sum()

    Df = -1 * np.sum(scores * (zeta.estim[:,1:] - logistic(-arg)),0)

    return f, Df

def beta_function_gradient_hessian(x, args, J):
    """Computes part of the likelihood function that has
    terms containing `beta`, and its gradient and hessian.
    """

    scores = args['scores']
    zeta = args['zeta']

    arg = insum(x * scores,[1])

    func = arg * zeta.estim[:,1:] - nplog(1 + np.exp(arg))
    f = -1. * func.sum()

    Df = -1 * np.sum(scores * (zeta.estim[:,1:] - logistic(-arg)),0)

    larg = scores * logistic(arg) * logistic(-arg)
    Hf = np.dot(scores.T, larg)

    return f, Df, Hf

def optimizer(xo, function_gradient, function_gradient_hessian, args, J):
    """Calls the appropriate nonlinear convex optimization solver
    in the package `cvxopt` to find optimal values for the relevant
    parameters, given subroutines that evaluate a function,
    its gradient, and hessian, this subroutine

    Arguments
        function : function object
        evaluates the function at the specified parameter values

        gradient : function object
        evaluates the gradient of the function

        hessian : function object
        evaluates the hessian of the function

    """
    # @jit
    def F(x=None, z=None):
        """A subroutine that the cvxopt package can call to get
        values of the function, gradient and hessian during
        optimization.
        """
        if x is None:
            return 0, cvx.matrix(x_init)

        xx = np.array(x).ravel().astype(np.float64)

        if z is None:

            # compute likelihood function and gradient
            f, Df = function_gradient(xx, args, J)

            # check for infs and nans in function and gradient
            if np.isnan(f) or np.isinf(f):
                f = np.array([np.finfo('float32').max]).astype('float')
            else:
                f = np.array([f]).astype('float')
            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * np.finfo('float32').max * np.ones((1,xx.size), dtype=float)
            else:
                Df = Df.reshape(1,xx.size)

            return cvx.matrix(f), cvx.matrix(Df)

        else:

            # compute likelihood function, gradient, and hessian
            f, Df, hess = function_gradient_hessian(xx, args, J)
            # check for infs and nans in function and gradient
            if np.isnan(f) or np.isinf(f):
                f = np.array([np.finfo('float32').max]).astype('float')
            else:
                f = np.array([f]).astype('float')
            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * np.finfo('float32').max * np.ones((1,xx.size), dtype=float)
            else:
                Df = Df.reshape(1,xx.size)
            Hf = z[0] * hess
            return cvx.matrix(f), cvx.matrix(Df), cvx.matrix(Hf)

    # warm start for the optimization
    V = xo.size
    x_init = xo.reshape(V,1)

    print("Calling CVXOPT Optimizer")
    # call the optimization subroutine in cvxopt
    if 'G' in args:
        # call a constrained nonlinear solver
        solution = solvers.cp(F, G=cvx.matrix(args['G']), h=cvx.matrix(args['h']))
    else:
        # call an unconstrained nonlinear solver
        solution = solvers.cp(F)

    x_final = np.array(solution['x']).ravel()

    return x_final

def compute_footprint_likelihood(data, pi, tau, pi_null, tau_null, model):
    """Evaluates the likelihood function for the
    footprint part of the bound model and background model.

    Arguments
        data : Data
        transformed read count data

        pi : Pi
        estimate of mean footprint parameters at bound sites

        tau : Tau
        estimate of footprint heterogeneity at bound sites

        pi_null : Pi
        estimate of mean cleavage pattern at unbound sites

        tau_null : Tau or None
        estimate of cleavage heterogeneity at unbound sites

        model : string
        {msCentipede, msCentipede-flexbgmean, msCentipede-flexbg}

    """

    lhood_bound = Data()
    lhood_unbound = Data()

    for j in range(data.J):
        valueA = np.sum(data.valueA[j],0)
        valueB = np.sum(data.valueB[j],0)

        lhood_bound.valueA[j] = np.sum([gammaln(data.valueA[j][r] + pi.value[j] * tau.estim[j]) \
            + gammaln(data.valueB[j][r] + (1 - pi.value[j]) * tau.estim[j]) \
            - gammaln(data.total[j][r] + tau.estim[j]) + gammaln(tau.estim[j]) \
            - gammaln(pi.value[j] * tau.estim[j]) - gammaln((1 - pi.value[j]) * tau.estim[j]) \
            for r in range(data.R)],0)

        if model in ['msCentipede','msCentipede_flexbgmean']:

            lhood_unbound.valueA[j] = valueA * nplog(pi_null.value[j]) \
                + valueB * nplog(1 - pi_null.value[j])

        elif model=='msCentipede_flexbg':

            lhood_unbound.valueA[j] = np.sum([gammaln(data.valueA[j][r] + pi_null.value[j] * tau_null.estim[j]) \
                + gammaln(data.valueB[j][r] + (1 - pi_null.value[j]) * tau_null.estim[j]) \
                - gammaln(data.total[j][r] + tau_null.estim[j]) + gammaln(tau_null.estim[j]) \
                - gammaln(pi_null.value[j] * tau_null.estim[j]) - gammaln((1 - pi_null.value[j]) * tau_null.estim[j]) \
                for r in range(data.R)],0)

    return lhood_bound, lhood_unbound

def likelihood(data, scores, \
    zeta, pi, tau, alpha, beta, \
    omega, pi_null, tau_null, model):
    """Evaluates the likelihood function of the full
    model, given estimates of model parameters.

    Arguments
        data : Data
        transformed read count data

        scores : array
        an array of scores for each motif instance. these could include
        PWM score, conservation score, a measure of various histone
        modifications, outputs from other algorithms, etc.

        zeta : zeta
        expected value of factor binding state for each site.

        pi : Pi
        estimate of mean footprint parameters at bound sites

        tau : Tau
        estimate of footprint heterogeneity at bound sites

        alpha : Alpha
        estimate of negative binomial parameters for each replicate

        beta : Beta
        weights for various scores in the logistic function

        omega : Omega
        estimate of negative binomial parameters for each replicate

        pi_null : Pi
        estimate of mean cleavage pattern at unbound sites

        tau_null : Tau or None
        estimate of cleavage heterogeneity at unbound sites

        model : string
        {msCentipede, msCentipede-flexbgmean, msCentipede-flexbg}

    """

    apriori = insum(beta.estim * scores,[1])

    lhoodA, lhoodB = compute_footprint_likelihood(data, pi, tau, pi_null, tau_null, model)

    footprint = np.zeros((data.N,1),dtype=float)
    for j in range(data.J):
        footprint += insum(lhoodA.valueA[j],[1])

    P_1 = footprint + insum(gammaln(zeta.total + alpha.estim[:,1]) - gammaln(alpha.estim[:,1]) \
        + alpha.estim[:,1] * nplog(omega.estim[:,1]) + zeta.total * nplog(1 - omega.estim[:,1]), [1])
    P_1[P_1==np.inf] = MAX
    P_1[P_1==-np.inf] = -MAX

    null = np.zeros((data.N,1), dtype=float)
    for j in range(data.J):
        null += insum(lhoodB.valueA[j],[1])

    P_0 = null + insum(gammaln(zeta.total + alpha.estim[:,0]) - gammaln(alpha.estim[:,0]) \
        + alpha.estim[:,0] * nplog(omega.estim[:,0]) + zeta.total * nplog(1 - omega.estim[:,0]), [1])
    P_0[P_0==np.inf] = MAX
    P_0[P_0==-np.inf] = -MAX

    LL = P_0 * zeta.estim[:,:1] + P_1 * zeta.estim[:,1:] + apriori * (1 - zeta.estim[:,:1]) \
        - nplog(1 + np.exp(apriori)) - insum(zeta.estim * nplog(zeta.estim),[1])

    L = LL.sum() / data.N

    if np.isnan(L):
        print("Nan in LogLike")
        return -np.inf

    if np.isinf(L):
        print("Inf in LogLike")
        return -np.inf

    return L

def EM(data, scores, \
    zeta, pi, tau, alpha, beta, \
    omega, pi_null, tau_null, model):
    """This subroutine updates all model parameters once and computes an
    estimate of the posterior probability of binding.

    Arguments
        data : Data
        transformed read count data

        scores : array
        an array of scores for each motif instance. these could include
        PWM score, conservation score, a measure of various histone
        modifications, outputs from other algorithms, etc.

        zeta : zeta
        expected value of factor binding state for each site.

        pi : Pi
        estimate of mean footprint parameters at bound sites

        tau : Tau
        estimate of footprint heterogeneity at bound sites

        alpha : Alpha
        estimate of negative binomial parameters for each replicate

        beta : Beta
        weights for various scores in the logistic function

        omega : Omega
        estimate of negative binomial parameters for each replicate

        pi_null : Pi
        estimate of mean cleavage pattern at unbound sites

        tau_null : Tau or None
        estimate of cleavage heterogeneity at unbound sites

        model : string
        {msCentipede, msCentipede-flexbgmean, msCentipede-flexbg}

    """

    # update binding posteriors
    zeta.update(data, scores, pi, tau, \
            alpha, beta, omega, pi_null, tau_null, model)

    # update multi-scale parameters
    starttime = time.time()
    pi.update(data, zeta, tau)
    print("p_jk update in %.3f secs"%(time.time()-starttime))

    starttime = time.time()
    tau.update(data, zeta, pi)
    print("tau update in %.3f secs"%(time.time()-starttime))

    # update negative binomial parameters
    #starttime = time.time()
    omega.update(zeta, alpha)
    #print "omega update in %.3f secs"%(time.time()-starttime)

    #starttime = time.time()
    alpha.update(zeta, omega)
    #print "alpha update in %.3f secs"%(time.time()-starttime)

    # update prior parameters
    #starttime = time.time()
    beta.update(scores, zeta)
    #print "beta update in %.3f secs"%(time.time()-starttime)

def square_EM(data, scores, zeta, pi, tau, alpha, beta, omega, pi_null, tau_null, model):
    """Accelerated update of model parameters and posterior probability of binding.

    Arguments
        data : Data
        transformed read count data

        scores : array
        an array of scores for each motif instance. these could include
        PWM score, conservation score, a measure of various histone
        modifications, outputs from other algorithms, etc.

        zeta : zeta
        expected value of factor binding state for each site.

        pi : Pi
        estimate of mean footprint parameters at bound sites

        tau : Tau
        estimate of footprint heterogeneity at bound sites

        alpha : Alpha
        estimate of negative binomial parameters for each replicate

        beta : Beta
        weights for various scores in the logistic function

        omega : Omega
        estimate of negative binomial parameters for each replicate

        pi_null : Pi
        estimate of mean cleavage pattern at unbound sites

        tau_null : Tau or None
        estimate of cleavage heterogeneity at unbound sites

        model : string
        {msCentipede, msCentipede-flexbgmean, msCentipede-flexbg}

    """

    parameters = [pi, tau, alpha, omega]
    oldvar = []
    for parameter in parameters:
        try:
            oldvar.append(parameter.estim.copy())
        except AttributeError:
            oldvar.append(np.hstack([parameter.value[j].copy() for j in range(parameter.J)]))
    oldvars = [oldvar]

    # take two update steps
    for step in [0,1]:
        EM(data, scores, zeta, pi, tau, alpha, beta, omega, pi_null, tau_null, model)
        oldvar = []
        for parameter in parameters:
            try:
                oldvar.append(parameter.estim.copy())
            except AttributeError:
                oldvar.append(np.hstack([parameter.value[j].copy() for j in range(parameter.J)]))
        oldvars.append(oldvar)

    R = [oldvars[1][j]-oldvars[0][j] for j in range(len(parameters))]
    V = [oldvars[2][j]-oldvars[1][j]-R[j] for j in range(len(parameters))]
    a = -1.*np.sqrt(np.sum([(r*r).sum() for r in R]) / np.sum([(v*v).sum() for v in V]))

    if a>-1:
        a = -1.

    # given two update steps, compute an optimal step that achieves
    # a better likelihood than the two steps.
    a_ok = False
    while not a_ok:
        invalid = np.zeros((0,), dtype='bool')
        for parameter,varA,varB,varC in zip(parameters,oldvars[0],oldvars[1],oldvars[2]):
            try:
                parameter.estim = (1+a)**2*varA - 2*a*(1+a)*varB + a**2*varC
                # ensure constraints on variables are satisfied
                invalid = np.hstack((invalid,(parameter.estim<=0).ravel()))
            except AttributeError:
                newparam = (1+a)**2*varA - 2*a*(1+a)*varB + a**2*varC
                # ensure constraints on variables are satisfied
                invalid = np.hstack((invalid, np.logical_or(newparam<0, newparam>1)))
                parameter.value = dict([(j,newparam[2**j-1:2**(j+1)-1]) \
                    for j in range(parameter.J)])
        if np.any(invalid):
            a = (a-1)/2.
            if np.abs(a+1)<1e-4:
                a = -1.
        else:
            a_ok = True

    EM(data, scores, zeta, pi, tau, alpha, beta, omega, pi_null, tau_null, model)

def estimate_optimal_model(reads, totalreads, scores, background, model, log_file, restarts, mintol):
    """Learn the model parameters by running an EM algorithm till convergence.
    Return the optimal parameter estimates from a number of EM results starting
    from random restarts.

    Arguments
        reads : array
        array of read counts at each base in a genomic window,
        across motif instances and several measurement replicates.

        totalreads : array
        array of total read counts in a genomic window,
        across motif instances and several measurement replicates.
        the size of the genomic window can be different for
        `reads` and `totalreads`.

        scores : array
        an array of scores for each motif instance. these could include
        PWM score, conservation score, a measure of various histone
        modifications, outputs from other algorithms, etc.

        background : array
        a uniform, normalized array for a uniform background model.
        when sequencing reads from genomic DNA are available, this
        is an array of read counts at each base in a genomic window,
        across motif instances.

        model : string
        {msCentipede, msCentipede-flexbgmean, msCentipede-flexbg}

        restarts : int
        number of independent runs of model learning

        mintol : float
        convergence criterion

    """

    log = "transforming data into multiscale representation ..."
    log_handle = open(log_file,'a')
    log_handle.write(log)
    log_handle.close()
    print(log)

    # transform data into multiscale representation
    data = Data()
    data.transform_to_multiscale(reads)
    data_null = Data()
    data_null.transform_to_multiscale(background)
    del reads

    # transform matrix of PWM scores and other prior information
    scores = np.hstack((np.ones((data.N,1), dtype=float), scores))
    S = scores.shape[1]

    log_handle = open(log_file,'a')
    log_handle.write("done\n")
    log_handle.close()

    # set background model
    pi_null = Pi(data_null.J)
    for j in range(pi_null.J):
        pi_null.value[j] = np.sum(np.sum(data_null.valueA[j],0),0) / np.sum(np.sum(data_null.total[j],0),0).astype('float')

    tau_null = Tau(data_null.J)
    if model=='msCentipede_flexbg':

        log = "learning a flexible background model ..."
        log_handle = open(log_file,'a')
        log_handle.write(log)
        log_handle.close()
        print(log)

        zeta_null = Zeta(background.sum(1), data_null.N, False)
        zeta_null.estim[:,1] = 1
        zeta_null.estim[:,0] = 0

        # iterative update of background model;
        # evaluate convergence based on change in estimated
        # background overdispersion
        change = np.inf
        while change>1e-2:
            oldtau = tau_null.estim.copy()

            tau_null.update(data_null, zeta_null, pi_null)
            pi_null.update(data_null, zeta_null, tau_null)

            change = np.abs(oldtau-tau_null.estim).sum() / tau_null.J

        log_handle = open(log_file,'a')
        log_handle.write("done\n")
        log_handle.close()

    maxLoglike = -np.inf
    restart = 0
    err = 1
    while restart<restarts:

        totaltime = time.time()
        log = "starting model estimation (restart %d)"%(restart+1)
        log_handle = open(log_file,'a')
        log_handle.write(log+'\n')
        log_handle.close()
        print(log)

        # initialize multi-scale model parameters
        pi = Pi(data.J)
        tau = Tau(data.J)

        # initialize negative binomial parameters
        alpha = Alpha(data.R)
        omega = Omega(data.R)

        # initialize prior parameters
        beta = Beta(S)

        # initialize posterior over latent variables
        zeta = Zeta(totalreads, data.N, False)
        for j in range(pi.J):
            pi.value[j] = np.sum(data.valueA[j][0] * zeta.estim[:,1:],0) \
                / np.sum(data.total[j][0] * zeta.estim[:,1:],0).astype('float')
            mask = pi.value[j]>0
            pi.value[j][~mask] = pi.value[j][mask].min()
            mask = pi.value[j]<1
            pi.value[j][~mask] = pi.value[j][mask].max()
            minj = 1./min([pi.value[j].min(), (1-pi.value[j]).min()])
            if minj<2:
                minj = 2.
            tau.estim[j] = minj+10*np.random.rand()

        # initial log likelihood of the model
        Loglike = likelihood(data, scores, zeta, pi, tau, \
                alpha, beta, omega, pi_null, tau_null, model)

        log = "initial log likelihood = %.2e"%Loglike
        log_handle = open(log_file,'a')
        log_handle.write(log+'\n')
        log_handle.close()
        print(log)

        tol = np.inf
        iteration = 0

        while np.abs(tol)>mintol:

            itertime = time.time()
            square_EM(data, scores, zeta, pi, tau, \
                    alpha, beta, omega, pi_null, tau_null, model)

            newLoglike = likelihood(data, scores, zeta, pi, tau, \
                    alpha, beta, omega, pi_null, tau_null, model)

            tol = newLoglike - Loglike
            Loglike = newLoglike
            log = "iteration %d: log likelihood = %.2e, change in log likelihood = %.2e, iteration time = %.3f secs"%(iteration+1, Loglike, tol, time.time()-itertime)
            log_handle = open(log_file,'a')
            log_handle.write(log+'\n')
            log_handle.close()
            print(log)
            iteration += 1
        totaltime = (time.time()-totaltime)/60.

        # test if mean cleavage rate at bound sites is greater than at
        # unbound sites, for each replicate; avoids local optima issues.
        negbinmeans = alpha.estim * (1-omega.estim)/omega.estim
        if np.any(negbinmeans[:,0]<negbinmeans[:,1]):
            restart += 1
            # choose these parameter estimates, if the likelihood is greater.
            if Loglike>maxLoglike:
                maxLoglikeres = Loglike
                if model in ['msCentipede','msCentipede_flexbgmean']:
                    footprint_model = (pi, tau, pi_null)
                elif model=='msCentipede_flexbg':
                    footprint_model = (pi, tau, pi_null, tau_null)
                count_model = (alpha, omega)
                prior = beta

    return footprint_model, count_model, prior

def infer_binding_posterior(reads, totalreads, scores, background, footprint, negbinparams, prior, model):
    """Infer posterior probability of factor binding, given optimal model parameters.

    Arguments
        reads : array
        array of read counts at each base in a genomic window,
        across motif instances and several measurement replicates.

        totalreads : array
        array of total read counts in a genomic window,
        across motif instances and several measurement replicates.
        the size of the genomic window can be different for
        `reads` and `totalreads`.

        scores : array
        an array of scores for each motif instance. these could include
        PWM score, conservation score, a measure of various histone
        modifications, outputs from other algorithms, etc.

        background : array
        a uniform, normalized array for a uniform background model.
        when sequencing reads from genomic DNA are available, this
        is an array of read counts at each base in a genomic window,
        across motif instances.

        footprint : tuple
        (Pi, Tau) instances
        estimate of footprint model parameters

        negbinparams : tuple
        (Alpha, Omega) instances
        estimate of negative binomial model parameters

        prior : Beta
        estimate of weights in logistic function in the prior

        model : string
        {msCentipede, msCentipede-flexbgmean, msCentipede-flexbg}

    """

    data = Data()
    data.transform_to_multiscale(reads)
    data_null = Data()
    data_null.transform_to_multiscale(background)
    scores = np.hstack((np.ones((data.N,1), dtype=float), scores))
    del reads

    # negative binomial parameters
    alpha = negbinparams[0]
    omega = negbinparams[1]

    # weights in logistic function in the prior
    beta = prior

    # multiscale parameters
    pi = footprint[0]
    tau = footprint[1]

    # setting background model
    pi_null = footprint[2]
    for j in range(pi_null.J):
        pi_null.value[j] = np.sum(np.sum(data_null.valueA[j],0),0) \
            / np.sum(np.sum(data_null.total[j],0),0).astype('float')
    tau_null = None

    if model=='msCentipede_flexbg':

        tau_null = footprint[3]

        if data_null.N>1000:

            zeta_null = Zeta(background.sum(1), data_null.N, False)
            zeta_null.estim[:,1] = 1
            zeta_null.estim[:,0] = 0

            # iterative update of background model, when
            # accounting for overdispersion
            change = np.inf
            while change>1e-1:
                change = tau_null.estim.copy()

                pi_null.update(data_null, zeta_null, tau_null)

                tau_null.update(data_null, zeta_null, pi_null)

                change = np.abs(change-tau_null.estim).sum()

    zeta = Zeta(totalreads, data.N, True)

    zeta.infer(data, scores, pi, tau, alpha, beta, omega, \
        pi_null, tau_null, model)

    return zeta.posterior_log_odds, \
        zeta.prior_log_odds, zeta.footprint_log_likelihood_ratio, \
        zeta.total_log_likelihood_ratio
