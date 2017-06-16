from __future__ import division

import numpy as np
import cvxopt
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['abstol'] = 1e-8
cvxopt.solvers.options['reltol'] = 1e-7
cvxopt.solvers.options['feastol'] = 1e-8
import cplex


class QPSolver(object):

    def __init__(self, target, models, scale=False, normalize=False):
        self._target = target
        self._models = models
        self._nconformers = models.shape[0]
        self.initialized = False
        self.normalize = normalize

        self.scale = scale
        self._solution = None
        self.weights = None
        self.occupancies = None

    def initialize(self):

        scaling_factor = 1
        if self.scale:
            target_sum = self._target[self._target > 0].sum()
            model_sum = self._models[0].sum()
            scaling_factor = model_sum / target_sum

        # Set up the matrices and restraints
        self._quad_obj = cvxopt.matrix(0, (self._nconformers, self._nconformers), tc='d')
        self._lin_obj = cvxopt.matrix(self._nconformers * [0], tc='d')
        for i in xrange(self._nconformers):
            for j in xrange(i, self._nconformers):
                self._quad_obj[i,j] = np.inner(self._models[i], self._models[j])
                # Matrix is symmetric
                self._quad_obj[j,i] = self._quad_obj[i,j]
            self._lin_obj[i] = -scaling_factor * np.inner(self._models[i], self._target)

        # lower-equal constraints.
        # Each weight falls in the closed interval [0..1] and its sum is <= 1.
        # There are 2 * nconformers bounds + 1 constraint
        # Make a sparse matrix to represent this information.
        self._le_constraints = cvxopt.spmatrix(
                self._nconformers * [-1.0] + 2 * self._nconformers * [1.0],
                range(2 * self._nconformers) + self._nconformers * [2 * self._nconformers],
                3 * range(self._nconformers),
                )
        self._le_bounds = cvxopt.matrix(
                self._nconformers * [0.0] + (self._nconformers + 1) * [1.0], tc='d')
        self.initialized = True

    def __call__(self):

        if not self.initialized:
            self.initialize()

        self._solution = cvxopt.solvers.qp(
                self._quad_obj, self._lin_obj, 
                self._le_constraints, self._le_bounds
                )
        self.weights = np.asarray(self._solution['x']).ravel()
        if self.normalize:
            self.occupancies = self.weights / self.weights.sum()
        else:
            self.occupancies = self.weights.copy()


class MIQPSolver(object):

    """Mixed Integer Quadratic Program based on CPLEX."""

    def __init__(self, target, models, scale=False, normalize=False):
        self._target = target
        self._models = models
        self._nconformers = models.shape[0]
        self.initialized = False
        self.normalize = normalize
        self.scale = scale

    def initialize(self):
        scaling_factor = 1
        if self.scale:
            target_sum = self._target[self._target > 0].sum()
            model_sum = self._models[0].sum()
            scaling_factor = model_sum / target_sum

        self._quad_obj = np.zeros((self._nconformers, self._nconformers))
        self._lin_obj = np.zeros(self._nconformers)
        for i in xrange(self._nconformers):
            for j in xrange(i, self._nconformers):
                self._quad_obj[i,j] = np.inner(self._models[i], self._models[j])
                # Matrix is symmetric
                self._quad_obj[j,i] = self._quad_obj[i,j]
            self._lin_obj[i] = -scaling_factor * np.inner(self._models[i], self._target)

        self.initialized = True

    def __call__(self, maxfits=None, exact=False, threshold=None, threads=None):

        if not self.initialized:
            self.initialize()

        miqp = cplex.Cplex()
        miqp.set_results_stream(None)
        miqp.set_log_stream(None)
        miqp.set_warning_stream(None)
        miqp.set_error_stream(None)
        # Set number of threads to use
        if threads is not None:
            miqp.parameters.threads.set(threads)

        # Setup QP part of the MIQP
        variable_names = ['w{:d}'.format(n) for n in xrange(self._nconformers)]
        upper_bounds = self._nconformers * [1.0]

        miqp.variables.add(names=variable_names, ub=upper_bounds)
        for i in xrange(self._nconformers):
            for j in xrange(i, self._nconformers):
                miqp.objective.set_quadratic_coefficients(i, j, self._quad_obj[i,j])
            miqp.objective.set_linear(i, self._lin_obj[i])

        # Sum of weights is <= 1
        ind = range(self._nconformers)
        val = [1] * self._nconformers
        lin_expr = [cplex.SparsePair(ind=ind, val=val)]
        miqp.linear_constraints.add(
                lin_expr=lin_expr,
                rhs=[1],
                senses=["L"],
                )

        # If maxfits or threshold is specified the problem is a MIQP, else its
        # a regular QP.
        if maxfits not in (None, 0) or threshold not in (None, 0):
            integer_names = ['z{:d}'.format(n) for n in xrange(self._nconformers)]
            variable_types = self._nconformers * miqp.variables.type.binary
            miqp.variables.add(names=integer_names, types=variable_types)

            # Only count weights for which zi is 1
            for n in xrange(self._nconformers):
                w = "w" + str(n)
                z = "z" + str(n)
                miqp.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[w, z], val=[1, -1])],
                        rhs=[0],
                        senses="L",
                        )
                # Set the threshold constraint
                if threshold not in (None, 0):
                    miqp.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=[z, w], val=[threshold, -1])],
                            rhs=[0],
                            senses=["L"],
                            )
            # Set the cardinality constraint
            if maxfits not in (None, 0):
                senses = "L"
                if exact:
                    senses = "E"
                    maxfits = min(maxfits, self._nconformers)
                miqp.linear_constraints.add(
                        lin_expr=[[range(self._nconformers, 2 * self._nconformers), 
                                   self._nconformers * [1]]],
                        rhs=[maxfits],
                        senses=senses,
                        )
        miqp.solve()

        self.weights = np.asarray(miqp.solution.get_values()[:self._nconformers])
        if self.normalize:
            self.occupancies = self.weights / self.weights.sum()
        else:
            self.occupancies = self.weights.copy()

        miqp.end()
