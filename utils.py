import logging
import time
import os
import itertools
import numpy as np
import sys
from mosek.fusion import *


def get_logger(logfile=None):
    format = "[%(asctime)-15s] [%(filename)s:%(funcName)s] : %(message)s"
    logger = logging.getLogger("experiment_logs")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(format)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # logging.basicConfig(filename=logfile, level=logging.DEBUG, format=format)
    else:
        logging.basicConfig(level=logging.INFO, format=format)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def generate_filepath(output_dir, name, extension):
    # generate filepath, of the format <name>_YYYYMMDD_HHMMDD<extension>
    timestr = time.strftime("%Y%m%d_%H%M%S")
    output_string = (name + "_%s." + extension) % timestr
    return os.path.join(output_dir, output_string)


def get_generator_item(generator, i):
    # return the i^th item of a generator
    return next(itertools.islice(generator, i, None))


def array_to_string(x):
    if x is None:
        return str(x)
    return "[" + " ".join(["%.3e" % i for i in x]) + "]"


def generate_random_point_nsphere(n, rs=None):
    # generate a random point on the n-dimensional sphere
    if rs is None:
        rs = np.random.RandomState(0)
    x = rs.normal(size=n)
    return x / np.linalg.norm(x, ord=2)


def dist_to_point(z, p):
    # return the distance of z (a hyperplane) to the point p
    # np.linalg.norm is 2-norm, by default
    return np.abs(np.dot(p, z)) / np.linalg.norm(z)


def U0_box(num_features):
    # create the B matrix and b vector for the set u \in [0,1]^n, and ||u||_1 = 1
    # that is: U^0 = {u | B * u >= b}, in this case:
    B_mat = np.concatenate((np.eye(num_features), -np.eye(num_features)))
    b_vec = -np.ones(2 * num_features)
    return B_mat, b_vec


def U0_positive_normed(num_features):

    # create the B matrix and b vector for the box u \in [-1, 1]^n
    # that is: U^0 = {u | B * u >= b}, in this case:
    B_mat = np.concatenate(
        (
            np.eye(num_features),
            -np.eye(num_features),
            np.stack((np.repeat(1.0, num_features), np.repeat(-1.0, num_features))),
        )
    )
    b_vec = np.concatenate(
        (np.repeat(0.0, num_features), np.repeat(-1.0, num_features), [1.0], [-1.0])
    )
    return B_mat, b_vec


def get_u0(u0_type, num_features):
    """return a polyhedral definition for U^0, B_mat and b_vec"""

    assert u0_type in ["box", "positive_normed"]

    if u0_type == "box":
        B_mat, b_vec = U0_box(num_features)
    if u0_type == "positive_normed":
        B_mat, b_vec = U0_positive_normed(num_features)

    return B_mat, b_vec


def find_analytic_center(answered_queries, num_features, gamma, u0_type, verbose=False):
    """use MOSEK to find the analytic center of the uncertainty set, given a set of answered queries"""

    # validate input
    for q in answered_queries:
        assert q.response in [-1, 1]

    assert u0_type in ["box", "positive_normed"]

    with Model("Analytic Center") as m:

        if verbose:
            m.setLogHandler(sys.stdout)

        # -- u-set variables --
        u_vars = m.variable("u", num_features, Domain.unbounded())
        xi_vars = m.variable("xi", len(answered_queries), Domain.greaterThan(0.0))
        if gamma == 0.0:
            m.constraint(xi_vars, Domain.equalsTo(0.0))

        # -- slack variables --
        # l variables - only used if there are answered_queries
        if len(answered_queries) > 0:
            l_vars = m.variable("l", len(answered_queries), Domain.greaterThan(0.0))
            log_l_vars = m.variable(
                "log(l)", len(answered_queries)
            )  # , Domain.unbounded())
            for k in range(len(answered_queries)):
                mosek_log(m, log_l_vars.index(k), l_vars.index(k))

        # w, v variables - always used
        w_vars = m.variable("w", num_features, Domain.greaterThan(0.0))
        log_w_vars = m.variable("log(w)", num_features)  # , Domain.unbounded())
        v_vars = m.variable("v", num_features, Domain.greaterThan(0.0))
        log_v_vars = m.variable("log(v)", num_features)  # , Domain.unbounded())
        for i in range(num_features):
            mosek_log(m, log_v_vars.index(i), v_vars.index(i))
            mosek_log(m, log_w_vars.index(i), w_vars.index(i))

        # p_var - only used if gamma > 0
        if gamma > 0:
            p_var = m.variable("p", Domain.greaterThan(0.0))
            log_p_var = m.variable("log(p)")  # , Domain.unbounded())
            mosek_log(m, log_p_var, p_var)

        # u-constraints defined by queries
        if len(answered_queries) > 0:
            for k, q in enumerate(answered_queries):
                assert len(q.z) == num_features
                utz_expr = Expr.dot(q.z, u_vars)
                # lhs = Var.vstack([utz_expr, xi_vars.index(k), l_vars.index(k)])
                lhs = Var.vstack([xi_vars.index(k), l_vars.index(k)])
                if q.response == 1:
                    # u^T z_k + xi_k - l_k = 0 (s_k = 1)
                    m.constraint(
                        "uz_{}".format(k),
                        Expr.add(utz_expr, Expr.dot([1, -1], lhs),),
                        Domain.equalsTo(0.0),
                    )
                if q.response == -1:
                    # u^T z_k - xi_k + l_k = 0 (s_k = -1)
                    m.constraint(
                        "uz_{}".format(k),
                        Expr.add(utz_expr, Expr.dot([-1, 1], lhs),),
                        Domain.equalsTo(0.0),
                    )

        if u0_type == "box":
            # u-constraints - within bounding box [-1, 1]^(num_features)
            # m.constraint('u + w = 1',
            #              Expr.add(u_vars, w_vars), Domain.equalsTo(1.0))
            # m.constraint('u - v = -1',
            #              Expr.sub(u_vars, v_vars), Domain.equalsTo(-1.0))
            for i in range(num_features):
                m.constraint(
                    "u + w = 1 : {}".format(i),
                    Expr.add(u_vars.index(i), w_vars.index(i)),
                    Domain.equalsTo(1.0),
                )
                m.constraint(
                    "u - v = -1 : {}".format(i),
                    Expr.sub(u_vars.index(i), v_vars.index(i)),
                    Domain.equalsTo(-1.0),
                )

        if u0_type == "positive_normed":
            # fix the 1-norm of u
            m.constraint(
                "||u||_1 = 1", Expr.sum(u_vars), Domain.equalsTo(1.0),
            )
            for i in range(num_features):
                m.constraint(
                    "u + w = 1 : {}".format(i),
                    Expr.add(u_vars.index(i), w_vars.index(i)),
                    Domain.equalsTo(1.0),
                )
                m.constraint(
                    "u - v = 0 : {}".format(i),
                    Expr.sub(u_vars.index(i), v_vars.index(i)),
                    Domain.equalsTo(0.0),
                )

        # constraints bounding xi
        if gamma > 0:
            m.constraint(
                "sum(xi) + p = Gamma",
                Expr.add(Expr.sum(xi_vars), p_var),
                Domain.equalsTo(gamma),
            )

        # objective value
        if len(answered_queries) > 0:
            obj_u = Expr.add(
                [Expr.sum(log_l_vars), Expr.sum(log_w_vars), Expr.sum(log_v_vars),]
            )
        else:
            obj_u = Expr.add([Expr.sum(log_w_vars), Expr.sum(log_v_vars),])

        if gamma > 0:
            obj = Expr.add(obj_u, log_p_var)
        else:
            obj = obj_u

        m.objective(ObjectiveSense.Maximize, obj)

        # optimize
        m.solve()

        sol_status = m.getPrimalSolutionStatus()

        if sol_status != SolutionStatus.Optimal:
            raise Exception(
                "mosek model status is not optimal. status: {}".format(str(sol_status))
            )

        # return the u-vars
        return u_vars.level()


# MOSEK: Logarithm
# t <= log(x), x>=0
def mosek_log(M, t, x):
    M.constraint(Expr.hstack(x, 1, t), Domain.inPExpCone())


def random_vector_bounded_sum(k, vector_sum, rs, max_samples=100000000):
    """
    return a uniform random vector of length k which has 1-norm <= vector_sum, with each element bounded by
     * vector sum (otherwise the support of this distribution is unbounded)

    do this using rejection sampling...
    """

    for i in range(max_samples):
        vec = rs.rand(int(k)) * vector_sum
        if sum(vec) <= vector_sum:
            signs = rs.choice([-1.0, 1.0], int(k))
            return vec * signs

    return Exception("could not find a vector in this distribution")


def rand_fixed_sum(n, seed):
    """
    Roger Stafford's randfixedsum algorithm, adapted from original matlab code
    (https://www.mathworks.com/matlabcentral/fileexchange/9700-random-vectors-with-fixed-sum)

    Copyright 2010 Paul Emberson, Roger Stafford, Robert Davis.
    All rights reserved.

    each element of the vector is on [0, 1], and the sum is fixed to 1

    args:
    - n (int): length of the vector
    - seed (int): random seed
    """
    assert n > 1
    assert isinstance(n, int)

    rand_state = np.random.RandomState(seed)

    s = 1
    k = 1

    tiny = np.finfo(float).tiny
    huge = np.finfo(float).max

    s1 = s - np.arange(k, k - n, -1.0)
    s2 = np.arange(k + n, k, -1.0) - s

    w = np.zeros((n, n + 1))
    w[0, 1] = huge
    t = np.zeros((n - 1, n))

    for i in np.arange(2, n + 1):
        tmp1 = w[i - 2, np.arange(1, i + 1)] * s1[np.arange(0, i)] / float(i)
        tmp2 = w[i - 2, np.arange(0, i)] * s2[np.arange(n - i, n)] / float(i)
        w[i - 1, np.arange(1, i + 1)] = tmp1 + tmp2
        tmp3 = w[i - 1, np.arange(1, i + 1)] + tiny
        tmp4 = s2[np.arange(n - i, n)] > s1[np.arange(0, i)]
        t[i - 2, np.arange(0, i)] = (tmp2 / tmp3) * tmp4 + (1 - tmp1 / tmp3) * (
            np.logical_not(tmp4)
        )

    x = np.zeros(n)
    rt = rand_state.uniform(size=(n - 1))  # rand simplex type
    rs = rand_state.uniform(size=(n - 1))  # rand position in simplex
    j = k + 1
    tmp_sum = 0.0
    tmp_prob = 1.0

    for i in np.arange(n - 1, 0, -1):  # iterate through dimensions
        # decide which direction to move in this dimension (1 or 0):
        e = 1 if rt[(n - i) - 1] <= t[i - 1, j - 1] else 0
        sx = rs[(n - i) - 1] ** (1.0 / i)  # next simplex coord
        tmp_sum = tmp_sum + (1.0 - sx) * tmp_prob * s / (i + 1)
        tmp_prob = sx * tmp_prob
        x[(n - i) - 1] = tmp_sum + tmp_prob * e
        s = s - e
        j = j - e  # change transition table column if required

    x[n - 1] = tmp_sum + tmp_prob * s

    # iterated in fixed dimension order but needs to be randomised
    # permute x row order within each column
    x_new = x[rand_state.permutation(n)]

    return x_new
