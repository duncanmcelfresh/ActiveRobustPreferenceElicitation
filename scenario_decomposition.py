# solve a scenario-decomposition of the robust elicitation-recommendation problem, iteratively.
import time

import numpy as np

from gurobi_functions import create_mip_model, TIME_LIM, optimize, M, GurobiTimeLimit
from gurobipy import *

from preference_classes import Query
from static_elicitation import static_mip_optimal
from utils import get_u0


def feasibility_subproblem(
    z_vec_list,
    valid_responses,
    K,
    items,
    B_mat,
    b_vec,
    time_lim=TIME_LIM,
    problem_type="maximin",
    gamma_inconsistencies=0.0,
):
    # solve the scenario decomposition subproblem.

    # indifference response is not supported
    assert set(valid_responses) == set([-1, 1])

    assert problem_type in ["maximin", "mmr"]

    num_items = len(items)
    num_features = len(items[0].features)

    # recall: B_mat (m_const x n), and b_vec (m_const x 1)
    m_const = len(b_vec)
    assert B_mat.shape == (m_const, num_features)

    m = create_mip_model(time_lim=time_lim)
    m.params.OptimalityTol = 1e-8

    if gamma_inconsistencies > 0:
        xi_vars = m.addVars(K, lb=0.0, ub=GRB.INFINITY)
        m.addConstr(quicksum(xi_vars) <= gamma_inconsistencies)
    else:
        xi_vars = np.zeros(K)

    # objective value
    theta_var = m.addVar(
        vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta"
    )

    # decision variables for response scenario
    # s_k = s_plus - s_minus, and either s_plus or s_minus == 1
    s_plus_vars = m.addVars(K, vtype=GRB.BINARY, name="s_plus")
    s_minus_vars = m.addVars(K, vtype=GRB.BINARY, name="s_minus")

    # only one response is possible
    for k in range(K):
        m.addConstr(s_plus_vars[k] + s_minus_vars[k] == 1, name="s_const")
        m.addSOS(GRB.SOS_TYPE1, [s_plus_vars[k], s_minus_vars[k]])

    # add constraints for the utility of each item x
    # u_vars for each item
    u_vars = m.addVars(
        num_items,
        num_features,
        vtype=GRB.CONTINUOUS,
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        name="u",
    )

    # v_vars_list[i] is the list of variables to select the MMR item in response to item i
    v_var_list = [None for _ in range(num_items)]
    nu_vars_list = [None for _ in range(num_items)]

    for i_item, item in enumerate(items):

        if problem_type == "mmr":

            # for mmr only: use binary variables to select the item that maximizes regret
            # v_vars[i, j] = 1 if item j is selected to maximize regret for item i
            # for each i, y_vars[i, j] can be >0 for only one j (sos1)
            v_vars = m.addVars(num_items, vtype=GRB.BINARY)
            m.addConstr(quicksum(v_vars) == 1.0)
            m.addSOS(GRB.SOS_TYPE1, [v_vars[i] for i in range(num_items)])

            v_var_list[i_item] = v_vars

            nu_vars = m.addVars(
                num_items,
                num_features,
                vtype=GRB.CONTINUOUS,
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
            )
            nu_vars_list[i_item] = nu_vars

            # linearize the term nu_ij = v_i * u_j
            for i in range(num_items):
                for j in range(num_features):
                    m.addConstr(nu_vars[i, j] <= M * v_vars[i])
                    m.addConstr(nu_vars[i, j] >= -M * v_vars[i])
                    m.addConstr(
                        nu_vars[i, j] <= u_vars[i_item, j] + M * (1.0 - v_vars[i])
                    )
                    m.addConstr(
                        nu_vars[i, j] >= u_vars[i_item, j] - M * (1.0 - v_vars[i])
                    )

        # U^0 constraints for each u^x
        for i_row in range(m_const):
            m.addConstr(
                quicksum(
                    B_mat[i_row, i_feat] * u_vars[i_item, i_feat]
                    for i_feat in range(num_features)
                )
                >= b_vec[i_row],
                name=("U0_const_row_r%d_i%d" % (i_row, i_item)),
            )

        if problem_type == "maximin":
            m.addConstr(
                theta_var
                >= quicksum(
                    [
                        u_vars[i_item, i_feat] * item.features[i_feat]
                        for i_feat in range(num_features)
                    ]
                ),
                name=("theta_constr_i%d" % i_item),
            )
        if problem_type == "mmr":
            rhs_1 = quicksum(
                [
                    quicksum(
                        [nu_vars[i, j] * items[i].features[j] for i in range(num_items)]
                    )
                    for j in range(num_features)
                ]
            )
            rhs_2 = quicksum(
                [
                    u_vars[i_item, i_feat] * item.features[i_feat]
                    for i_feat in range(num_features)
                ]
            )
            m.addConstr(theta_var <= rhs_1 - rhs_2, name=("theta_constr_i%d" % i_item))

        # add constraints on U(z, s)
        for i_k, z_vec in enumerate(z_vec_list):
            m.addConstr(
                quicksum(
                    [
                        u_vars[i_item, i_feat] * z_vec[i_feat]
                        for i_feat in range(num_features)
                    ]
                )
                + xi_vars[i_k]
                >= -M * (1 - s_plus_vars[i_k]),
                name=("U_s_plus_k%d" % i_k),
            )
            m.addConstr(
                quicksum(
                    [
                        u_vars[i_item, i_feat] * z_vec[i_feat]
                        for i_feat in range(num_features)
                    ]
                )
                - xi_vars[i_k]
                <= M * (1 - s_minus_vars[i_k]),
                name=("U_s_minus_k%d" % i_k),
            )

    if problem_type == "maximin":
        m.setObjective(theta_var, sense=GRB.MINIMIZE)
    if problem_type == "mmr":
        m.setObjective(theta_var, sense=GRB.MAXIMIZE)

    m.update()

    # set dualreductions = 0 to distinguish between infeasible/unbounded
    # m.params.DualReductions = 0
    optimize(m)

    try:
        # get the optimal response scenario
        s_opt = [
            int(round(s_plus_vars[i_k].x - s_minus_vars[i_k].x)) for i_k in range(K)
        ]
        objval = m.objval

    except Exception as e:
        print(e)
        raise

    return s_opt, objval


def solve_scenario_decomposition(
    items,
    K,
    rs,
    valid_responses,
    u0_type,
    max_iter=10000,
    cut_1=True,
    cut_2=True,
    start_queries=None,
    fixed_queries=None,
    delta=1e-3,
    time_limit=1e10,
    logger=None,
    problem_type="maximin",
    gamma_inconsistencies=0.0,
    log_problem_size=False,
):
    """note that for minimax regret (problem_type='mmr'), the same logic is used to check UB and LB, but these are
    equal to the *negative* obective values of the MMR problems - because MMR is a minimization problem, while the maximin
    problem is a maximization.

    return values:
        queries, objval, status

        - queries : (list(preference_classes.Query)) a list of K queries
        - objval : (float) the objective value
        - status : (str) three possibilities:
        --- "no_incumbent" : no incumbent solution found, queries and objval will be None
        --- "suboptimal" : the algorithm did not finish in time, and the queries and objval are suboptimal
        --- "optimal" : the queries and objval are suboptimal
    """

    assert set(valid_responses) == {-1, 1}

    assert problem_type in ["maximin", "mmr"]

    # change the effective polarity of the problem for minimax regret
    if problem_type == "maximin":
        obj_sign = 1.0
    if problem_type == "mmr":
        obj_sign = -1.0

    # initialize with a single random response scenario
    s_init = rs.choice(valid_responses, K, replace=True)

    num_features = len(items[0].features)

    # polyhedral definition for U^0, B_mat and b_vec
    B_mat, b_vec = get_u0(u0_type, num_features)

    # initialize a random subproblem list
    subproblem_list = [tuple(s_init)]

    if logger is not None:
        logger.info("initial scenario: S0 = %s" % str(subproblem_list[0]))

    # keep track of time. only penalize time for solving the RMP
    time_remaining = time_limit

    # add scenarios incrementally
    rmp_queries = None
    UB = 9999
    LB = -9999
    SP_objval = None
    for i in range(max_iter):

        if UB - LB <= delta:
            if UB - LB < - 1e-3:
                raise Exception(f"UB < LB : UB = {UB}, LB = {LB}")
            break

        if logger is not None:
            logger.info(
                "iter %d, solving RMP with S=%s. time remaining=%f s:"
                % (i, str(subproblem_list), time_remaining)
            )

        # solve reduced problem with scenarios S

        if fixed_queries is not None:
            start_queries = None

        if (start_queries is None) and (fixed_queries is None):
            start_queries = rmp_queries

        t0 = time.time()
        if logger is not None:
            logger.info(f"solving RMP for k={K}")
        rmp_queries_new, RMP_objval, time_lim_reached, _ = static_mip_optimal(
            items,
            K,
            valid_responses,
            time_lim=time_remaining,
            cut_1=cut_1,
            cut_2=cut_2,
            start_queries=start_queries,
            fixed_queries=fixed_queries,
            subproblem_list=subproblem_list,
            problem_type=problem_type,
            gamma_inconsistencies=gamma_inconsistencies,
            raise_gurobi_time_limit=False,
            log_problem_size=log_problem_size,
            logger=logger,
            u0_type=u0_type,
            artificial_bounds=True,
        )

        time_remaining -= time.time() - t0
        if (time_remaining <= 0) or time_lim_reached:
            if i == 0:
                if logger is not None:
                    logger.info(
                        f"time limit reached while solving RMP, with {time_remaining}s remaining. no incumbent found."
                    )
                return None, None, "no_incumbent"
            else:
                if logger is not None:
                    logger.info(
                        f"time limit reached while solving RMP, with {time_remaining}s remaining. returning suboptimal"
                        "solution from prior iteration."
                    )
                return rmp_queries, SP_objval, "suboptimal"

        UB = obj_sign * RMP_objval

        rmp_queries = rmp_queries_new

        if logger is not None:
            logger.info(
                "solved RMP, iter=%d; UB=%f (RMP objval=%f)" % (i, UB, RMP_objval)
            )
            logger.info(
                "new RMP queries: %s" % str([q.to_tuple() for q in rmp_queries])
            )
            logger.info("solving feasibility subproblem, iter=%d" % i)

        t0 = time.time()
        try:
            subproblem_opt, SP_objval = feasibility_subproblem(
                [q.z for q in rmp_queries],
                valid_responses,
                K,
                items,
                B_mat,
                b_vec,
                time_lim=time_remaining,
                problem_type=problem_type,
                gamma_inconsistencies=gamma_inconsistencies,
            )
            time_limit -= time.time() - t0
        except GurobiTimeLimit:
            if logger is not None:
                logger.info(
                    f"time limit reached while solving feasibility subproblem, with {time_remaining}s remaining. "
                    "returning suboptimal solution from previous iteration"
                )
            return rmp_queries, SP_objval, "suboptimal"

        LB = obj_sign * SP_objval

        if logger is not None:
            logger.info(
                "solved feasibility subproblem, iter=%d; LB=%f (SP objval=%f)"
                % (i, LB, SP_objval)
            )
            logger.info("new subproblem: %s" % str(subproblem_opt))

        if (subproblem_opt in subproblem_list) and (UB - LB > delta):
            raise Exception(
                "SP identified a scenario that is already in S, and is not optimal. UB={}, LB={}".format(
                    UB, LB
                )
            )

        subproblem_list.append(tuple(subproblem_opt))

    return rmp_queries, SP_objval, "optimal"


def solve_warm_start_decomp(
    items,
    K,
    valid_responses,
    u0_type,
    cut_1=True,
    time_lim=TIME_LIM,
    time_lim_overall=True,
    logger=None,
    displayinterval=None,
    problem_type="maximin",
    gamma_inconsistencies=0.0,
    return_incremental_results=False,
):
    """
    incrementally build a solution using small K, up to the desired K
    smaller-K solutions are used as warm starts for larger-K solutions

    use the scenario decomposition to find the optimal solution
    """

    assert isinstance(time_lim, int) or isinstance(time_lim, float)

    rs = np.random.RandomState()

    if return_incremental_results:
        queries_list = []
        timing_list = []
        start_time = time.time()

    solve_opt = lambda k, queries_apx, time_lim: solve_scenario_decomposition(
        items,
        k,
        rs,
        valid_responses,
        u0_type,
        max_iter=10000,
        start_queries=queries_apx,
        logger=logger,
        time_limit=time_lim,
        problem_type=problem_type,
        gamma_inconsistencies=gamma_inconsistencies,
    )

    # this is the time budget; subtract from it each time we run anything...
    t_remaining = time_lim

    t0 = time.time()
    k = 1

    if logger is not None:
        logger.info("solving warm start + decomp, iter=%d" % k)

    queries_opt, objval, time_lim_reached, rec_inds = static_mip_optimal(
        items,
        k,
        valid_responses,
        cut_1=cut_1,
        cut_2=False,
        logger=logger,
        time_lim=t_remaining,
        displayinterval=displayinterval,
        problem_type=problem_type,
        gamma_inconsistencies=gamma_inconsistencies,
        u0_type=u0_type,
    )

    if time_lim_overall:
        t_remaining -= time.time() - t0

    if return_incremental_results:
        queries_list.append(queries_opt)
        timing_list.append((time.time() - start_time))

        if logger is not None:
            logger.info(
                "finished warm start + decomp, iter=%d; objval=%f" % (k, objval)
            )

    for k in range(2, K + 1):

        # complete the approximate solution by adding a random query to the end
        done = False
        while not done:
            item_pair = np.random.choice(len(items), 2, replace=False)
            done = True
            for q in queries_opt:
                if (item_pair[0] == q.item_A.id) and (item_pair[1] == q.item_B.id):
                    done = False

        q_next = Query(items[min(item_pair)], items[max(item_pair)])

        queries_apx = queries_opt + [q_next]

        if logger is not None:
            logger.info("solving warm start + decomp, iter=%d" % k)

        # use approx. queries as a warm start for the OPTIMAL run
        t0 = time.time()
        queries_opt, objval, status = solve_opt(k, queries_apx, t_remaining)
        assert status != "no_incumbent"

        if time_lim_overall:
            t_remaining -= time.time() - t0

        if return_incremental_results:
            queries_list.append(queries_opt)
            timing_list.append((time.time() - start_time))

        if logger is not None:
            logger.info(
                "finished warm start + decomp, iter=%d; objval=%f" % (k, objval)
            )

        if t_remaining <= 0:
            # if fewer than K queries were found return only the queries already identified (possibly fewer than K)
            if return_incremental_results:
                return queries_opt, objval, True, queries_list, timing_list
            else:
                return queries_opt, objval, True

    if return_incremental_results:
        return queries_opt, objval, time_lim_reached, queries_list, timing_list
    else:
        return queries_opt, objval, time_lim_reached


def evaluate_query_list_decomp(
    queries,
    items,
    valid_responses,
    u0_type,
    gamma_inconsistencies=0.0,
    problem_type="maximin",
    logger=None,
):
    """given a list of queries, calculate the objective of the robust rec-learning problem by solving the scenario decomp
       decision variables for the queries"""

    K = len(queries)

    rs = np.random.RandomState()

    _, objval, status = solve_scenario_decomposition(
        items,
        K,
        rs,
        valid_responses,
        u0_type,
        max_iter=10000,
        cut_1=False,
        cut_2=False,
        fixed_queries=queries,
        delta=1e-3,
        time_limit=1e10,
        gamma_inconsistencies=gamma_inconsistencies,
        problem_type=problem_type,
        logger=logger,
    )
    assert status == "optimal"

    return objval
