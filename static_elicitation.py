from gurobipy import *
import numpy as np
import time

import itertools

from gurobi_functions import create_mip_model, optimize, M, TIME_LIM, GurobiTimeLimit
from preference_classes import Query, robust_utility, Item
from utils import get_u0, generate_filepath


class StaticMIPFailed(Exception):
    pass


def evaluate_query_list_exhaustive(queries, items, valid_responses):
    """given a list of queries, calculate the objective of the robust rec-learning problem by exhaustively checking
    each item and scenario"""

    K = len(queries)

    # contains the worst-case recommendation utility for each item
    objval_list = []
    r_list = list(itertools.product(valid_responses, repeat=K))

    for resp in r_list:
        # find the worst-case objective value over all responses
        for q, r in zip(queries, resp):
            q.response = r
        item_objvals = []
        for item in items:
            item_objvals.append(robust_utility(item, answered_queries=queries)[0])
        if max(item_objvals) is not None:
            objval_list.append(max(item_objvals))

    if len(item_objvals) == 0:
        raise Exception(
            "all worst-case item utilities are None. all U-sets are probably empty."
        )

    return min(objval_list)


def get_next_optimal_query(answered_queries, items, valid_responses, gamma):
    """given a set of answered queries, find the next optimal query to ask, by solving the full MIP and fixing the
    answered queries"""

    K = len(answered_queries) + 1
    if len(answered_queries) == 0:
        fixed_queries = []
        fixed_responses = None
    else:
        fixed_queries = answered_queries
        fixed_responses = [q.response for q in fixed_queries]

    query_list, _, _, _ = static_mip_optimal(
        items,
        K,
        valid_responses,
        cut_1=True,
        cut_2=False,
        fixed_queries=fixed_queries,
        fixed_responses=fixed_responses,
        gamma_inconsistencies=gamma,
    )

    for i, q in enumerate(answered_queries):
        # just a sanity check
        assert query_list[i] == q

    return query_list[K - 1]


def static_mip_optimal(
    items,
    K,
    valid_responses,
    time_lim=TIME_LIM,
    cut_1=True,
    cut_2=True,
    start_queries=None,
    fixed_queries=None,
    fixed_responses=None,
    start_rec=None,
    subproblem_list=None,
    displayinterval=None,
    gamma_inconsistencies=0.0,
    problem_type="maximin",
    raise_gurobi_time_limit=True,
    log_problem_size=False,
    logger=None,
    u0_type="box",
    artificial_bounds=False,
):
    """
    finds the robust-optimal query set, given a set of items.

    input:
    - items : a list of Item objects
    - K : the number of queries to be selected
    - start_queries : list of K queries to use as a warm start. do not need to be sorted.
    - fixed_queries : list of queries to FIX. length of this list must be <=K. these are fixed as the FIRST queries (order is arbitrary anyhow)
    - fixed_responses : list of responses for FIX, for the first n <= K queries. (alternative to using arg response_subset)
    - cut_1 : (bool) use cut restricting values of p and q (p < q)
    - cut_2 : (bool) use cut restricting order of queries (lexicographical order of (p,q) pairs)
    - valid_responses : list of ints, either [1, -1, 0] (indifference) or [1, -1] (no indifference)
    - response_subset : subset of scenarios S, where S[i] is a list of ints {-1, 0, 1}, of len K
    - logfile: if specified, write a gurobi logfile at this path
    - gamma_inconsistencies: (float). assumed upper bound of agent inconsistencies. increasing gamma increases the
        size of the uncertainty set
    - problem_type : (str). either 'maximin' or 'mmr'. if maximin, solve the maximin robust recommendation
        problem. if mmr, solve the minimax regret problem.

    output:
    - query_list : a list of Query objects
    - start_rec : dict where keys are response scenarios, values are indices of recommended item
    """

    if fixed_queries is None:
        fixed_queries = []
    assert problem_type in ["maximin", "mmr"]

    # indifference responses not supported
    assert set(valid_responses) == {-1, 1}

    # number of features for each item
    num_features = len(items[0].features)

    # polyhedral definition for U^0, B_mat and b_vec
    B_mat, b_vec = get_u0(u0_type, num_features)

    # number of items
    num_items = len(items)

    # lambda variables (dual variables for the initial uncertainty set):
    # lam_vars[r,i] is the i^th dual variable (for i = 1,...,m_const) for the r^th response scenario
    # recall: B_mat (m_const x n), and b_vec (m_const x 1)
    m_const = len(b_vec)
    assert B_mat.shape == (m_const, num_features)

    # get the logfile from the logger, if there is one
    if logger is not None:
        log_file = logger.handlers[0].baseFilename
    else:
        log_file = None

    # define the mip model
    m = create_mip_model(
        time_lim=time_lim, log_file=log_file, displayinterval=displayinterval
    )

    # the objective
    tau = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="tau")

    if problem_type == "maximin":
        m.setObjective(tau, sense=GRB.MAXIMIZE)
        if artificial_bounds:
            # artificial objective bound
            obj_bound = 1000
            m.addConstr(tau <= obj_bound, name="artificial_obj_bound")
    if problem_type == "mmr":
        m.setObjective(tau, sense=GRB.MINIMIZE)
        # artificial objective bound
        obj_bound = -1000
        m.addConstr(tau >= obj_bound, name="artificial_obj_bound")

    # all possible agent response scenarios
    if subproblem_list is None:
        # each subproblem is a single response scenario
        scenario_list = list(itertools.product(valid_responses, repeat=K))
        num_scenarios = int(np.power(len(valid_responses), K))
        assert num_scenarios == len(scenario_list)
    else:
        # each subproblem should be a single response scenario
        # assert that every response in the subset is a valid response
        for r in subproblem_list:
            assert set(r).difference(set(valid_responses)) == set([])
        scenario_list = subproblem_list

    if fixed_responses is not None:
        # assert subproblem_list is None
        # f = len(fixed_responses)
        # t = tuple(fixed_responses)
        # assert f <= K
        # r_list = list(r for r in itertools.product(valid_responses, repeat=K) if r[:f] == t)
        raise NotImplemented("not implemented")

    # define integer variables - this is the same for both MMR and maximin problem types
    p_vars, q_vars, w_vars = add_integer_variables(
        m,
        num_items,
        K,
        start_queries=start_queries,
        cut_1=cut_1,
        cut_2=cut_2,
        fixed_queries=fixed_queries,
    )

    # now add continuous variables for each response scenario
    if problem_type == "maximin":
        y_vars = {}
        alpha_vars = {}
        beta_vars = {}
        v_bar_vars = {}
        w_bar_vars = {}
        for i, r in enumerate(scenario_list):
            (
                alpha_vars[r],
                beta_vars[r],
                v_bar_vars[r],
                w_bar_vars[r],
            ) = add_r_constraints(
                m,
                tau,
                p_vars,
                q_vars,
                K,
                r,
                i,
                m_const,
                items,
                num_items,
                num_features,
                B_mat,
                b_vec,
                y_vars=y_vars,
                problem_type=problem_type,
                fixed_queries=fixed_queries,
                gamma_inconsistencies=gamma_inconsistencies,
            )

    if problem_type == "mmr":
        # store y_vars for each scenario
        y_vars = {}
        alpha_vars = {}
        beta_vars = {}
        v_bar_vars = {}
        w_bar_vars = {}
        for i, r in enumerate(scenario_list):
            for item in items:
                (
                    alpha_vars[r, item.id],
                    beta_vars[r, item.id],
                    v_bar_vars[r, item.id],
                    w_bar_vars[r, item.id],
                ) = add_r_constraints(
                    m,
                    tau,
                    p_vars,
                    q_vars,
                    K,
                    r,
                    i,
                    m_const,
                    items,
                    num_items,
                    num_features,
                    B_mat,
                    b_vec,
                    y_vars=y_vars,
                    problem_type=problem_type,
                    mmr_item=item,
                    fixed_queries=fixed_queries,
                    gamma_inconsistencies=gamma_inconsistencies,
                )

    m.update()

    if log_problem_size and logger is not None:
        logger.info(f"total variables: {m.numvars}")
        logger.info(f"total constraints: {m.numconstrs}")

    # m.params.DualReductions = 0
    try:
        optimize(m, raise_warnings=False)
    except GurobiTimeLimit:
        if raise_gurobi_time_limit:
            raise GurobiTimeLimit

    if m.status == GRB.TIME_LIMIT:
        time_limit_reached = True
    else:
        time_limit_reached = False

    if artificial_bounds and logger is not None:
        if abs(tau.x - obj_bound) <= 1e-3:
            logger.info(f"problem is likely unbounded: tau = obj_bound = {obj_bound}")
    try:
        # get the indices of the optimal queries
        p_inds = [-1 for _ in range(K)]
        q_inds = [-1 for _ in range(K)]
        for k in range(K):
            p_list = [np.round(p_vars[i, k].x) for i in range(num_items)]
            p_inds[k] = int(np.argwhere(p_list))
            q_list = [np.round(q_vars[i, k].x) for i in range(num_items)]
            q_inds[k] = int(np.argwhere(q_list))
    except:
        # if failed for some reason...

        lp_file = generate_filepath(os.getenv("HOME"), "static_milp_problem", "lp")
        m.write(lp_file)
        if logger is not None:
            logger.info(
                f"static MIP failed, model status = {m.status}, writing LP file to {lp_file}"
            )
        raise StaticMIPFailed

    # get indices of recommended items
    rec_inds = {}
    # for i_r, r in enumerate(r_list):
    #     y_list = [np.round(y_vars[i_r][i].x) for i in range(num_items)]
    #     rec_inds[r] = int(np.argwhere(y_list))

    return (
        [Query(items[p_inds[k]], items[q_inds[k]]) for k in range(K)],
        m.objVal,
        time_limit_reached,
        rec_inds,
    )


def add_integer_variables(
    model, num_items, K, start_queries=None, cut_1=True, cut_2=True, fixed_queries=[],
):
    """
    :param model:
    :param num_items:
    :param K:
    :param r_list:
    :param start_queries: list of K queries to use as a warm start. do not need to be sorted.
    :param cut_1:
    :param cut_2:
    :param fixed_queries : list of queries to FIX. length of this list must be <=K. these are fixed as the FIRST queries (order is arbitrary anyhow)
    :param start_rec: a dict with keys corresponding to response scenarios, and values corresponding to the
            index of recommended item : { r : ind, r : ind , ...}, to be used for warm starts. the response scenarios
            here correspond to the start_queries -- i.e., if start_rec is used, then start_queries *must*
            be used as well
    """
    # p and q vars : to select the k^th comparison
    # z^k = \sum_i (p^k_i - q^k_i) x^i
    # p_vars[i,k] = p^k_i
    # q_vars[i,k] = q^k_i
    # get the indices of non-fixed variables

    p_vars = model.addVars(num_items, K, vtype=GRB.BINARY, name="p")
    q_vars = model.addVars(num_items, K, vtype=GRB.BINARY, name="q")

    # # auxiliary variable
    # new_start_rec = None

    model.update()
    for _, var in p_vars.items():
        var.BranchPriority = 1

    for _, var in q_vars.items():
        var.BranchPriority = 2

    # exactly one item can be selected in p^k and q^k

    # fix queries if fixed_queries is specified
    if len(fixed_queries) > 0:
        if len(fixed_queries) > K:
            raise Exception("number of fixed queries must be <= K")

        if cut_2:
            raise Exception("cut_2 must be off in order to use fixed queries")

        # item_A.id is the index of p, and item_B.id is the index of q
        for i_q, q in enumerate(fixed_queries):
            model.addConstr(p_vars[q.item_A.id, i_q] == 1)
            model.addConstr(q_vars[q.item_B.id, i_q] == 1)

            # make sure this query isn't repeated later. ("cut off" this query, making it infeasible)
            for i_q_free in range(len(fixed_queries), K):
                model.addConstr(
                    p_vars[q.item_A.id, i_q_free] + q_vars[q.item_B.id, i_q_free] <= 1
                )

    for k in range(K):
        model.addSOS(GRB.SOS_TYPE1, [p_vars[i, k] for i in range(num_items)])
        model.addSOS(GRB.SOS_TYPE1, [q_vars[i, k] for i in range(num_items)])

        model.addConstr(
            quicksum(p_vars[i, k] for i in range(num_items)) == 1,
            name=("p_constr_k%d" % k),
        )
        model.addConstr(
            quicksum(q_vars[i, k] for i in range(num_items)) == 1,
            name=("q_constr_k%d" % k),
        )

    # add warm start queries if they're provided
    if start_queries is not None:
        if len(start_queries) != K:
            raise Exception("must provide exactly K start queries")

        # build (p,q) list, and sort s.t. p<q for each
        pq_list = [
            (min([q.item_A.id, q.item_B.id]), max([q.item_A.id, q.item_B.id]))
            for q in start_queries
        ]

        # order the queries (w is an auxiliary base-10 number indicating the order of queries)
        w_list = [(num_items + 1) * a + b for a, b in pq_list]

        # this is the correct order of comparisons... :  reversed(np.array(w_list).argsort())
        k_order = list(reversed(np.array(w_list).argsort()))
        pq_list_sorted = [pq_list[i] for i in k_order]

        # # if the start responses were added, re-order the keys (we assume they correspond to the new comparison
        # if start_rec is not None:
        #     new_start_rec = {}
        #     for key, val in start_rec.items():
        #         new_key = tuple([key[i] for i in k_order])
        #         new_start_rec[new_key] = val

        for k in range(K):
            for i in range(num_items):
                q_vars[i, k].start = 0
                p_vars[i, k].start = 0

        model.update()

        # now set the q and p values...
        for k in range(K):
            p_vars[pq_list_sorted[k][0], k].start = 1
            q_vars[pq_list_sorted[k][1], k].start = 1

        model.update()

    if cut_1:
        # cut 1: redundant comparison sequences
        for k in range(K):
            for i in range(num_items):
                model.addConstr(
                    1 - q_vars[i, k]
                    >= quicksum([p_vars[j, k] for j in range(num_items) if j >= i]),
                    name=("q_cut_k%d_i%d" % (k, i)),
                )

    # w vars : w^k represents k^th comparison
    # note: these are "y" in the paper
    # w^k = p^k + q^k
    # w_vars = model.addVars(num_items, K, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='w')
    w_vars = model.addVars(num_items, K, vtype=GRB.BINARY, name="w")
    for i in range(num_items):
        for k in range(K):
            model.addConstr(
                w_vars[i, k] == p_vars[i, k] + q_vars[i, k],
                name=("w_constr_i%d_k%d" % (i, k)),
            )
            # add warm start values for w, if provided
            if start_queries is not None:
                w_vars[i, k].start = p_vars[i, k].start + q_vars[i, k].start

    if cut_2:

        # only need to define these for k<k'
        z_vars = {}
        for k_prime in range(K):
            for k in range(K):
                if k < k_prime:
                    z_vars[k, k_prime] = model.addVars(
                        num_items, vtype=GRB.BINARY, name=("z_k%d_kp%d" % (k, k_prime))
                    )
                    for i in range(num_items):
                        # constraints to define v_vars
                        model.addConstr(
                            z_vars[k, k_prime][i] <= w_vars[i, k] + w_vars[i, k_prime],
                            name=("z_constrA_k%d_kp%d_i%d" % (k, k_prime, i)),
                        )
                        model.addConstr(
                            z_vars[k, k_prime][i]
                            <= 2 - w_vars[i, k] - w_vars[i, k_prime],
                            name=("z_constrB_k%d_kp%d_i%d" % (k, k_prime, i)),
                        )
                        model.addConstr(
                            z_vars[k, k_prime][i] >= w_vars[i, k] - w_vars[i, k_prime],
                            name=("z_constrC_k%d_kp%d_i%d" % (k, k_prime, i)),
                        )
                        model.addConstr(
                            z_vars[k, k_prime][i] >= -w_vars[i, k] + w_vars[i, k_prime],
                            name=("z_constrD_k%d_kp%d_i%d" % (k, k_prime, i)),
                        )

                        # cut 2: now enforce lex. ordering of w vectors
                        model.addConstr(
                            w_vars[i, k_prime]
                            >= w_vars[i, k]
                            - quicksum(
                                z_vars[k, k_prime][i_prime]
                                for i_prime in range(num_items)
                                if i_prime < i
                            ),
                            name=("lex_cut_k%d_kp%d_i%d" % (k, k_prime, i)),
                        )

                    # finally, ban identical queries (using the z_vars..)
                    model.addConstr(quicksum(z_vars[k, k_prime]) >= 1)

    # else:
    #     w_vars = []
    #
    # # loop over all response scenarios...
    # y_vars_list = []
    # for i_r, r in enumerate(r_list):
    #     # Note: all variables defined here are only for the current response scenario r
    #
    #     # y vars : to select x^r, the recommended item in scenario r
    #     y_vars = model.addVars(num_items, vtype=GRB.BINARY, name="y_r" + str(i_r))
    #
    #     if new_start_rec is None:
    #         new_start_rec = start_rec
    #
    #     # if start y_vars are provided
    #     if start_rec is not None:
    #         for i_n in range(num_items):
    #             y_vars[i_n].start = 0
    #         model.update()
    #         y_vars[new_start_rec[r]].start = 1
    #         model.update()
    #
    #     y_vars_list.append(y_vars)
    #     # exactly one item must be selected
    #     if use_sos:
    #         model.addSOS(GRB.SOS_TYPE1, [y_vars[i] for i in range(num_items)])
    #
    #     model.addConstr(quicksum(y_vars[i] for i in range(num_items)) == 1, name=('y_constr_r%d' % i_r))

    return p_vars, q_vars, w_vars


def add_r_constraints(
    m,
    tau,
    p_vars,
    q_vars,
    K,
    response_scenario,
    i_r,
    m_const,
    items,
    num_items,
    num_features,
    B_mat,
    b_vec,
    y_vars,
    problem_type="maximin",
    mmr_item=None,
    fixed_queries=[],
    gamma_inconsistencies=0.0,
):
    """
    add constraints for a single response scenario

    input vars:
    - m : gurobi model
    - tau : gurobi variable tau from the model
    - r : response scenario (K-length vector)
    - i_r : index of r (only used for printing and naming variables / constraints)
    - problem_type : (str). either 'maximin' or 'mmr'. if maximin, add constraints for the maximin robust recommendation
        problem. if mmr, add constraints for the minimax regret problem.
    - mmr_item: if problem_type is mmr, then create constraints where x' on the RHS of the equality constraint is mmr_item
    - y_vars: (dict) keys are response scenarios, values are arrays of binary y-variables. if y_vars[r] is not defined,
        add this to the dict
    """

    assert problem_type in ["mmr", "maximin"]

    for ri in response_scenario:
        assert ri in [-1, 1]

    if problem_type == "mmr":
        assert isinstance(mmr_item, Item)

    if problem_type == "mmr":
        id_str = f"r{i_r}_i{mmr_item.id}"
    if problem_type == "maximin":
        id_str = f"r{i_r}"

    if y_vars.get(response_scenario) is None:
        # y vars : to select x^r, the recommended item in scenario r
        y_vars[response_scenario] = m.addVars(
            num_items, vtype=GRB.BINARY, name=(f"y_{id_str}")
        )

        m.addSOS(
            GRB.SOS_TYPE1, [y_vars[response_scenario][i] for i in range(num_items)]
        )

        m.addConstr(
            quicksum(y_vars[response_scenario][i] for i in range(num_items)) == 1,
            name=f"y_constr_{id_str}",
        )

    if gamma_inconsistencies > 0:
        # dual variable for inconsistencies constraint
        if problem_type == "maximin":
            mu_var = m.addVar(
                vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=0.0, name=f"mu_{id_str}"
            )
        if problem_type == "mmr":
            mu_var = m.addVar(
                vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name=f"mu_{id_str}"
            )
    else:
        mu_var = 0

    # the dual variables have a different sign for mmr and maximin
    if problem_type == "maximin":
        dual_lb = 0.0
        dual_ub = GRB.INFINITY
    if problem_type == "mmr":
        dual_lb = -GRB.INFINITY
        dual_ub = 0.0

    beta_vars = m.addVars(
        m_const, vtype=GRB.CONTINUOUS, lb=dual_lb, ub=dual_ub, name=f"beta_r_{id_str}"
    )
    alpha_vars = m.addVars(
        K, vtype=GRB.CONTINUOUS, lb=dual_lb, ub=dual_ub, name=f"alpha_{id_str}"
    )

    # only define these variables for queries which are not fixed
    # define v_bar, w_bar vars (dual variables of the epigraph constraints, for linearization)
    # get the indices of non-fixed variables
    K_free = [k for k in range(K) if k >= len(fixed_queries)]
    K_fixed = [k for k in range(K) if k < len(fixed_queries)]

    v_bar_vars = m.addVars(
        num_items,
        K_free,
        vtype=GRB.CONTINUOUS,
        lb=dual_lb,
        ub=dual_ub,
        name=f"gamma_{id_str}",
    )
    w_bar_vars = m.addVars(
        num_items,
        K_free,
        vtype=GRB.CONTINUOUS,
        lb=dual_lb,
        ub=dual_ub,
        name=f"lambda_{id_str}",
    )

    if gamma_inconsistencies > 0:
        if problem_type == "maximin":
            for k in K_free:
                m.addConstr(
                    alpha_vars[k] + mu_var <= 0, name=f"alpha_constr_k{k}_{id_str}",
                )
        if problem_type == "mmr":
            for k in K_free:
                m.addConstr(
                    alpha_vars[k] + mu_var >= 0, name=f"alpha_constr_k{k}_{id_str}",
                )

    # constraints defining gamma and lambda - identical for mmr and maximin
    if problem_type == "maximin":
        for k in K_free:
            for i in range(num_items):
                m.addConstr(
                    v_bar_vars[i, k] <= M * p_vars[i, k],
                    name=f"p_constrA_k{k}_i{i}_{id_str}",
                )
                m.addConstr(
                    v_bar_vars[i, k] <= alpha_vars[k],
                    name=f"p_constrB_k{k}_i{i}_{id_str}",
                )
                m.addConstr(
                    v_bar_vars[i, k] >= alpha_vars[k] - M * (1 - p_vars[i, k]),
                    name=f"p_constrC_k{k}_i{i}_{id_str}",
                )

                m.addConstr(
                    w_bar_vars[i, k] <= M * q_vars[i, k],
                    name=f"q_constrA_k{k}_i{i}_{id_str}",
                )
                m.addConstr(
                    w_bar_vars[i, k] <= alpha_vars[k],
                    name=f"q_constrB_k{k}_i{i}_{id_str}",
                )
                m.addConstr(
                    w_bar_vars[i, k] >= alpha_vars[k] - M * (1 - q_vars[i, k]),
                    name=f"q_constrC_k{k}_i{i}_{id_str}",
                )
    if problem_type == "mmr":
        for k in K_free:
            for i in range(num_items):
                m.addConstr(
                    v_bar_vars[i, k] >= -M * p_vars[i, k],
                    name=f"p_constrA_k{k}_i{i}_{id_str}",
                )
                m.addConstr(
                    v_bar_vars[i, k] >= alpha_vars[k],
                    name=f"p_constrB_k{k}_i{i}_{id_str}",
                )
                m.addConstr(
                    v_bar_vars[i, k] <= alpha_vars[k] + M * (1 - p_vars[i, k]),
                    name=f"p_constrC_k{k}_i{i}_{id_str}",
                )

                m.addConstr(
                    w_bar_vars[i, k] >= -M * q_vars[i, k],
                    name=f"q_constrA_k{k}_i{i}_{id_str}",
                )
                m.addConstr(
                    w_bar_vars[i, k] >= alpha_vars[k],
                    name=f"q_constrB_k{k}_i{i}_{id_str}",
                )
                m.addConstr(
                    w_bar_vars[i, k] <= alpha_vars[k] + M * (1 - q_vars[i, k]),
                    name=f"q_constrC_k{k}_i{i}_{id_str}",
                )

    # the big equality constraint
    for f in range(num_features):
        lhs_1_fixed = 0
        if len(fixed_queries) > 0:
            lhs_1_fixed = quicksum(
                response_scenario[k]
                * alpha_vars[k]
                * (
                    fixed_queries[i_q].item_A.features[f]
                    - fixed_queries[i_q].item_B.features[f]
                )
                for i_q, k in enumerate(K_fixed)
            )
        lhs_1_free = quicksum(
            items[i].features[f]
            * quicksum(
                response_scenario[k] * (v_bar_vars[i, k] - w_bar_vars[i, k])
                for k in K_free
            )
            for i in range(num_items)
        )
        lhs_2 = quicksum(B_mat[j, f] * beta_vars[j] for j in range(m_const))

        if problem_type == "maximin":
            rhs = quicksum(
                y_vars[response_scenario][i] * items[i].features[f]
                for i in range(num_items)
            )

        if problem_type == "mmr":
            rhs = mmr_item.features[f] - quicksum(
                y_vars[response_scenario][i] * items[i].features[f]
                for i in range(num_items)
            )

        m.addConstr(lhs_1_fixed + lhs_1_free + lhs_2 == rhs, name=f"big_f{f}_{id_str}")

    # bound tau
    if problem_type == "maximin":
        m.addConstr(
            tau
            <= quicksum(b_vec[j] * beta_vars[j] for j in range(m_const))
            + gamma_inconsistencies * mu_var,
            name=f"tau_{id_str}",
        )
    if problem_type == "mmr":
        m.addConstr(
            tau
            >= quicksum(b_vec[j] * beta_vars[j] for j in range(m_const))
            + gamma_inconsistencies * mu_var,
            name=f"tau_{id_str}",
        )

    return alpha_vars, beta_vars, v_bar_vars, w_bar_vars


# def find_cluster_queries(queries, k):
#     # queries based on cluster centers of z vectors (assume k clusters)
#     queries_cluster = []
#     # cluster z vectors
#     Z = [np.array(q.z) for q in queries]
#     kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=0).fit(Z)
#
#     # find the query with
#     for center in kmeans.cluster_centers_:
#         q_ind_min = np.argmin([np.linalg.norm(z_vec - center) for z_vec in Z])
#         queries_cluster.append(queries[q_ind_min])
#
#     return queries_cluster
#
#
# def greedy_CSS_queries(queries, k):
#     # find k queries that are nearly orthogonal
#     # algorithm 1 from : Greedy Column Subset Selection: New Bounds and Distributed Algorithms
#     # goal: select columns of A matrix (each column is one Z vector
#     Z = [np.array(q.z) for q in queries]
#
#     A = np.matrix(Z).T
#
#     queries_css = []
#     for i in range(k):
#         # find column that maximizes Frobenius norm of Proj(V)*A
#         norms = np.zeros(len(queries))
#         for i_q,q in enumerate(queries):
#             if q not in queries_css:
#                 q_aug = copy.copy(queries_css)
#                 q_aug.append(q)
#                 V = np.matrix([qq.z for qq in q_aug]).T
#                 norms[i_q] = np.power(np.linalg.norm(np.matmul(proj_mat(V),A)),2)
#
#         # set the i^th query as that which maximizes the Frobenius norm ^2
#         queries_css.append(queries[np.argmax(norms)])
#
#     return queries_css
#
# def proj_mat(A):
#     # return the projection operator (onto the columns of A)
#     # USE THE PSEUDOINVERSE, TO DEAL WITH SINGULAR MATRICES
#     return np.matmul(np.matmul(A, np.linalg.pinv(np.matmul(A.T,A))),A.T)


def solve_warm_start(
    items,
    K,
    valid_responses,
    cut_1=True,
    time_lim=TIME_LIM,
    time_lim_overall=True,
    logfile=None,
    displayinterval=None,
    logger=None,
    problem_type="maximin",
    gamma_inconsistencies=0.0,
):
    """
    incrementally build a solution using small K, up to the desired K
    smaller-K solutions are used as warm starts for larger-K solutions

    items: (list(Item)).
    K: (int)
    valid_responses: (list(tuple)).
    cut_1: (bool).
    time_lim: (float)
    time_lim_overall: (bool). if true, time limit applies to overall runtime. if false, time limit is applied to each
        iteration (each K) independently.
    logfile: (str).
    displayinterval: (float)
    """

    assert isinstance(time_lim, int) or isinstance(time_lim, float)

    solve_opt = lambda k, queries_apx, time_lim, start_rec: static_mip_optimal(
        items,
        k,
        valid_responses,
        cut_1=cut_1,
        cut_2=True,
        start_queries=queries_apx,
        time_lim=time_lim,
        start_rec=start_rec,
        logfile=logfile,
        displayinterval=displayinterval,
        problem_type=problem_type,
        gamma_inconsistencies=gamma_inconsistencies,
    )

    # this is the time budget; subtract from it each time we run anything...
    t_remaining = time_lim

    t0 = time.time()
    k = 1

    if logger is not None:
        logger.info("solving warm start iteration k=%d" % k)

    queries_opt, objval, time_lim_reached, rec_inds = static_mip_optimal(
        items,
        k,
        valid_responses,
        cut_1=cut_1,
        cut_2=False,
        time_lim=time_lim,
        logfile=logfile,
        displayinterval=displayinterval,
        problem_type=problem_type,
        gamma_inconsistencies=gamma_inconsistencies,
    )

    # if we apply the time limit to the overall run, then subtract after each iteration. otherwise, never subtract.
    if time_lim_overall:
        t_remaining -= time.time() - t0

    if logger is not None:
        logger.info("finished warm start iteration k=%d; objval=%f" % (k, objval))

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

        # also set up the y vars (recommended items for each response scenario)
        # add a new response to each response scenario in rec_inds with the *same* recommendation...
        start_rec_list = {}
        for key, val in rec_inds.items():
            for response in valid_responses:
                # add one entry for each new response
                # keep the same index for the recommendation
                new_key = key + tuple([response])
                start_rec_list[new_key] = val

        # use approx. queries as a warm start for the OPTIMAL run

        if logger is not None:
            logger.info("solving warm start iteration k=%d" % k)

        t0 = time.time()
        queries_opt, objval, time_lim_reached, rec_inds = solve_opt(
            k, queries_apx, t_remaining, start_rec_list
        )

        if time_lim_overall:
            t_remaining -= time.time() - t0

        if logger is not None:
            logger.info("finished warm start iteration k=%d; objval=%f" % (k, objval))

        if t_remaining <= 0:
            # if fewer than K queries were found return only the queries already identified (possisbly fewer than K)
            return queries_opt, objval, True

    return queries_opt, objval, time_lim_reached


def evaluate_query_list_mip(
    queries, items, valid_responses, problem_type="maximin", gamma_inconsistencies=0.0
):
    """given a list of queries, calculate the objective of the robust rec-learning problem by solving the static mip"""

    K = len(queries)

    _, objval, _, _ = static_mip_optimal(
        items,
        K,
        valid_responses,
        cut_1=False,
        cut_2=False,
        fixed_queries=queries,
        problem_type=problem_type,
        gamma_inconsistencies=gamma_inconsistencies,
    )

    return objval
