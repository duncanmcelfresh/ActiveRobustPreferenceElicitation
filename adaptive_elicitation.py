# Implement the adaptive preference elicitation methods of "Robust Active Preference Learning", for a fixed set of items
# that constitute both the recommendation and query sets.

import numpy as np
import scipy
from gurobipy import *
from scipy import misc

from gurobi_functions import GurobiTimeLimit
from preference_classes import Query, EPS_ANSWER
from recommendation import solve_recommendation_problem
from static_elicitation import static_mip_optimal
from utils import get_generator_item, dist_to_point, find_analytic_center


def next_optimal_query_iterative(
    answered_queries, items, query_list, problem_type, gamma
):
    """
    iteratively search through all queries (except for those already answered, to find the optimal query, according to
    the problem type

    check all queries in query_list. don't check for duplicates in answered_queries
    """

    valid_responses = [-1, 1]
    assert problem_type in ["mmr", "maximin"]

    # for maximin, we want to maximize the minimum robust rec. utility. for mmr, we want to minimize the maximum regret
    if problem_type == "maximin":
        obj_sign = 1.0
    if problem_type == "mmr":
        obj_sign = -1.0

    M = 1e8
    opt_objval = -M
    next_query = None
    for q in query_list:
        answered_queries_new = answered_queries + [q]
        min_response_objval = M
        for i, r in enumerate(valid_responses):
            q.response = r
            objval, _ = solve_recommendation_problem(
                answered_queries_new, items, problem_type, gamma=gamma
            )
            if (obj_sign * objval) < min_response_objval:
                min_response_objval = obj_sign * objval

        if min_response_objval > opt_objval:
            opt_objval = min_response_objval
            next_query = q

    return (obj_sign * opt_objval), next_query


def next_optimal_query_mip(
    answered_queries,
    items,
    problem_type,
    gamma,
    time_limit=10800,
    log_problem_size=False,
    logger=None,
    u0_type="box",
):
    """
    use the static elicitation MIP to find the next optimal query to ask. the next query can be constructed using any
    of the items.
    """

    valid_responses = [-1, 1]
    assert problem_type in ["mmr", "maximin"]

    for q in answered_queries:
        assert q.item_A.id < q.item_B.id

    response_list = [q.response for q in answered_queries]

    assert set(response_list).issubset(set(valid_responses))

    scenario_list = [tuple(response_list + [r]) for r in valid_responses]

    K = len(answered_queries) + 1

    if logger is not None:
        logger.debug("calling static_mip_optimal, writing logs to file")
    queries, objval, _, _ = static_mip_optimal(
        items,
        K,
        valid_responses,
        cut_1=True,
        cut_2=False,
        fixed_queries=answered_queries,
        subproblem_list=scenario_list,
        gamma_inconsistencies=gamma,
        problem_type=problem_type,
        time_lim=time_limit,
        raise_gurobi_time_limit=False,
        log_problem_size=log_problem_size,
        logger=logger,
        u0_type=u0_type,
    )

    return objval, queries[-1]


def get_next_query_ac(answered_queries, items, gamma, u0_type):
    """return the query vector (created using the set of items), with hyperplane closest to the AC of the u-set formed
    by answered_queries"""
    num_features = len(items[0].features)
    ac = find_analytic_center(answered_queries, num_features, gamma, u0_type)

    min_dist = 9999
    query_opt = None
    for item_a, item_b in itertools.combinations(items, 2):
        if item_a.id < item_b.id:
            q = Query(item_a, item_b)
        else:
            q = Query(item_b, item_a)
        if q in answered_queries:
            continue
        z = q.z
        dist = np.dot(z, ac) / np.dot(z, z)
        if dist < min_dist:
            min_dist = dist
            query_opt = q

    return query_opt


def get_random_query(items, rs):
    """select a random query"""
    a, b = rs.choice(len(items), 2, replace=False)
    return Query(items[min(a, b)], items[max(a, b)])
