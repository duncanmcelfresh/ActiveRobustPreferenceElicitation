# heuristic methods for solving the static elicitation problem
import time

import numpy as np

from gurobi_functions import TIME_LIM, GurobiTimeLimit
from preference_classes import Query
from scenario_decomposition import solve_scenario_decomposition
from static_elicitation import static_mip_optimal


def solve_warm_start_heuristic(
    items,
    K,
    valid_responses,
    u0_type,
    cut_1=True,
    time_lim=TIME_LIM,
    time_lim_overall=True,
    problem_type="maximin",
    logger=None,
    displayinterval=None,
    return_incremental_times=False,
    gamma_inconsistencies=0.0,
):
    """
    incrementally build a solution - solving first the K=1 problem to opimality, then incrementally adding more queries
    by solving the K=1 problem
    """

    # this is the time budget; subtract from it each time we run anything...
    t_remaining = time_lim

    if return_incremental_times:
        incremental_times = []

    t0_global = time.time()

    objval_list = []

    # note: cut_2 must be false because fixed_queries are used
    solve_opt = lambda k, fixed_queries, start_rec, time_lim: static_mip_optimal(
        items,
        k,
        valid_responses,
        u0_type=u0_type,
        cut_1=cut_1,
        cut_2=False,
        time_lim=time_lim,
        logger=logger,
        fixed_queries=fixed_queries,
        start_rec=start_rec,
        problem_type=problem_type,
        displayinterval=displayinterval,
        gamma_inconsistencies=gamma_inconsistencies,
        raise_gurobi_time_limit=False,
    )

    k = 1

    if logger is not None:
        logger.info("solving K=1 problem")
    queries_opt, objval, time_lim_reached, rec_inds = static_mip_optimal(
        items,
        k,
        valid_responses,
        cut_1=cut_1,
        cut_2=True,
        logger=logger,
        time_lim=time_lim,
        problem_type=problem_type,
        displayinterval=displayinterval,
        gamma_inconsistencies=gamma_inconsistencies,
        raise_gurobi_time_limit=False,
        u0_type=u0_type,
    )
    objval_list.append(objval)
    if logger is not None:
        logger.info(
            "finished K=1 problem, objval=%f, optimal query=%s"
            % (objval, str(queries_opt[0].to_tuple()))
        )

    if return_incremental_times:
        incremental_times.append(time.time() - t0_global)

    for k in range(2, K + 1):

        queries_fixed = queries_opt

        # also set up the y vars (recommended items for each response scenario)
        # add a new response to each response scenario in rec_inds with the *same* recommendation...
        start_rec_list = {}
        for key, val in rec_inds.items():
            for response in valid_responses:
                # add one entry for each new response
                # keep the same index for the recommendation
                new_key = key + tuple([response])
                start_rec_list[new_key] = val

        if logger is not None:
            logger.info("solving K=%d problem" % k)

        t0 = time.time()
        queries_opt, objval, time_limit_reached, rec_inds = solve_opt(
            k, queries_fixed, start_rec_list, t_remaining
        )

        objval_list.append(objval)
        if return_incremental_times:
            incremental_times.append(time.time() - t0)

        if time_lim_overall:
            t_remaining -= time.time() - t0

        if logger is not None:
            logger.info(
                "finished K=%d problem, objval=%f, optimal queries=%s"
                % (k, objval, str([q.to_tuple() for q in queries_opt]))
            )

        if t_remaining <= 0:
            # if time limit reached, return all of the queries found, and the current objval
            time_limit_reached = True
            return queries_opt, objval, time_limit_reached

    assert len(queries_opt) == K

    if return_incremental_times:
        return queries_opt, objval_list, time_lim_reached, incremental_times
    else:
        return queries_opt, objval_list, time_lim_reached


def solve_warm_start_decomp_heuristic(
    items,
    K,
    valid_responses,
    u0_type,
    time_lim=TIME_LIM,
    logger=None,
    time_lim_overall=True,
    displayinterval=None,
    return_incremental_times=False,
    problem_type="maximin",
    gamma_inconsistencies=0.0,
    log_problem_size=False,
):
    """
    incrementally build a solution - solving first the K=1 problem to opimality, then incrementally adding more queries
    by solving the K=1 problem (using the scenario decomp.)
    """

    rs = np.random.RandomState()

    if return_incremental_times:
        incremental_times = []

    objval_list = []

    # note: cut_2 must be false because fixed_queries are used
    solve_opt = lambda k, fixed_queries, time_lim: solve_scenario_decomposition(
        items,
        k,
        rs,
        valid_responses,
        u0_type,
        max_iter=10000,
        cut_2=False,
        time_limit=time_lim,
        logger=logger,
        fixed_queries=fixed_queries,
        problem_type=problem_type,
        gamma_inconsistencies=gamma_inconsistencies,
        log_problem_size=log_problem_size,
    )

    # this is the time budget; subtract from it each time we run anything...
    t_remaining = time_lim
    t0_global = time.time()

    k = 1
    if logger is not None:
        logger.info("solving K=1 problem")

    t0 = time.time()
    try:
        if log_problem_size and logger is not None:
            logger.info("problem size for K=1 problem...")
        if logger is not None:
            logger.debug("writing gurobi logs")
        queries_opt, objval, time_lim_reached, rec_inds = static_mip_optimal(
            items,
            k,
            valid_responses,
            cut_2=False,
            time_lim=time_lim,
            logger=logger,
            displayinterval=displayinterval,
            problem_type=problem_type,
            gamma_inconsistencies=gamma_inconsistencies,
            raise_gurobi_time_limit=False,
            log_problem_size=log_problem_size,
            u0_type=u0_type,
        )
    except GurobiTimeLimit:
        if logger is not None:
            logger.info("K=1 did not complete within time limit. attempting to return ")
            try:
                for q in queries_opt:
                    assert isinstance(q, Query)
                assert isinstance(objval, float)
            except:
                raise Exception("K=1 problem did not find a feasible solution.")

    objval_list.append(objval)

    if return_incremental_times:
        incremental_times.append(time.time() - t0_global)

    if time_lim_overall:
        t_remaining -= time.time() - t0_global

    if logger is not None:
        logger.info(
            "finished K=1 problem in %f sec., objval=%f, optimal queries=%s"
            % ((time.time() - t0), objval, str([q.to_tuple() for q in queries_opt]))
        )

    for k in range(2, K + 1):

        queries_fixed = queries_opt

        if logger is not None:
            logger.info("solving K=%d problem" % k)

        t0 = time.time()
        queries_opt, objval, status = solve_opt(k, queries_fixed, t_remaining)
        assert status != "no_incumbent"

        if return_incremental_times:
            incremental_times.append(time.time() - t0_global)

        objval_list.append(objval)

        if time_lim_overall:
            t_remaining -= time.time() - t0

        if logger is not None:
            logger.info(
                "finished K=%d problem in %f sec., objval=%f, optimal queries=%s"
                % (
                    k,
                    (time.time() - t0),
                    objval,
                    str([q.to_tuple() for q in queries_opt]),
                )
            )

        if t_remaining <= 0:
            # if time limit reached, return all of the queries found, and the current objval
            time_limit_reached = True
            if return_incremental_times:
                return queries_opt, objval, time_limit_reached, incremental_times
            else:
                return queries_opt, objval, time_limit_reached

    if return_incremental_times:
        return queries_opt, objval_list, time_lim_reached, incremental_times
    else:
        return queries_opt, objval_list, time_lim_reached
