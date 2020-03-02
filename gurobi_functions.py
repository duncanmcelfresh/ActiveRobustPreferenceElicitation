from gurobipy import *

TIME_LIM = 18000  # 18,000s = 5 hours

# MIP constants
EPS_MIP = 1e-3
EPS_SMALL = EPS_MIP
M = 1e3


class GurobiTimeLimit(Exception):
    pass


def create_mip_model(
    time_lim=TIME_LIM,
    verbose=False,
    mipgap=None,
    mipgapabs=None,
    log_file=None,
    displayinterval=None,
):
    """Create a Gurobi MIP model."""

    m = Model("mip")
    if not verbose:
        m.params.LogToConsole = 0

    if time_lim is not None:
        m.params.timelimit = time_lim

    if mipgap is not None:
        assert mipgap >= 0
        m.params.MIPGap = mipgap

    if mipgapabs is not None:
        assert mipgapabs >= 0
        m.params.MIPGapAbs = mipgapabs

    if log_file is not None:
        m.params.LogFile = log_file

    if displayinterval is not None:
        m.params.DisplayInterval = displayinterval

    return m


def optimize(model, raise_warnings=True):
    """optimize a Gurobi model"""

    # if cfg.lp_file:
    #     model.update()
    #     model.write(cfg.lp_file)
    #     sys.exit(0)
    # elif cfg.relax:
    #     model.update()
    #     r = model.relax()
    #     r.optimize()
    #     print "lp_relax_obj_val:", r.obj_val
    #     print "lp_relax_solver_status:", r.status
    #     sys.exit(0)
    # else:
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        if raise_warnings:
            raise Warning("model is infeasible")
            m.computeIIS()
            m.write("/Users/duncan/research/model.mps")
            m.write("/Users/duncan/research/model.rlp")
            m.write("/Users/duncan/research/model.ilp")
    elif model.status == GRB.UNBOUNDED:
        if raise_warnings:
            raise Warning("model is unbounded")

    if model.status == GRB.TIME_LIMIT:
        raise GurobiTimeLimit
