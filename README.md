# ActiveRobustPreferenceElicitation

Code accompanying the paper Active Preference Elicitation via Adjustable Robust Optimization (Vayanos et al.), http://www.optimization-online.org/DB_HTML/2020/02/7647.html. 

This code requires the following:
- Python 3 (with numpy and pandas)
- Gurobi and python module gurobipy (free for academic use)

## General Outline
- **preference_classes.py**: contains core classes for recommendations (Item, Agent, Query)
- **static_elicitation.py**: contains functions for static/offline elicitation, where queries are selected all-at-once
- **adaptive_elicitation.py**: contains functions for adaptive/online elicitation, where queries are selected one at a time
- **recommendation.py**: contains functions for making recommendations to agents
- **scenario_decomposition.py**: contains functions for solving the static/offline elicitation problem, using column-and-constraint generation
- **static_heuristics.py**: contains code for solving the static elicitation problem using heuristic methods
