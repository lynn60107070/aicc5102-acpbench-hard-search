"""Registry of methods the runner can call by name."""

from methods import best_of_n, direct, search, structured, vote_check

METHODS = {
    "direct": direct.solve,
    "structured": structured.solve,
    "best_of_n": best_of_n.solve,
    "search": search.solve,
    "vote_check": vote_check.solve,
}
