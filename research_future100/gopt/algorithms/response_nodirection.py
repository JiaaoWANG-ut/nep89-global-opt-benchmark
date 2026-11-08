"""Response search ablation with response-guided directions disabled."""
from . import response_search

LABEL = "Response search without guided directions"
FAMILY = "experimental ablation"
REFERENCE = response_search.REFERENCE
DESCRIPTION = "Control for the response-guided direction in response_search."
DEFAULTS = {**response_search.DEFAULTS, "direction_weight": 0.0}


def run(ctx, rng, **kwargs):
    return response_search.run(ctx, rng, **kwargs)
