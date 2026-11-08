"""Transition search ablation with context conditioning disabled."""
from . import transition_aos

LABEL = "Transition search without context conditioning"
FAMILY = "experimental ablation"
REFERENCE = transition_aos.REFERENCE
DESCRIPTION = "Control for the context-dependent selector in transition_aos."
DEFAULTS = {**transition_aos.DEFAULTS, "contextual": False}


def run(ctx, rng, **kwargs):
    return transition_aos.run(ctx, rng, **kwargs)
