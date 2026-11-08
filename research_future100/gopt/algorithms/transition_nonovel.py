"""Transition search ablation with transition novelty reward disabled."""
from . import transition_aos

LABEL = "Transition search without novelty reward"
FAMILY = "experimental ablation"
REFERENCE = transition_aos.REFERENCE
DESCRIPTION = "Control for the directed transition reward in transition_aos."
DEFAULTS = {**transition_aos.DEFAULTS, "transition_reward": False}


def run(ctx, rng, **kwargs):
    return transition_aos.run(ctx, rng, **kwargs)
