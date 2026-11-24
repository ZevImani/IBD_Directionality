"""Input-data augmentations: functions Events -> Events applied before the usable cut.

Add one with:

    @register("my_aug")
    def my_aug(events, rng, some_param=1.0):
        ...
        return Events(new_vtx, new_cap)

and use it in a config:

    augment:
      apply_to: both            # both = meas and reference; meas = reference stays clean
      steps:
        - {type: my_aug, some_param: [0.5, 1.0]}   # list values are swept
"""
import numpy as np

from .data import Events

AUGMENTATIONS = {}


def register(name):
    def deco(fn):
        AUGMENTATIONS[name] = fn
        return fn
    return deco


def apply(events, steps, rng):
    """Apply a list of resolved steps [{'type': name, **params}] in order."""
    for step in steps:
        params = {k: v for k, v in step.items() if k != "type"}
        events = AUGMENTATIONS[step["type"]](events, rng, **params)
    return events


@register("smear")
def smear(events, rng, sigma_mm=0.0):
    """Isotropic Gaussian position smearing (x, y) of both vertex and capture."""
    if sigma_mm <= 0:
        return events
    out = []
    for p in (events.vtx, events.cap):
        q = p.copy()
        q[:, :2] += rng.normal(0.0, sigma_mm, size=(len(p), 2))
        out.append(q)
    return Events(*out)
