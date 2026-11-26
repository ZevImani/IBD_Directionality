"""Reconstruction-method registry.

A method turns a measurement sample (and optionally a reference sample drawn from the
same pool at theta = 0) into a direction estimate in degrees (CCW from +x).

To add one, drop a module in this package (and import it at the bottom of this file):

    from . import Method, register

    @register("my_method")
    class MyMethod(Method):
        def __init__(self, my_option=1.0, **common):
            super().__init__(**common)
            self.my_option = my_option

        def estimate(self, meas, ref, geom):
            # meas, ref: dirreco.data.Events ; geom: dirreco.data.Geometry
            return angle_deg

Then reference it from a config under `methods:` as  `label: {type: my_method, my_option: 2}`.
Template-matching methods should subclass `template.ScanMethod` and only implement `loss()`;
they then get the angle scan, sub-degree refinement and loss-landscape output for free.
"""

REGISTRY = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        cls.type_name = name
        return cls
    return deco


class Method:
    """Base class. Common options (any method): max_N, min_pitch, max_grid skip a method
    where it is too slow / not meaningful (rows are simply not produced).

    ref_mult: reference (template) size as a multiple of N (default 1 = the split-half
    design). A large multiple approximates a noiseless template. Only methods whose
    supports_ref_mult() is true accept it; the engine skips the method at grid points where
    the usable pool cannot supply N * (1 + ref_mult) events."""
    needs_ref = True

    def __init__(self, max_N=None, min_pitch=None, max_grid=None, ref_mult=1):
        self.max_N = max_N
        self.min_pitch = min_pitch
        self.max_grid = max_grid
        if int(ref_mult) != ref_mult or ref_mult < 1:
            raise ValueError(f"ref_mult must be a positive integer, got {ref_mult}")
        self.ref_mult = int(ref_mult)

    def supports_ref_mult(self):
        return False

    def applicable(self, N, geom):
        if self.max_N is not None and N > self.max_N:
            return False
        if self.min_pitch is not None and geom.pitch < self.min_pitch:
            return False
        if self.max_grid is not None and geom.grid_size > self.max_grid:
            return False
        return True

    def estimate(self, meas, ref, geom):
        raise NotImplementedError


def build(label, spec):
    """Instantiate a method from a config entry `label: {type: ..., **options}`."""
    spec = dict(spec or {})
    type_name = spec.pop("type", label)
    if type_name not in REGISTRY:
        raise KeyError(f"Unknown method type '{type_name}' (label '{label}'). "
                       f"Known: {sorted(REGISTRY)}")
    m = REGISTRY[type_name](**spec)
    if m.ref_mult != 1 and not m.supports_ref_mult():
        raise ValueError(f"'{label}': {type_name} with these options does not support ref_mult")
    return m


# Built-in methods (import registers them).
from . import centroid, cnn, emd, template  # noqa: E402,F401
