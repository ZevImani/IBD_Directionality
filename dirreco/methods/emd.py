"""Earth mover's / optimal-transport template-matching methods on point clouds.

Every method is a ScanMethod on the continuous (unbinned) displacement clouds:
L(phi) = distance(R(phi) meas, ref), theta_hat = -argmin (refined). Each event carries
mass 1/N_ref, so the reference has unit mass. Balanced methods renormalise both clouds to
1; unbalanced ones keep the mass ratio (it is 1 while N_meas = N_ref, but unbalanced OT
still differs: transporting farther than about 2*lam costs more than destroying and
re-creating the mass, so far tails stop pulling on the fit).

| type           | distance                                     | balanced / unbalanced                  |
|----------------|----------------------------------------------|----------------------------------------|
| emd_paired     | mean |R m_i - r_i|, random pairing (Max)     | -- (upper bound on W1, not W1)         |
| emd_exact      | exact W1, POT network simplex                | unbalanced: partial (mass) | kr (lam)  |
| emd_sinkhorn   | entropic W_p / Sinkhorn divergence, geomloss | reach: null | <mm> (KL marginals)      |
| emd_sliced     | sliced W1 (mean, or max-sliced)              | balanced                               |
| emd_sliced_kr  | sliced exact 1D Kantorovich-Rubinstein       | unbalanced (lam)                       |
| mmd            | energy / gaussian / laplacian kernel (geomloss) | non-OT baselines                    |

Ported from the proton/electron inference project (inference_loop/protoninfer/losses.py,
paper_plots/compare_losses/unbalanced_sliced/unbalanced_study.py):
emd/w1/sinkhorn, uemd, w2 -> emd_sinkhorn; sw -> emd_sliced; suw1 -> emd_sliced_kr;
exact_w1 / exact_kr -> emd_exact; energy/gmmd/lmmd -> mmd.
Not ported: sw_mass (with N_meas = N_ref it equals emd_sliced exactly); POT's
sinkhorn_unbalanced2 (same objective as emd_sinkhorn + reach, but not log-domain, so it
underflows at mm-scale costs / mm-scale epsilon).
Cell-centre points do not work with this scheme: they sit on the lattice, so the loss has
a spurious minimum at phi = 0 whenever the rotated lattice realigns with itself.

discrete: true (emd_sliced, emd_exact) -- detector (cell-hop) readout, done the other way round:
the measurement is never rotated; it is the histogram of its cell hops (cell of capture - cell
of vertex) on the lattice Z^2 * pitch, exactly what centroid_disc sees. The template for each
hypothesis is forward-modelled from the reference's TRUE displacements: rotate them by
-phi and deposit each one cloud-in-cell on the hop lattice. That is the exact expected hop
distribution for a vertex cell phase uniform in the cell (as in dirreco/fisher.py), so it is
smooth in phi and has no lattice artefact. W1 is then taken between the two lattice
histograms (ground cost = distance between hop vectors, mm). Assumes the reference comes from
MC with truth, as a real analysis template would.

ref_mult: k (emd_sliced, emd_sinkhorn balanced, mmd, discrete emd_exact) -- template of k*N
events, approximating a noiseless template (the split-half design pays template noise, about
a factor sqrt(1 + 1/k) in sigma).
"""
import numpy as np

from . import register
from .template import CHUNK, ScanMethod, _chunks, _rotations


def _torch_device(device):
    import torch
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


class PointCloudScan(ScanMethod):
    def clouds(self, meas, ref, geom):
        """-> x, y (measurement) and xr, yr (reference) displacement coordinates [mm]."""
        m, r = meas.disp, ref.disp
        return m[:, 0], m[:, 1], r[:, 0], r[:, 1]


# ----------------------------------------------------------------------------- cell-hop lattice
class HopLattice:
    """Cell-hop readout on the lattice Z^2 * pitch for one (measurement, reference) pair.

    meas:  normalised histogram of the measured hops, flat index (hx + H) * S + (hy + H).
    templates(psis): expected hop histograms of the reference rotated CCW by each psi.
    sites: (S^2, 2) hop vectors [mm] in the same flat order.
    """

    def __init__(self, meas, ref, geom):
        p = float(geom.pitch)
        hop = np.rint(geom.discrete_disp(meas) / p).astype(np.int64)
        self.s = ref.disp / p                                              # [cells]
        H = int(max(np.abs(hop).max(initial=0),
                    np.ceil(np.hypot(self.s[:, 0], self.s[:, 1]).max(initial=0.0)))) + 1
        S = 2 * H + 1
        self.H, self.S, self.pitch = H, S, p
        m = np.bincount((hop[:, 0] + H) * S + (hop[:, 1] + H), minlength=S * S).astype(float)
        self.meas = m / m.sum()
        ii = (np.arange(S) - H) * p
        self.sites = np.stack(np.meshgrid(ii, ii, indexing="ij"), -1).reshape(-1, 2)

    def templates(self, psis):
        """(len(psis), S^2) cloud-in-cell hop distributions, each summing to 1. Along each axis
        a displacement of s cells hops floor(s) cells with probability 1 - frac(s) and
        floor(s) + 1 with probability frac(s) (uniform vertex phase; x and y independent)."""
        S, H, R = self.S, self.H, len(self.s)
        out = []
        for ps in _chunks(np.asarray(psis, dtype=float), 4 * R):
            rx, ry = _rotations(self.s[:, 0], self.s[:, 1], ps)          # (F, R)
            x0, y0 = np.floor(rx), np.floor(ry)
            fx, fy = rx - x0, ry - y0
            base = ((x0.astype(np.int64) + H) * S + (y0.astype(np.int64) + H)
                    + (np.arange(len(ps)) * S * S)[:, None])
            T = np.zeros(len(ps) * S * S)
            for d, w in ((0, (1 - fx) * (1 - fy)), (S, fx * (1 - fy)),
                         (1, (1 - fx) * fy), (S + 1, fx * fy)):
                T += np.bincount((base + d).ravel(), weights=w.ravel(), minlength=T.size)
            out.append(T.reshape(len(ps), S * S) / R)
        return np.concatenate(out)


def _lattice_sliced_w1(lat, T, n_proj, reduce):
    """Sliced W1 between lat.meas and each template row of T, on the fixed lattice sites:
    1D W1 = sum |CDF difference| * gap, with sites sorted along each direction."""
    a = _directions(n_proj)
    proj = np.cos(a)[:, None] * lat.sites[:, 0] + np.sin(a)[:, None] * lat.sites[:, 1]  # (L, S2)
    order = np.argsort(proj, axis=1, kind="stable")
    gaps = np.diff(np.take_along_axis(proj, order, axis=1), axis=1)                    # (L, S2-1)
    out, step = [], max(1, CHUNK // (n_proj * T.shape[1]))
    for i in range(0, len(T), step):
        d = lat.meas[None, :] - T[i:i + step]                                           # (F, S2)
        c = np.cumsum(d[:, order], axis=2)[..., :-1]                                    # (F, L, S2-1)
        w1 = (np.abs(c) * gaps[None]).sum(axis=2)
        out.append(w1.mean(axis=1) if reduce == "mean" else w1.max(axis=1))
    return np.concatenate(out)


# ----------------------------------------------------------------------------- Max's proxy
@register("emd_paired")
class EMDPaired(PointCloudScan):
    """Max's "2D spatial W1 chord proxy": mean |R(phi) m_i - r_i| with measurement and
    reference events paired by (random) index. An upper bound on W1, not W1 itself."""

    def loss(self, meas, ref, geom):
        x, y, xr, yr = self.clouds(meas, ref, geom)
        out = []
        for ph in _chunks(self.phis, len(x)):
            rx, ry = _rotations(x, y, ph)
            out.append(np.sqrt((rx - xr) ** 2 + (ry - yr) ** 2).mean(axis=1))
        return np.concatenate(out)


# ----------------------------------------------------------------------------- exact (POT)
@register("emd_exact")
class EMDExact(PointCloudScan):
    """Exact (non-entropic) W1, Euclidean ground cost [mm], via POT's network simplex.

    unbalanced: null     balanced W1 (ot.emd2)
                partial  exact partial OT moving only a fraction `mass` (ot.partial)
                kr       exact unbalanced W1 / Kantorovich-Rubinstein: mass may be
                         destroyed or created at cost `lam` [mm] per unit. Solved as a
                         balanced EMD with one dummy point per side (real<->dummy = lam,
                         dummy<->dummy = 0), as exact_kr in unbalanced_study.py.
    O(N^2) memory, roughly O(N^3) time per angle: keep N small (default max_N=2000).
    POT's non-entropic KL solver (ot.unbalanced.mm_unbalanced2) is not offered: 75 s per
    estimate at N=500 and it did not recover a 10 deg injection.
    """

    def __init__(self, unbalanced=None, mass=0.9, lam=100.0, max_N=2000, discrete=False, **kw):
        super().__init__(max_N=max_N, **kw)
        if unbalanced not in (None, "partial", "kr"):
            raise ValueError("unbalanced must be null, partial or kr")
        if discrete and unbalanced:
            raise ValueError("discrete: true is balanced only")
        self.unbalanced, self.mass, self.lam = unbalanced, float(mass), float(lam)
        self.discrete = bool(discrete)

    def supports_ref_mult(self):
        return self.discrete          # lattice histograms: cost does not grow with the template

    def loss_options(self):
        return (self.unbalanced, self.mass, self.lam, self.discrete)

    def loss(self, meas, ref, geom):
        import ot
        if self.discrete:
            # Exact W1 between lattice histograms, on the sites either one occupies.
            lat = HopLattice(meas, ref, geom)
            T = lat.templates(-self.phis)
            keep = (lat.meas > 0) | (T > 0).any(axis=0)
            X = lat.sites[keep]
            C = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
            a = lat.meas[keep]
            return np.array([ot.emd2(a, t / t.sum(), C, numItermax=10_000_000) for t in T[:, keep]])
        x, y, xr, yr = self.clouds(meas, ref, geom)
        n, nr = len(x), len(xr)
        if self.unbalanced == "kr":       # unit mass per event / N_ref, plus dummies
            a = np.append(np.ones(n), nr) / nr
            b = np.append(np.ones(nr), n) / nr
            M = np.zeros((n + 1, nr + 1))
            M[:-1, -1] = M[-1, :-1] = self.lam
        else:
            a, b = np.full(n, 1.0 / n), np.full(nr, 1.0 / nr)
        out = []
        for phi in self.phis:
            rx, ry = _rotations(x, y, np.array([phi]))
            C = np.sqrt((rx[0][:, None] - xr[None, :]) ** 2 + (ry[0][:, None] - yr[None, :]) ** 2)
            if self.unbalanced == "kr":
                M[:-1, :-1] = C
                out.append(ot.emd2(a, b, M, numItermax=10_000_000))
            elif self.unbalanced == "partial":
                out.append(ot.partial.partial_wasserstein2(a, b, C, m=self.mass, nb_dummies=1))
            else:
                out.append(ot.emd2(a, b, C, numItermax=10_000_000))
        return np.array(out)


# ----------------------------------------------------------------------------- geomloss
class _GeomLossScan(PointCloudScan):
    """Shared geomloss driver: batches several rotation angles per call on GPU."""

    def __init__(self, device=None, backend="tensorized", batch_bytes=2e9, **kw):
        super().__init__(**kw)
        self.device, self.backend, self.batch_bytes = device, backend, float(batch_bytes)
        self._loss = None  # built lazily so CUDA is never initialised before the engine forks

    def _make_loss(self):
        raise NotImplementedError

    def supports_ref_mult(self):
        return True                   # weighted clouds of any size (N x k*N cost on the GPU)

    def _weights(self, n, nr):
        """Per-event masses (meas, ref). Default: each cloud normalised to 1."""
        return 1.0 / n, 1.0 / nr

    def loss(self, meas, ref, geom):
        import torch
        dev = _torch_device(self.device)
        if self._loss is None:
            self._loss = self._make_loss()
        x, y, xr, yr = self.clouds(meas, ref, geom)
        X = torch.tensor(np.column_stack([x, y]), dtype=torch.float32, device=dev)
        Y = torch.tensor(np.column_stack([xr, yr]), dtype=torch.float32, device=dev)
        wa, wb = self._weights(len(x), len(xr))
        B = 1 if self.backend != "tensorized" else max(1, int(self.batch_bytes // (len(x) * len(xr) * 4 * 3)))
        out = []
        with torch.no_grad():
            for i in range(0, len(self.phis), B):
                t = torch.tensor(np.radians(self.phis[i:i + B]), dtype=torch.float32, device=dev)[:, None]
                c, s = torch.cos(t), torch.sin(t)
                Xr = torch.stack([c * X[:, 0] - s * X[:, 1], s * X[:, 0] + c * X[:, 1]], dim=-1)
                k = len(t)
                a = torch.full((k, len(x)), wa, dtype=torch.float32, device=dev)
                b = torch.full((k, len(xr)), wb, dtype=torch.float32, device=dev)
                out.append(self._loss(a, Xr, b, Y[None].expand(k, -1, -1).contiguous()).cpu().numpy())
        return np.concatenate(out).astype(float)


@register("emd_sinkhorn")
class EMDSinkhorn(_GeomLossScan):
    """Entropic optimal transport via geomloss SamplesLoss("sinkhorn"), GPU if available.

    p:       1 = W1 (cost |x - y|, mm); 2 = W2 (cost |x - y|^2 / 2)
    blur:    entropic length scale in mm (epsilon = blur^p); W_p is recovered as blur -> 0
    scaling: epsilon-annealing factor; geomloss runs ~one Sinkhorn step per annealing step,
             so this is the accuracy knob, not blur. Measured at N=300 against exact W1:
             the loss-curve shape error over phi was 0.43 mm at 0.9 and 0.21 mm at 0.95
             (landscape range 6 mm; identical in float64). Iterations ~ log(diam/blur)/log(1/scaling).
    debias:  true = Sinkhorn divergence (0 iff equal); false = raw entropic OT
    reach:   null = balanced; <mm> = unbalanced, KL marginal penalty rho = reach^p. Mass
             farther than ~reach is destroyed/created rather than transported. Each event
             then has mass 1/N_ref (as uemd divides by the target mass).
    backend: tensorized (N x N in GPU memory, several angles per batch) | online (pykeops)
    A100 MIG 3g.20gb, tensorized, p=1, scaling 0.7: ~6 ms/angle at N=1k, 0.15 s at 5k,
    0.6 s at 10k; scaling 0.9 needs ~3.4x more iterations.
    """

    def __init__(self, p=1, blur=1.0, scaling=0.9, debias=True, reach=None, **kw):
        super().__init__(**kw)
        self.p, self.blur, self.scaling = int(p), float(blur), float(scaling)
        self.debias, self.reach = bool(debias), (None if reach is None else float(reach))

    def loss_options(self):
        return (self.p, self.blur, self.scaling, self.debias, self.reach, self.backend)

    def supports_ref_mult(self):
        return self.reach is None     # unbalanced masses are 1/N_ref each: a k*N template breaks that

    def _weights(self, n, nr):
        return (1.0 / nr, 1.0 / nr) if self.reach is not None else (1.0 / n, 1.0 / nr)

    def _make_loss(self):
        from geomloss import SamplesLoss
        return SamplesLoss("sinkhorn", p=self.p, blur=self.blur, scaling=self.scaling,
                           debias=self.debias, reach=self.reach, backend=self.backend)


@register("mmd")
class MMD(_GeomLossScan):
    """Kernel (non-OT) baselines via geomloss: kernel = energy | gaussian | laplacian
    (blur = kernel width in mm; ignored for energy)."""

    def __init__(self, kernel="energy", blur=50.0, **kw):
        super().__init__(**kw)
        if kernel not in ("energy", "gaussian", "laplacian"):
            raise ValueError("kernel must be energy, gaussian or laplacian")
        self.kernel, self.blur = kernel, float(blur)

    def loss_options(self):
        return (self.kernel, self.blur, self.backend)

    def _make_loss(self):
        from geomloss import SamplesLoss
        return SamplesLoss(self.kernel, blur=self.blur, backend=self.backend)


# ----------------------------------------------------------------------------- sliced
def _directions(n_proj):
    return np.pi * np.arange(n_proj) / n_proj


def _meas_directions(n_proj, phis):
    """Measurement direction for every (phi, slice) pair of a sliced scan, deduplicated.

    (R(phi) m) . u_a == m . u_(a - phi), so pair (phi, a) needs the measurement sorted along
    a - phi. Many pairs share a direction (all of them fall on one 1-deg grid when the slice
    spacing 180/n_proj and the scan angles are whole degrees, e.g. n_proj = 36), and u_(d+180)
    = -u_d, whose sorted projections are those along d reversed and negated. Returns
    (a_deg, base, inv, flip): base = distinct directions in [0, 180) [deg]; pair (i, l) uses
    base[inv[i, l]], reversed and negated where flip[i, l]."""
    a_deg = 180.0 * np.arange(n_proj) / n_proj
    A = np.mod(np.round(np.mod(a_deg[None, :] - np.asarray(phis, float)[:, None], 360.0), 9), 360.0)
    flip = A >= 180.0
    base, inv = np.unique(np.where(flip, A - 180.0, A), return_inverse=True)
    return a_deg, base, inv.reshape(A.shape), flip


@register("emd_sliced")
class EMDSliced(PointCloudScan):
    """Sliced W1 over `n_proj` evenly spaced (deterministic) directions in [0, 180).
    Equal sizes: 1D W1 = mean |sorted difference|.
    reduce: mean = sliced W1 (as `sw`); max = max-sliced W1 (worst direction).
    Each distinct measurement direction is sorted once per scan (see _meas_directions). With
    n_proj dividing 180 (36, 45, 60, 90, 180) and whole-degree scan angles there are at most 180
    sorts per loss call, however many angles and slices; n_proj = 32 gets less reuse."""

    def __init__(self, n_proj=32, reduce="mean", discrete=False, device=None, **kw):
        super().__init__(**kw)
        if reduce not in ("mean", "max"):
            raise ValueError("reduce must be mean or max")
        self.n_proj, self.reduce = int(n_proj), reduce
        self.discrete, self.device = bool(discrete), device

    def supports_ref_mult(self):
        return True

    def loss_options(self):
        return (self.n_proj, self.reduce, self.discrete)

    def loss(self, meas, ref, geom):
        if self.discrete:
            lat = HopLattice(meas, ref, geom)
            return _lattice_sliced_w1(lat, lat.templates(-self.phis), self.n_proj, self.reduce)
        x, y, xr, yr = self.clouds(meas, ref, geom)
        if len(x) != len(xr):
            return self._loss_large_ref(x, y, xr, yr)
        a_deg, base, inv, flip = _meas_directions(self.n_proj, self.phis)
        a = np.radians(a_deg)
        P_ref = np.sort(np.cos(a)[:, None] * xr + np.sin(a)[:, None] * yr, axis=1)      # (L, n)
        # sorted meas along d+180 is -(sorted along d)[::-1]; |that - q| = |sorted along d - (-q[::-1])|
        Q2 = np.stack([P_ref, -P_ref[:, ::-1]])                                         # (2, L, n)
        w1 = np.empty(inv.shape)                                                        # (phi, slice)
        step = max(1, CHUNK // len(x))
        for i in range(0, len(base), step):                     # a block of distinct directions
            t = np.radians(base[i:i + step])
            S = np.sort(np.cos(t)[:, None] * x + np.sin(t)[:, None] * y, axis=1)       # (B, n)
            f, l = np.nonzero((inv >= i) & (inv < i + step))    # the pairs that use this block
            for j in range(0, len(f), step):
                fj, lj = f[j:j + step], l[j:j + step]
                w1[fj, lj] = np.abs(S[inv[fj, lj] - i] - Q2[flip[fj, lj].astype(np.intp), lj]).mean(axis=1)
        return w1.mean(axis=1) if self.reduce == "mean" else w1.max(axis=1)

    def _loss_large_ref(self, x, y, xr, yr):
        """Exact 1D W1 per direction for R = c*n reference points (torch; GPU if available).
        The monotone coupling sends sorted measurement point i to the sorted reference block
        [i*c, (i+1)*c); its cost sum_j |p_i - q_j| comes from prefix sums of q and one
        searchsorted of p_i, so the cost is O(n log R) per direction, not O(R).
        Measurement sorts are shared across the scan as in loss()."""
        import torch
        n, R = len(x), len(xr)
        if R % n:
            raise ValueError(f"emd_sliced: reference size {R} is not a multiple of {n}")
        c = R // n
        dev = _torch_device(self.device)
        f64 = dict(dtype=torch.float64, device=dev)
        a_deg, base, inv, flip = _meas_directions(self.n_proj, self.phis)
        L = len(a_deg)
        a = torch.tensor(np.radians(a_deg), **f64)
        X = torch.tensor(np.column_stack([x, y]), **f64)
        Q = torch.sort(torch.cos(a)[:, None] * torch.tensor(xr, **f64)
                       + torch.sin(a)[:, None] * torch.tensor(yr, **f64), dim=1).values    # (L, R)
        Sq = torch.cat([torch.zeros(L, 1, **f64), torch.cumsum(Q, dim=1)], dim=1)       # (L, R+1)
        lo = torch.arange(n, device=dev) * c
        hi = lo + c
        w1 = torch.empty(inv.shape, **f64)                                              # (phi, slice)
        step = max(1, CHUNK // n)
        with torch.no_grad():
            for i in range(0, len(base), step):                 # a block of distinct directions
                t = torch.tensor(np.radians(base[i:i + step]), **f64)
                S = torch.sort(torch.cos(t)[:, None] * X[:, 0] + torch.sin(t)[:, None] * X[:, 1],
                               dim=1).values                                            # (B, n)
                S = torch.cat([S, -S.flip(1)])                  # row B + b: direction base[i + b] + 180
                for l in range(L):
                    sel = np.flatnonzero((inv[:, l] >= i) & (inv[:, l] < i + step))
                    for j in range(0, len(sel), step):
                        fs = sel[j:j + step]
                        rows = inv[fs, l] - i + len(t) * flip[fs, l]
                        p = S[torch.as_tensor(rows, device=dev)]                        # (F', n)
                        k = torch.searchsorted(Q[l].contiguous(), p)
                        k = torch.minimum(torch.maximum(k, lo), hi)                    # split point in block
                        s = Sq[l]
                        cost = p * (k - lo) - (s[k] - s[lo]) + (s[hi] - s[k]) - p * (hi - k)
                        w1[torch.as_tensor(fs, device=dev), l] = cost.sum(dim=1) / R
        return (w1.mean(dim=1) if self.reduce == "mean" else w1.max(dim=1).values).cpu().numpy()


def kr_1d(m, n_half, bin_width):
    """Exact 1D Kantorovich-Rubinstein value for each row of m [R, K] (signed mass per bin):

        max_f  sum_k f_k m_k   s.t.  |f_k| <= n_half * bin_width,  |f_{k+1} - f_k| <= bin_width

    Viterbi max-plus pass over the 2 * n_half + 1 levels (exact: the constraints have
    integer vertices in units of bin_width). Value only, no traceback; same recursion as
    _kr_dual_potential in protoninfer/losses.py. m: torch tensor. Returns [R]."""
    import torch
    levels = bin_width * torch.arange(-n_half, n_half + 1, dtype=m.dtype, device=m.device)
    V = m[:, :1] * levels
    for k in range(1, m.shape[1]):
        V = torch.nn.functional.max_pool1d(V[:, None], 3, stride=1, padding=1)[:, 0] + m[:, k:k + 1] * levels
    return V.max(dim=1).values


@register("emd_sliced_kr")
class EMDSlicedKR(PointCloudScan):
    """Sliced unbalanced W1 (`suw1`): mean over `n_proj` directions of the exact 1D
    Kantorovich-Rubinstein distance (transport at |x - y|, create/destroy at `lam` [mm]
    per unit; each event has mass 1/N_ref). Projections are binned on a `bin_width` [mm]
    grid with linear interpolation; lam is rounded to a multiple of bin_width.
    lam -> inf with equal sizes recovers sliced W1 up to binning error. Each direction
    chooses independently where to create/destroy mass. torch; GPU if available."""

    def __init__(self, n_proj=32, lam=100.0, bin_width=2.0, device=None, chunk_values=5e7, **kw):
        super().__init__(**kw)
        self.n_proj, self.lam, self.bin_width = int(n_proj), float(lam), float(bin_width)
        self.device, self.chunk_values = device, float(chunk_values)

    def loss_options(self):
        return (self.n_proj, self.lam, self.bin_width)

    def loss(self, meas, ref, geom):
        import torch
        dev = _torch_device(self.device)
        x, y, xr, yr = self.clouds(meas, ref, geom)
        bw, nr = self.bin_width, len(xr)
        n_half = max(1, int(round(self.lam / bw)))
        r_max = float(max(np.hypot(x, y).max(), np.hypot(xr, yr).max())) + bw
        K = int(np.ceil(2 * r_max / bw)) + 2
        a = torch.tensor(_directions(self.n_proj), dtype=torch.float64, device=dev)
        X = torch.tensor(np.column_stack([x, y]), dtype=torch.float64, device=dev)
        Y = torch.tensor(np.column_stack([xr, yr]), dtype=torch.float64, device=dev)

        def binned(P, w):
            """P [..., n] projections -> [..., K] mass histogram (linear interpolation)."""
            u = (P + r_max) / bw
            lo = u.floor()
            hi_w = u - lo
            lo = lo.long()
            h = torch.zeros(*P.shape[:-1], K, dtype=P.dtype, device=dev)
            h.scatter_add_(-1, lo, w * (1 - hi_w))
            h.scatter_add_(-1, lo + 1, w * hi_w)
            return h

        H_ref = binned(torch.cos(a)[:, None] * Y[:, 0] + torch.sin(a)[:, None] * Y[:, 1], 1.0 / nr)  # [L, K]
        step = max(1, int(self.chunk_values // (self.n_proj * max(len(x), K))))
        out = []
        with torch.no_grad():
            for i in range(0, len(self.phis), step):
                t = torch.tensor(np.radians(self.phis[i:i + step]), dtype=torch.float64, device=dev)
                A = a[None, :] - t[:, None]                                  # [phi, L]
                P = torch.cos(A)[..., None] * X[:, 0] + torch.sin(A)[..., None] * X[:, 1]
                m = binned(P, 1.0 / nr) - H_ref[None]                        # [phi, L, K]
                v = kr_1d(m.reshape(-1, K), n_half, bw).view(len(t), self.n_proj)
                out.append(v.mean(dim=1).cpu().numpy())
        return np.concatenate(out)
