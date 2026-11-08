"""Minima hopping (Goedecker 2004) with softening and variable-cell escapes."""
import math

import numpy as np
from ase.data import atomic_masses

from .. import ops

LABEL = "Minima hopping"
FAMILY = "baseline"
REFERENCE = ("Goedecker, J. Chem. Phys. 120, 9911 (2004); Schoenborn, Goumans & Goedecker, "
             "J. Chem. Phys. 130, 144108 (2009); Amsler & Goedecker, J. Chem. Phys. 133, "
             "224104 (2010); ASE MinimaHopping (Peterson, Top. Catal. 57, 40 (2014))")
DESCRIPTION = (
    "Minima hopping: velocity-Verlet MD escape from the current minimum with "
    "Maxwell-Boltzmann velocities at exactly temperature T, stopped at the mdmin-th "
    "potential-energy minimum along the trajectory (ASE 2-down/2-up rule, restart from that "
    "point), then L-BFGS relaxation. Time step starts at 1 fs (ASE) and adapts after each MD "
    "(x1.05 if the total-energy range is < 1% of the potential-energy range, else /1.05; "
    "Goedecker-group codes), which makes heavy-atom escapes several times cheaper. Feedback: same minimum as current -> T*=beta1 and retry; "
    "previously visited -> T*=beta2; new -> T*=beta3; every non-identical minimum is accepted "
    "if E_new-E_cur < Ediff (Ediff*=alpha1) else rejected (Ediff*=alpha2). ASE defaults "
    "(T0 1000 K, Ediff0 0.5 eV, beta 1.1, alpha 0.98, mdmin 2). Deviations/simplifications: "
    "(1) paper logic, not ASE's: visited minima also face the Ediff test and the history holds "
    "every minimum found; minima are identified by energy (0.01 eV) + ops.fingerprint; "
    "(2) softening (Schoenborn 2009) of the initial velocity direction with gradient-only dimer "
    "iterations (dimer 0.01 A, step 1e-3 A^2/eV as in the Goedecker-group reference code), "
    "using force differences F(x0+d)-F(x0) because our minima are only relaxed to "
    "fmax=0.05 eV/A; free clusters: COM translation/rotation projected out, bulk: COM "
    "translation; (3) variable-cell bulk (Amsler 2010): the escape MD also moves the lattice "
    "vectors (mass 0.075*sum(m), symmetric velocities, MB at T) in a simplified generalised-"
    "coordinate dynamics (atoms in cell-attached reference Cartesians, not the full Parrinello-"
    "Rahman metric), followed by a variable-cell relaxation; (4) robustness: MD is stopped and "
    "relaxed from the last sane point if energy conservation breaks (NEP holes, NaN) or the "
    "cell degenerates, MD capped at md_max steps, detached search atoms re-seated before "
    "relaxation (non-bulk), T kept within [10, 1e5] K, aborted variable-cell relaxations are "
    "discarded (no feedback).")
DEFAULTS = dict(T0=1000.0, Ediff0=0.5, beta1=1.1, beta2=1.1, beta3=1.0 / 1.1, alpha1=0.98,
                alpha2=1.0 / 0.98, mdmin=2, dt=1.0, nsoft=20, soft_alpha=1e-3, soft_dx=0.01,
                enhanced=0.0, md_max=1000, cell_mass=0.075, dt_tol=0.01, dt_fac=1.05)

CONV = 1.0 / 103.642697  # (eV/A)/amu -> A/fs^2 ; 1 amu A^2/fs^2 = 103.64 eV


class _Min:
    __slots__ = ("pos", "cell", "e", "fp", "g0")

    def __init__(self, pos, cell, e, fp):
        self.pos, self.cell, self.e, self.fp = pos, cell, e, fp
        self.g0 = None  # cached generalised force at the minimum (for softening + MD start)


class _Space:
    """Generalised coordinates of the escape dynamics.

    Fixed cell: Cartesian positions of the free atoms. Variable cell: positions in a
    Cartesian frame attached to the starting cell (R0, pos = R0 h0^-1 H) plus the lattice
    vectors H (rows). Generalised forces: f (h0^-1 H)^T for atoms and H^-T W for the
    lattice (W = virial).
    """

    def __init__(self, ctx, cell_mass):
        p = ctx.p
        self.ctx, self.p = ctx, p
        self.free = np.where(p.free_mask)[0]
        self.nf = len(self.free)
        self.na = 3 * self.nf
        self.vc = bool(p.variable_cell)
        masses = atomic_masses[p.numbers]
        self.m_at = masses[self.free]
        m = np.repeat(self.m_at, 3)
        if self.vc:
            m = np.concatenate([m, np.full(9, cell_mass * masses.sum())])
        self.m = m
        self.ndof = self.na + (6 if self.vc else 0)
        self.transl = bool(p.free_mask.all())  # no frozen atoms: translations are zero modes
        self.rot = self.transl and not self.vc and p.category != "bulk"
        self.etol = max(2.0, 0.5 * self.nf)  # eV; energy-conservation breakdown threshold

    # -------------------------------------------------------------- frame
    def start(self, mn):
        self.base = np.array(mn.pos, float)
        self.h0 = np.array(mn.cell, float)
        self.h0inv = np.linalg.inv(self.h0)
        self.v0 = abs(np.linalg.det(self.h0))
        x = self.base[self.free].ravel()
        if self.vc:
            x = np.concatenate([x, self.h0.ravel()])
        if self.rot:
            r = ops.unwrap_search(self.p, self.base, self.h0)[self.free]
            w = self.m_at / self.m_at.sum()
            self.rref = r - w @ r
            inertia = (self.m_at * (self.rref ** 2).sum(1)).sum() * np.eye(3) - \
                (self.m_at[:, None] * self.rref).T @ self.rref
            self.iinv = np.linalg.pinv(inertia)
        return x

    def struct(self, x):
        pos = self.base.copy()
        if not self.vc:
            pos[self.free] = x.reshape(-1, 3)
            return pos, self.h0
        h = x[self.na:].reshape(3, 3)
        mt = self.h0inv @ h
        pos = pos @ mt
        pos[self.free] = x[:self.na].reshape(-1, 3) @ mt
        return pos, h

    def evaluate(self, x):
        pos, h = self.struct(x)
        e, f = self.ctx.evaluate(pos, h)
        if not self.vc:
            return e, f[self.free].ravel()
        mt = self.h0inv @ h
        w = self.ctx.last_virial
        w = 0.5 * (w + w.T)
        try:
            gh = np.linalg.solve(h.T, w)
        except np.linalg.LinAlgError:
            gh = np.zeros((3, 3))
        return e, np.concatenate([(f[self.free] @ mt.T).ravel(), gh.ravel()])

    def cell_ok(self, x):
        if not self.vc:
            return True
        h = x[self.na:].reshape(3, 3)
        vol = abs(np.linalg.det(h))
        if not (0.5 * self.v0 < vol < 2.0 * self.v0):
            return False
        try:
            hmin = 1.0 / np.linalg.norm(np.linalg.inv(h).T, axis=1).max()
        except np.linalg.LinAlgError:
            return False
        return hmin > 0.3 * vol ** (1.0 / 3.0)

    # -------------------------------------------------------- velocities
    def project(self, v):
        """Remove rigid-body zero modes (COM translation; rotation for free clusters;
        cell rotation for variable-cell)."""
        v = np.array(v, float)
        if self.transl:
            va = v[:self.na].reshape(-1, 3)
            va -= (self.m_at @ va) / self.m_at.sum()
            if self.rot:
                ang = (self.m_at[:, None] * np.cross(self.rref, va)).sum(0)
                va -= np.cross(self.iinv @ ang, self.rref)
            v[:self.na] = va.ravel()
        if self.vc:
            a = self.h0inv @ v[self.na:].reshape(3, 3)
            v[self.na:] = (self.h0 @ (0.5 * (a + a.T))).ravel()
        return v

    def ekin(self, v):
        return 0.5 * float(self.m @ (v * v)) / CONV

    def random_velocities(self, temperature, rng):
        v = ops.mb_velocities(self.p, temperature, rng)[self.free].ravel()
        if self.vc:
            sig = math.sqrt(ops.KB * temperature / self.m[-1] * CONV)
            v = np.concatenate([v, rng.normal(size=9) * sig])
        return self.project(v)


def _soften(sp, x0, g0, d, nsoft, alpha, dx):
    """Schoenborn et al. 2009 gradient-only dimer iterations: rotate the escape direction
    towards low curvature (stopped early, by design). Returns a direction vector."""
    d = sp.project(d)
    nd = np.linalg.norm(d)
    if nd < 1e-12:
        return d, 0
    d = d * (dx / nd)
    it = 0
    for it in range(1, nsoft + 1):
        e, g = sp.evaluate(x0 + d)
        if e >= 1e5:  # bad evaluation: keep current direction
            break
        g = g - g0
        curv = -float(g @ d) / float(d @ d)
        r = g + curv * d
        if np.linalg.norm(r) < curv * dx * 0.5:
            break
        dn = sp.project(d + alpha * r)
        nn = np.linalg.norm(dn)
        if not np.isfinite(nn) or nn < 1e-12:
            break
        d = dn * (dx / nn)
    return d, it


def _md(sp, x0, e0, g0, v, dt, mdmin, md_max):
    """Velocity Verlet until the mdmin-th potential-energy minimum (ASE PassedMinimum:
    two downward then two upward steps; the configuration at the minimum is returned).
    Also returns the energy-conservation ratio (Etot range / Epot range) used for the
    time-step feedback of the Goedecker-group codes."""
    afac = 0.5 * dt * CONV / sp.m
    x, g, e = x0, g0, e0
    etot0 = e0 + sp.ekin(v)
    tmin = tmax = etot0
    ens = []
    xs = []
    nmin = 0
    status = "cap"
    nst = 0
    for nst in range(1, md_max + 1):
        vh = v + afac * g
        xn = x + dt * vh
        if not sp.cell_ok(xn):
            status = "cell"
            break
        en, gn = sp.evaluate(xn)
        vn = vh + afac * gn
        etot = en + sp.ekin(vn)
        if not np.isfinite(etot) or abs(etot - etot0) > sp.etol:
            status = "explode"
            break
        x, e, g, v = xn, en, gn, vn
        tmin, tmax = min(tmin, etot), max(tmax, etot)
        ens.append(e)
        xs.append(x)
        if len(xs) > 3:
            xs.pop(0)
        if (len(ens) >= 5 and ens[-1] >= ens[-2] >= ens[-3]
                and ens[-3] <= ens[-4] <= ens[-5]):
            nmin += 1
            if nmin >= mdmin:
                x = xs[0]
                status = "ok"
                break
    erange = (max(ens) - min(min(ens), e0)) if ens else 0.0
    ratio = (tmax - tmin) / erange if erange > 1e-9 else None
    if status == "explode":
        ratio = np.inf
    return x, nst, status, ratio


class _History:
    def __init__(self, p):
        self.p = p
        self.e = np.zeros(0)
        self.fp = []
        self.visits = []

    def find(self, e, fp):
        if not len(self.e):
            return None
        for k in np.where(np.abs(self.e - e) < 0.01)[0]:
            if ops.same_structure(self.p, e, fp, self.e[k], self.fp[k]):
                return int(k)
        return None

    def add(self, e, fp):
        self.e = np.append(self.e, e)
        self.fp.append(fp)
        self.visits.append(1)
        return len(self.visits) - 1


def run(ctx, rng, T0=1000.0, Ediff0=0.5, beta1=1.1, beta2=1.1, beta3=1.0 / 1.1, alpha1=0.98,
        alpha2=1.0 / 0.98, mdmin=2, dt=1.0, nsoft=20, soft_alpha=1e-3, soft_dx=0.01,
        enhanced=0.0, md_max=1000, cell_mass=0.075, dt_tol=0.01, dt_fac=1.05):
    p = ctx.p
    sp = _Space(ctx, cell_mass)
    reseat = not p.variable_cell and p.category != "bulk"

    def fingerprint(res):
        return ops.fingerprint(p, res.pos, res.cell)

    def bad(res):
        if res.aborted or not np.isfinite(res.e) or res.e >= 1e5:
            return True
        if p.variable_cell:  # _relax_cell may stop at its step limit with a degenerate cell
            try:
                vol = abs(np.linalg.det(res.cell))
                hmin = 1.0 / np.linalg.norm(np.linalg.inv(res.cell).T, axis=1).max()
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                return True
            return not (vol > 0.1 * p.n * float(p.vpa or 1.0) and hmin > 0.2 * vol ** (1 / 3))
        return False

    while True:
        pos, cell = p.random_structure(rng)
        res = ctx.relax(pos, cell)
        if not bad(res):
            break
    cur = _Min(res.pos, res.cell, res.e, fingerprint(res))
    hist = _History(p)
    hist.add(cur.e, cur.fp)

    T, Ediff, dt = float(T0), float(Ediff0), float(dt)
    st = dict(escapes=0, same=0, new=0, old=0, accepted=0, md_steps=0, soft_steps=0,
              relax_steps=0, explode=0, cellstop=0, mdcap=0, reseated=0, aborted=0)
    while True:
        # ---------------------------------------------------- escape trial
        x0 = sp.start(cur)
        if cur.g0 is None:
            _, cur.g0 = sp.evaluate(x0)
        v = sp.random_velocities(T, rng)
        if nsoft > 0:
            v, nit = _soften(sp, x0, cur.g0, v, nsoft, soft_alpha, soft_dx)
            st["soft_steps"] += nit
        ek = sp.ekin(v)
        if ek > 0:
            v = v * math.sqrt(0.5 * sp.ndof * ops.KB * T / ek)
        x, nmd, status, ratio = _md(sp, x0, cur.e, cur.g0, v, dt, mdmin, md_max)
        st["md_steps"] += nmd
        if ratio is not None:  # time-step feedback (Krummenacher et al. 2024 reference code)
            dt = min(max(dt * (dt_fac if ratio < dt_tol else 1.0 / dt_fac), 0.1), 20.0)
        if status == "explode":
            st["explode"] += 1
        elif status == "cell":
            st["cellstop"] += 1
        elif status == "cap":
            st["mdcap"] += 1
        pos, cell = sp.struct(x)
        if reseat:
            lost = ops.detached_atoms(p, pos, cell)
            if len(lost):
                pos = ops.relocate_atoms(p, pos, rng, lost, cell)
                st["reseated"] += 1
        res = ctx.relax(pos, cell)
        st["escapes"] += 1
        st["relax_steps"] += res.nsteps
        if bad(res):
            st["aborted"] += 1
            continue
        fp = fingerprint(res)
        # ------------------------------------------------------- feedback
        if ops.same_structure(p, res.e, fp, cur.e, cur.fp):
            T *= beta1
            st["same"] += 1
        else:
            k = hist.find(res.e, fp)
            if k is not None:
                hist.visits[k] += 1
                T *= beta2 * (1.0 + enhanced * math.log(hist.visits[k]))
                st["old"] += 1
            else:
                hist.add(res.e, fp)
                T *= beta3
                st["new"] += 1
            if res.e - cur.e < Ediff:
                cur = _Min(res.pos, res.cell, res.e, fp)
                Ediff *= alpha1
                st["accepted"] += 1
            else:
                Ediff *= alpha2
        T = min(max(T, 10.0), 1e5)
        n = st["escapes"]
        ctx.stats.update(st, T=round(T, 1), Ediff=round(Ediff, 4), n_minima=len(hist.visits),
                         dt=round(dt, 3),
                         md_avg=round(st["md_steps"] / n, 1),
                         relax_avg=round(st["relax_steps"] / n, 1),
                         e_cur=round(float(cur.e), 4))
