"""Evaluation context shared by every algorithm.

``Context.evaluate`` is the only way algorithms may obtain energies/forces. Each
call is one *ionic step* (one NEP89 energy+force evaluation of the full system)
and is counted against the run budget; when the budget is spent it raises
``BudgetExhausted``. The context also keeps the best-so-far energy trace that is
used to measure how many ionic steps an algorithm needed to reach the global
minimum.
"""
import time

import numpy as np
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.stress import full_3x3_to_voigt_6_stress

from .guard import Guard
from .nep import FastNEP

TRACE_RESOLUTION = 1e-3  # eV; best-so-far trace granularity
BAD_ENERGY = 1e6


class BudgetExhausted(Exception):
    pass


class RelaxResult:
    __slots__ = ("pos", "cell", "e", "converged", "nsteps", "fmax", "aborted")

    def __init__(self, pos, cell, e, converged, nsteps, fmax, aborted=False):
        self.pos, self.cell, self.e = pos, cell, e
        self.converged, self.nsteps, self.fmax, self.aborted = converged, nsteps, fmax, aborted

    def __repr__(self):
        return (f"RelaxResult(e={self.e:.4f}, converged={self.converged}, nsteps={self.nsteps}, "
                f"fmax={self.fmax:.3g}, aborted={self.aborted})")


class Context:
    def __init__(self, problem, budget=None, nep=None):
        self.p = problem
        self.budget = int(budget or problem.budget)
        if nep is None:
            nep = FastNEP(problem.symbols, problem.cell0, problem.random_structure(np.random.default_rng(0))[0])
        self.nep = nep
        self.guard = Guard(problem, problem.guard_factor, problem.guard_k) \
            if problem.guard_factor > 0 else None
        self.n_guard = 0
        self.free = problem.free_mask
        self._frozen = np.flatnonzero(~problem.free_mask)
        self._pbc = problem.pbc if problem.pbc.any() else None
        self.n_frozen_fix = 0
        self.nsteps = 0
        self.best_e = np.inf
        self.best_pos = None
        self.best_cell = None
        self.best_step = 0
        self._trace_e = np.inf
        self.trace_steps = []
        self.trace_e = []
        self.n_relax = 0
        self.n_bad = 0
        self.minima = []  # energies of finished local relaxations
        self.stats = {}  # free-form algorithm statistics
        self.last_virial = None
        self.last_atomic_e = None
        self.t0 = time.time()

    # ---------------------------------------------------------------- budget
    def remaining(self):
        return self.budget - self.nsteps

    # ------------------------------------------------------------ evaluation
    def evaluate(self, pos, cell=None):
        """One ionic step. Returns (energy, forces) with forces zeroed on fixed atoms."""
        if self.nsteps >= self.budget:
            raise BudgetExhausted()
        if cell is None or not self.p.variable_cell:
            cell_arg = None
            cell = self.p.cell0
        else:
            cell_arg = cell
        if self._frozen.size:
            # frozen host atoms stay at their reference positions (counted, so that
            # algorithms that move them can be found in the smoke tests)
            ref = self.p.sub_pos[self._frozen]
            d = pos[self._frozen] - ref
            if np.abs(d).max() > 1e-6:
                frac = d @ np.linalg.inv(cell)
                frac[:, self.p.pbc] -= np.round(frac[:, self.p.pbc])
                if np.abs(frac @ cell).max() > 1e-6:
                    self.n_frozen_fix += 1
                pos = np.array(pos, float)
                pos[self._frozen] = ref
        x = pos
        if self._pbc is not None and np.abs(pos).max() > 4.0 * np.abs(cell).sum():
            # NEP_CPU slows down in proportion to how far atoms sit outside the cell; shifting
            # them by whole lattice vectors along periodic directions leaves E, F and virial unchanged
            fr = np.asarray(pos, float) @ np.linalg.inv(cell)
            fr[:, self._pbc] -= np.floor(fr[:, self._pbc])
            x = fr @ cell
        try:
            e, f, v, ea = self.nep.compute(x, cell_arg)
        except RuntimeError:
            # NEP_CPU throws on collapsed cells (too many neighbours); treat as a bad step
            self.nep.cell = np.full((3, 3), np.nan)
            e, f, v, ea = np.nan, np.zeros((len(pos), 3)), np.zeros((3, 3)), np.zeros(len(pos))
        if self.guard is not None:
            eg, fg, vg, eag = self.guard(pos, cell)
            if fg is not None:
                self.n_guard += 1
                e += eg
                f = f + fg
                v = v + vg
                ea = ea + eag
        self.nsteps += 1
        if not np.isfinite(e) or not np.all(np.isfinite(f)):
            self.n_bad += 1
            e = BAD_ENERGY
            f = np.zeros_like(f)
            v = np.zeros((3, 3))
            ea = np.full(len(ea), BAD_ENERGY / max(len(ea), 1))
        f[~self.free] = 0.0
        self.last_virial = v
        self.last_atomic_e = ea
        if e < self.best_e:
            self.best_e = e
            self.best_pos = np.array(pos, float)
            self.best_cell = np.array(cell, float)
            self.best_step = self.nsteps
            if e < self._trace_e - TRACE_RESOLUTION:
                self._trace_e = e
                self.trace_steps.append(self.nsteps)
                self.trace_e.append(e)
        return e, f

    def stress(self, cell):
        """Voigt stress (eV/A^3) of the last evaluation."""
        vol = abs(np.linalg.det(cell))
        return full_3x3_to_voigt_6_stress(-self.last_virial / vol)

    def atomic_energies(self):
        return None if self.last_atomic_e is None else np.array(self.last_atomic_e)

    # ------------------------------------------------------------ relaxation
    def relax(self, pos, cell=None, fmax=None, steps=None, method="lbfgs", abort=None, **kw):
        """Local relaxation; every energy/force call is one ionic step.

        abort(i, e, fmax_now) -> True stops early (the result is flagged aborted).
        Cell degrees of freedom are relaxed too when the problem has variable_cell.
        """
        fmax = self.p.fmax if fmax is None else fmax
        steps = self.p.max_relax_steps if steps is None else steps
        if self.p.variable_cell:
            cell = self.p.cell0 if cell is None else cell
            res = _relax_cell(self, pos, cell, fmax, steps, method, abort, **kw)
        else:
            fun = lambda x: self.evaluate(x)  # noqa: E731
            opt = lbfgs if method == "lbfgs" else fire
            x, e, f, conv, n, ab = opt(fun, np.array(pos, float), fmax, steps, abort=abort, **kw)
            res = RelaxResult(x, self.p.cell0.copy(), e, conv, n, _fmax(f), ab)
        self.n_relax += 1
        if not res.aborted:
            self.minima.append(round(float(res.e), 5))
        return res

    # ------------------------------------------------------------------ md
    def md(self, pos, vel, dt=1.0, nsteps=100, masses=None, callback=None, cell=None):
        """Velocity-Verlet MD on free atoms (dt in fs, vel in A/fs).

        callback(i, pos, vel, e, f) -> True stops the run. Returns
        (pos, vel, e, f, steps_done). Each step costs one ionic step.
        """
        if masses is None:
            from ase.data import atomic_masses
            masses = atomic_masses[self.p.numbers]
        conv = 1.0 / 103.642697  # (eV/A)/amu -> A/fs^2
        m = np.asarray(masses)[:, None]
        pos = np.array(pos, float)
        vel = np.array(vel, float)
        vel[~self.free] = 0.0
        e, f = self.evaluate(pos, cell)
        i = 0
        for i in range(1, nsteps + 1):
            vel += 0.5 * dt * f / m * conv
            pos = pos + dt * vel
            e, f = self.evaluate(pos, cell)
            vel += 0.5 * dt * f / m * conv
            vel[~self.free] = 0.0
            if callback is not None and callback(i, pos, vel, e, f):
                break
        return pos, vel, e, f, i

    # ---------------------------------------------------------------- output
    def summary(self):
        return dict(
            nsteps=self.nsteps, budget=self.budget, best_e=float(self.best_e),
            best_step=self.best_step, trace_steps=self.trace_steps,
            trace_e=[round(float(x), 6) for x in self.trace_e], n_relax=self.n_relax,
            n_bad=self.n_bad, n_guard=self.n_guard, n_frozen_fix=self.n_frozen_fix, n_minima=len(self.minima),
            n_unique_minima=len(set(round(x, 3) for x in self.minima)),
            wall=round(time.time() - self.t0, 3), stats=self.stats)


def _fmax(f):
    return float(np.sqrt((np.asarray(f) ** 2).sum(axis=1).max())) if len(f) else 0.0


# ======================================================================
# Optimizers. ``fun(x) -> (e, f)`` with f = -grad (zero on frozen rows).
# Return (x, e, f, converged, nsteps, aborted).
# ======================================================================
def lbfgs(fun, x, fmax=0.05, steps=500, memory=20, maxstep=0.2, h0=1.0 / 70.0,
          maxerise=1e-4, abort=None):
    """L-BFGS with step-length cap and GMIN-style backtracking on energy rises."""
    e, f = fun(x)
    nst = 1
    s_hist, y_hist, rho_hist = [], [], []
    for it in range(steps):
        fm = _fmax(f)
        if fm < fmax:
            return x, e, f, True, nst, False
        if abort is not None and abort(it, e, fm):
            return x, e, f, False, nst, True
        g = -f.ravel()
        q = g.copy()
        alphas = []
        for s, y, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
            a = rho * np.dot(s, q)
            q -= a * y
            alphas.append(a)
        if s_hist:
            gamma = np.dot(s_hist[-1], y_hist[-1]) / np.dot(y_hist[-1], y_hist[-1])
            gamma = min(max(gamma, h0 * 0.01), h0 * 100)
        else:
            gamma = h0
        z = gamma * q
        for s, y, rho, a in zip(s_hist, y_hist, rho_hist, reversed(alphas)):
            b = rho * np.dot(y, z)
            z += s * (a - b)
        d = -z.reshape(-1, 3)
        if np.dot(d.ravel(), g) >= 0:  # not a descent direction: reset
            s_hist, y_hist, rho_hist = [], [], []
            d = h0 * f
        lmax = np.sqrt((d ** 2).sum(axis=1).max())
        if lmax > maxstep:
            d *= maxstep / lmax
        for _ in range(6):
            xn = x + d
            en, fn = fun(xn)
            nst += 1
            if en <= e + maxerise:
                break
            d *= 0.5
            s_hist, y_hist, rho_hist = [], [], []
        else:
            # could not decrease: accept anyway to avoid stalling
            pass
        sk = (xn - x).ravel()
        yk = (f - fn).ravel()
        sy = np.dot(sk, yk)
        if sy > 1e-12:
            s_hist.append(sk)
            y_hist.append(yk)
            rho_hist.append(1.0 / sy)
            if len(s_hist) > memory:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
        x, e, f = xn, en, fn
    return x, e, f, _fmax(f) < fmax, nst, False


def fire(fun, x, fmax=0.05, steps=500, dt=0.1, maxstep=0.2, dtmax=1.0, nmin=5, finc=1.1,
         fdec=0.5, astart=0.1, fa=0.99, abort=None):
    """FIRE (Bitzek et al. 2006) with ASE default parameters."""
    e, f = fun(x)
    nst = 1
    v = np.zeros_like(x)
    a = astart
    npos = 0
    for it in range(steps):
        fm = _fmax(f)
        if fm < fmax:
            return x, e, f, True, nst, False
        if abort is not None and abort(it, e, fm):
            return x, e, f, False, nst, True
        vf = np.vdot(f, v)
        if vf > 0.0:
            v = (1.0 - a) * v + a * f / np.sqrt(np.vdot(f, f)) * np.sqrt(np.vdot(v, v))
            if npos > nmin:
                dt = min(dt * finc, dtmax)
                a *= fa
            npos += 1
        else:
            v[:] = 0.0
            a = astart
            dt *= fdec
            npos = 0
        v += dt * f
        dr = dt * v
        lmax = np.sqrt((dr ** 2).sum(axis=1).max())
        if lmax > maxstep:
            dr *= maxstep / lmax
        x = x + dr
        e, f = fun(x)
        nst += 1
    return x, e, f, _fmax(f) < fmax, nst, False


# ----------------------------------------------------------- variable cell
def _min_height(cell):
    return float(1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1).max())


def _relax_cell(ctx, pos, cell, fmax, steps, method, abort, **kw):
    """Joint relaxation of positions and cell (ASE UnitCellFilter parameterisation).

    Variables are reference-frame positions r0 = r F^-T and cell_factor * F, where
    F is the deformation gradient (cell = cell_start F^T) and cell_factor = n.
    Generalised forces are f F for atoms and virial F^-T / cell_factor for the
    cell. The relaxation is aborted if the volume leaves 0.25-4x the start or
    the cell collapses (thinnest interplanar spacing < 0.2 V^(1/3)).

    Repeated strain moves let the cell *basis* drift flat while the lattice stays
    well shaped; such a start cell (spacing < 0.4 V^(1/3)) is replaced by the
    Minkowski-reduced basis of the same lattice (same crystal and energy,
    Cartesian positions unchanged), so the collapse check only sees real collapses.
    """
    cell0 = np.array(cell, float)
    if _min_height(cell0) < 0.4 * abs(np.linalg.det(cell0)) ** (1 / 3):
        # variable-cell bulk is periodic in 3D (proxies of ctx may lack .p)
        pbc = getattr(getattr(ctx, "p", None), "pbc", (True, True, True))
        cell0 = np.array(minkowski_reduce(cell0, pbc)[0], float)
    n = len(pos)
    cf = float(n)
    v0 = abs(np.linalg.det(cell0))
    bad = {"flag": False}

    def unpack(x):
        F = x[n:] / cf
        return x[:n] @ F.T, cell0 @ F.T, F

    def fun(x):
        p, c, F = unpack(x)
        vol = abs(np.linalg.det(c))
        if not (0.25 * v0 < vol < 4.0 * v0) or _min_height(c) < 0.2 * vol ** (1 / 3):
            bad["flag"] = True
        e, f = ctx.evaluate(p, c)
        vir = ctx.last_virial
        vir = 0.5 * (vir + vir.T)
        vir = np.linalg.solve(F, vir.T).T
        return e, np.vstack([f @ F, vir / cf])

    def stop(i, e, fm):
        if bad["flag"]:
            return True
        return abort(i, e, fm) if abort is not None else False

    x0 = np.vstack([np.array(pos, float), cf * np.eye(3)])
    opt = lbfgs if method == "lbfgs" else fire
    x, e, f, conv, nst, ab = opt(fun, x0, fmax, steps, abort=stop, **kw)
    p, c, _ = unpack(x)
    return RelaxResult(p, c, e, conv, nst, _fmax(f), ab)
