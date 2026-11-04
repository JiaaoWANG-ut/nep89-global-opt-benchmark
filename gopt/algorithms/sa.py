"""Simulated annealing by Langevin molecular dynamics followed by a quench."""
import numpy as np
from ase.data import atomic_masses
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .. import ops
from ..problem import min_image_distances

LABEL = "Simulated annealing (MD)"
FAMILY = "baseline"
REFERENCE = ("Kirkpatrick, Gelatt & Vecchi, Science 220, 671 (1983); Langevin-MD annealing + quench: "
             "Biswas & Hamann, Phys. Rev. B 34, 895 (1986); barostat: Berendsen et al., "
             "J. Chem. Phys. 81, 3684 (1984)")
DESCRIPTION = (
    "Multi-start MD simulated annealing. Every cycle starts from a fresh random structure "
    "(random cell for bulk), draws Maxwell-Boltzmann velocities at t_hot=2500 K and runs "
    "n_md=3000 Langevin (BAOAB, friction 0.01/fs) MD steps on the search atoms only (frozen "
    "and relax-only substrate atoms stay fixed): 10% at t_hot, then geometric cooling "
    "T_k = t_hot (t_final/t_hot)^(k/K) to t_final=300 K, then an L-BFGS quench (ctx.relax). "
    "Time step 2 fs, scaled by sqrt(m_min/40 amu) for lighter elements (min 0.5 fs). Bulk "
    "(variable cell): anisotropic Berendsen barostat at P=0 (virial + kinetic stress, tau_P=1 ps, "
    "compressibility 1/(50 GPa)), volume kept within 0.5-2x the nominal volume, Gauss lattice "
    "reduction of skewed cells, variable-cell quench. Explosion guards: speeds capped at 5x the "
    "rms thermal speed at t_hot; a non-finite energy aborts the cycle. Non-bulk (container-like "
    "evaporation guard): every 50 steps, fragments farther than 2x the covalent-radius sum from "
    "the largest one are moved back along the line of centres to 1.5x that sum and their "
    "centre-of-mass velocity is reflected (elastic wall); before the quench isolated atoms are "
    "re-seated on the cluster as in BH (ops.relocate_atoms). Budget tail: when less than a full "
    "cycle remains, the last anneal is compressed (same schedule shape) to the remaining steps "
    "minus a 150-step quench reserve, so the tail is not spent on an anneal that would never be "
    "quenched. Deviations from the textbook method: the short-dt rule for light elements, the "
    "evaporation guard, the speed cap and the compressed last cycle (all only fix broken or "
    "wasted trajectories); parameters are standard values (no tuning).")
DEFAULTS = dict(t_hot=2500.0, t_final=300.0, n_md=3000, hot_frac=0.1, schedule="geometric",
                dt_max=2.0, dt_min=0.5, m_ref=40.0, friction=0.01, tau_p=1000.0,
                bulk_modulus=50.0, check_every=50, frag_factor=2.0, dock_ratio=1.5, vcap=5.0,
                restart="random", quench_reserve=150)

KB = 8.617333262e-5          # eV/K
MV2E = 103.642697            # amu*A^2/fs^2 -> eV
GPA = 1.0 / 160.21766208     # GPa -> eV/A^3


def _reduce_cell(cell):
    """Pairwise (Gauss) lattice reduction; same lattice, shorter / more orthogonal vectors."""
    c = np.array(cell, float)
    for _ in range(60):
        changed = False
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                k = np.round(c[i] @ c[j] / (c[j] @ c[j]))
                if k != 0:
                    new = c[i] - k * c[j]
                    if new @ new < c[i] @ c[i] - 1e-10:
                        c[i] = new
                        changed = True
        if not changed:
            break
    if np.linalg.det(c) < 0:
        c = -c
    return c


def _min_height(cell):
    return float(1.0 / np.linalg.norm(np.linalg.inv(cell), axis=0).max())


class _Annealer:
    def __init__(self, ctx, rng, t_hot, t_final, n_md, hot_frac, schedule, dt_max, dt_min, m_ref,
                 friction, tau_p, bulk_modulus, check_every, frag_factor, dock_ratio, vcap, restart,
                 quench_reserve):
        p = ctx.p
        self.ctx, self.p, self.rng = ctx, p, rng
        self.bulk = p.category == "bulk" or p.variable_cell
        self.baro = bool(p.variable_cell)
        self.idx = np.asarray(p.search_idx)
        self.loc = np.full(p.n, -1)
        self.loc[self.idx] = np.arange(len(self.idx))
        m = atomic_masses[p.numbers[self.idx]][:, None]
        self.m = m
        self.t_hot = float(t_hot)
        self.dt = float(np.clip(dt_max * np.sqrt(m.min() / m_ref), dt_min, dt_max))
        self.t_final, self.hot_frac, self.schedule = float(t_final), float(hot_frac), schedule
        self.temps = self._schedule(n_md)
        self.quench_reserve = int(quench_reserve)
        # BAOAB Langevin constants
        self.hdt = 0.5 * self.dt
        self.hdt_acc = self.hdt / (m * MV2E)
        self.c1 = np.exp(-friction * self.dt)
        self.sig = np.sqrt((1.0 - self.c1 ** 2) * KB / (m * MV2E))
        self.vcap2 = (vcap ** 2) * 3.0 * KB * t_hot / (m[:, 0] * MV2E)
        # Berendsen barostat: strain per step = dt / (3 tau_p B) * (P - P0)
        self.kappa = self.dt / (3.0 * tau_p * bulk_modulus * GPA)
        self.check_every = max(int(check_every), 1)
        self.frag_factor = float(frag_factor)
        self.dock_ratio = float(dock_ratio)
        self.restart = restart
        if self.baro:
            self.v_nom = float(p.vpa) * p.n_search if p.vpa else abs(np.linalg.det(p.cell0))
        self.best = None
        self.last = None
        self.trace = None  # diagnostics: list of (step, T_target, T_kin, E, volume) when set
        self.st = dict(cycles=0, explode=0, reseat=0, vcap=0, reduce=0, vclamp=0,
                       quench_steps=0, short=0, dt=self.dt)
        self.quench_e = []

    # ------------------------------------------------------------- helpers
    def _schedule(self, n):
        """Target temperatures of an n-step anneal: hot_frac*n steps at t_hot, then cooling."""
        n = max(int(n), 2)
        n_hot = int(round(self.hot_frac * n))
        n_cool = max(n - n_hot, 1)
        x = np.arange(n_cool) / max(n_cool - 1, 1)
        if self.schedule == "linear":
            cool = self.t_hot + (self.t_final - self.t_hot) * x
        else:
            cool = self.t_hot * (self.t_final / self.t_hot) ** x
        return np.concatenate([np.full(n_hot, self.t_hot), cool])

    def _mic(self, d, cell):
        if not np.any(self.p.pbc):
            return d
        f = d @ np.linalg.inv(cell)
        f -= np.where(self.p.pbc, np.round(f), 0.0)
        return f @ cell

    def _start(self):
        p, rng = self.p, self.rng
        src = self.best if self.restart == "best" else self.last if self.restart == "last" else None
        if src is not None:
            return np.array(src[0]), np.array(src[1])
        for attempt in range(3):
            try:
                return p.random_structure(rng)
            except RuntimeError:
                if attempt == 2 and self.last is None:
                    raise
        pos, cell = np.array(self.last[0]), np.array(self.last[1])
        return ops.displace(p, pos, rng, 1.0, cell=cell), cell

    def _dock(self, pos, cell, frag, main):
        """Translate a fragment rigidly along the line of centres back to dock_ratio x the
        covalent-radius sum of the main cluster. Returns (new positions, outward unit normal)
        or (None, None) if every approach direction overlaps other atoms."""
        p, rng = self.p, self.rng
        mx = pos[main[0]] + self._mic(pos[main] - pos[main[0]], cell)
        cm = mx.mean(axis=0)
        a0 = cm + self._mic(pos[frag[0]] - cm, cell)
        fx = a0 + self._mic(pos[frag] - pos[frag[0]], cell)
        u0 = fx.mean(axis=0) - cm
        if np.linalg.norm(u0) < 1e-6:
            u0 = ops.random_unit(rng)
        u0 = u0 / np.linalg.norm(u0)
        lim = self.dock_ratio * (p.rcov[frag][:, None] + p.rcov[main][None, :])
        others = np.setdiff1d(np.arange(p.n), frag)
        dmin = p.dmin_factor * (p.rcov[frag][:, None] + p.rcov[others][None, :])
        a = fx[:, None, :] - mx[None, :, :]          # (k, m, 3)
        a2 = np.einsum("kmx,kmx->km", a, a)
        for trial in range(12):
            u = u0 if trial == 0 else u0 + 0.5 * ops.random_unit(rng)
            if p.category == "surface" and trial and u[2] > 0:
                u[2] = -u[2]  # perturbed approaches move the fragment upwards, away from the slab
            u = u / np.linalg.norm(u)
            # first t >= 0 at which some pair reaches |a - t u| = lim
            au = a @ u
            disc = au ** 2 - a2 + lim ** 2
            ok = disc >= 0
            if not ok.any():
                continue
            t = au[ok] - np.sqrt(disc[ok])
            t = t[t > -1e-9]
            if not len(t):
                continue
            new = fx - t.min() * u
            d = min_image_distances(new, pos[others], cell, p.pbc)
            if np.all(d >= dmin):
                out = np.array(pos)
                out[frag] = new
                return out, u
        return None, None

    def _reseat(self, pos, cell, v=None, temp=None, final=False):
        """Keep the search atoms together (non-bulk); returns new positions (v updated in place).

        Fragments farther than frag_factor x the covalent-radius sum from the largest fragment
        are moved back to dock_ratio x that sum along the line of centres and their
        centre-of-mass velocity is reflected (elastic container wall). Before the quench
        (final) isolated atoms are re-seated on the cluster like in BH (ops.relocate_atoms).
        """
        p, rng = self.p, self.rng
        s = self.idx
        redraw = []
        moved = False
        if p.connected and len(s) > 1:
            d = min_image_distances(pos[s], pos[s], cell, p.pbc)
            adj = d < self.frag_factor * (p.rcov[s][:, None] + p.rcov[s][None, :])
            ncomp, lab = connected_components(csr_matrix(adj), directed=False)
            if ncomp > 1:
                main = int(np.argmax(np.bincount(lab)))
                for c in range(ncomp):
                    if c == main:
                        continue
                    frag = s[lab == c]
                    new, u = self._dock(pos, cell, frag, s[lab == main])
                    if new is None:
                        new = ops.relocate_atoms(p, pos, rng, frag, cell)
                        redraw.extend(frag.tolist())
                    elif v is not None:
                        lf = self.loc[frag]
                        mf = self.m[lf]
                        vu = float(((mf * v[lf]).sum(axis=0) / mf.sum()) @ u)
                        if vu > 0:
                            v[lf] -= 2.0 * vu * u
                    pos = new
                    lab[lab == c] = main
                    moved = True
        if final or not p.connected:
            lost = ops.detached_atoms(p, pos, cell)
            if len(lost):
                pos = ops.relocate_atoms(p, pos, rng, lost, cell)
                redraw.extend(np.asarray(lost).tolist())
                moved = True
        if redraw and v is not None:
            loc = self.loc[np.asarray(redraw)]
            loc = loc[loc >= 0]
            v[loc] = rng.standard_normal((len(loc), 3)) * np.sqrt(KB * temp / (self.m[loc] * MV2E))
        if moved:
            self.st["reseat"] += 1
        return pos, moved

    def _fix_cell(self, pos, cell):
        """Lattice-reduce skewed cells and wrap atoms (bulk; physically identical structure)."""
        vol = abs(np.linalg.det(cell))
        if _min_height(cell) < 0.5 * vol ** (1 / 3):
            new = _reduce_cell(cell)
            if not np.allclose(new, cell):
                self.st["reduce"] += 1
                frac = pos @ np.linalg.inv(new)
                frac -= np.floor(frac)
                return frac @ new, new
        return pos, cell

    # --------------------------------------------------------------- anneal
    def anneal(self, pos, cell, temps=None):
        ctx, p, rng = self.ctx, self.p, self.rng
        idx = self.idx
        temps = self.temps if temps is None else temps
        pos = np.array(pos, float)
        cell = np.array(cell, float)
        v = ops.mb_velocities(p, self.t_hot, rng, idx=idx)[idx]
        e, f = ctx.evaluate(pos, cell)
        if not e < 1e5:
            return None
        hdt, hdt_acc, c1, sig, sqrt_t = self.hdt, self.hdt_acc, self.c1, self.sig, np.sqrt(temps)
        check, eye = self.check_every, np.eye(3)
        for k in range(len(temps)):
            v += hdt_acc * f[idx]
            x = pos[idx] + hdt * v
            v *= c1
            v += (sqrt_t[k] * sig) * rng.standard_normal(v.shape)
            x += hdt * v
            pos[idx] = x
            if self.baro:
                vol = abs(np.linalg.det(cell))
                w = ctx.last_virial
                press = (0.5 * (w + w.T) + MV2E * (self.m * v).T @ v) / vol
                eps = np.clip(self.kappa * press, -0.01, 0.01)
                mu = eye + 0.5 * (eps + eps.T)
                vnew = vol * np.linalg.det(mu)
                if not 0.5 * self.v_nom < vnew < 2.0 * self.v_nom:
                    mu = eye + (eps - np.trace(eps) / 3.0 * eye)  # keep shear, drop dilation
                    self.st["vclamp"] += 1
                pos = pos @ mu
                cell = cell @ mu
            e, f = ctx.evaluate(pos, cell)
            v += hdt_acc * f[idx]
            sp2 = np.einsum("ij,ij->i", v, v)
            over = sp2 > self.vcap2
            if over.any():
                v[over] *= np.sqrt(self.vcap2[over] / sp2[over])[:, None]
                self.st["vcap"] += 1
            if not e < 1e5:
                return None
            if self.trace is not None and k % 10 == 0:
                tkin = MV2E * float((self.m * v * v).sum()) / (3 * len(idx) * KB)
                self.trace.append((k, temps[k], tkin, e, abs(np.linalg.det(cell))))
            if (k + 1) % check == 0 and k + 1 < len(temps):
                if self.bulk:
                    if self.baro:
                        # equivalent lattice + wrapped atoms: energy and forces are unchanged
                        pos, cell = self._fix_cell(pos, cell)
                else:
                    pos, moved = self._reseat(pos, cell, v, temps[k])
                    if moved:
                        e, f = ctx.evaluate(pos, cell)
                        if not e < 1e5:
                            return None
        if self.bulk:
            if self.baro:
                pos, cell = self._fix_cell(pos, cell)
        else:
            pos, _ = self._reseat(pos, cell, final=True)
        return pos, cell

    def cycle(self):
        temps = self.temps
        # last cycle: compress the schedule into the remaining budget (minus a quench reserve)
        # instead of starting a full anneal that the budget would cut off before its quench
        room = self.ctx.remaining() - self.quench_reserve - 1
        if room < len(temps):
            temps = self._schedule(max(room, 100))
            self.st["short"] += 1
        pos, cell = self._start()
        out = self.anneal(pos, cell, temps)
        self.st["cycles"] += 1
        if out is None:
            self.st["explode"] += 1
            self.ctx.stats.update(self.st)
            return
        pos, cell = out
        res = self.ctx.relax(pos, cell)
        self.st["quench_steps"] += int(res.nsteps)
        if not res.aborted and np.isfinite(res.e) and res.e < 1e5:
            self.last = (res.pos, res.cell, res.e)
            if self.best is None or res.e < self.best[2]:
                self.best = self.last
        self.quench_e.append(round(float(res.e), 4))
        self.ctx.stats.update(self.st, quench_e=self.quench_e[-60:])


def run(ctx, rng, **params):
    kw = dict(DEFAULTS)
    kw.update(params)
    sa = _Annealer(ctx, rng, **kw)
    while True:
        sa.cycle()
