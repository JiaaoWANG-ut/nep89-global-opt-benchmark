"""Cut-and-regrow search with a learned site-energy model (CRG)."""
import numpy as np

from .. import ops
from ..context import BudgetExhausted
from ..problem import min_image_distances
from ._growth_sites import Geo, SiteModel

LABEL = "Cut-and-regrow (learned-site growth)"
FAMILY = "innovative"
REFERENCE = ("Builds on Wales & Doye, JPCA 101, 5111 (1997) (basin hopping); "
             "Shao, Cheng, Cai, JCP 120, 11401 (2004) (dynamic lattice searching: "
             "moving high-energy atoms to surface sites); Rondina & Da Silva, JCIM 53, "
             "2282 (2013) (revised basin hopping, surface/atomic-energy-guided moves); "
             "Deaven & Ho, PRL 75, 288 (1995) (fragment cut-and-splice); "
             "cluster growth-sequence searches (e.g. Northby, JCP 87, 6166 (1987); "
             "Takeuchi, JCIM 46, 2066 (2006)); Pickard & Needs (AIRSS) random restarts.")
DESCRIPTION = (
    "Metropolis walk over relaxed minima whose only structural move is CUT-AND-REGROW: "
    "k atoms are removed (the highest NEP per-atom-energy atoms, a random contiguous cap/"
    "periodic neighbourhood, or the least-coordinated atoms) and regrown one at a time "
    "on the remaining fragment, on the substrate/zeolite wall, or inside the periodic "
    "cell. Candidate sites are enumerated geometrically (3-atom trilateration hollows, "
    "bridges, atop sites at learned contact distances) and ranked by a free surrogate: "
    "a per-species ridge model of NEP per-atom energies on smooth species-resolved "
    "coordination features, refitted on the fly from the per-atom energies of every "
    "relaxation start and end point (no extra ionic steps); a site's cost is the new "
    "atom's energy plus the change of its neighbours' energies. Sites are drawn by a "
    "rank-geometric rule, vacated sites are tabu and species are regrown in random "
    "order (multi-element: plus energy-guided swaps). Relaxations are two-stage: a "
    "coarse L-BFGS to fmax_coarse, completed to the benchmark fmax only if the trial can "
    "still pass the pre-drawn Metropolis threshold. k grows on stagnation (1..~n/3); "
    "beyond kmax the walk restarts from a random structure. Bulk: regrowth in "
    "fractional space with periodic images plus random cell strain. Novelty vs BH/DLS/"
    "revised BH: growth-based moves of only the worst atoms with an on-the-fly learned "
    "per-site energy model replacing both random displacements and fixed lattices, "
    "adaptive cut size, and Metropolis-informed early termination of local relaxations.")
DEFAULTS = dict(kT=0.1, fmax_coarse=0.2, margin=0.05, stag_moves=6, kmax_frac=0.34,
                p_energy=0.5, p_fragment=0.3, site_q=0.5, p_swap=0.2, p_strain=0.5,
                strain=0.06, p_restart_best=0.0, n_init=1, species_aware=0.0)


class _Min:
    __slots__ = ("pos", "cell", "e", "ea")

    def __init__(self, pos, cell, e, ea):
        self.pos, self.cell, self.e, self.ea = pos, cell, e, ea


class _CRG:
    def __init__(self, ctx, rng, kT, fmax_coarse, margin, stag_moves, kmax_frac, p_energy,
                 p_fragment, site_q, p_swap, p_strain, strain, p_restart_best, n_init,
                 species_aware):
        self.ctx, self.rng, self.p = ctx, rng, ctx.p
        p = ctx.p
        self.geo = Geo(p)
        self.model = SiteModel(self.geo.ns)
        self.n = n = p.n_search
        self.multi = len(set(p.search_symbols)) > 1
        self.kmax = int(max(1, min(n - 1, max(2, int(round(kmax_frac * n))))))
        self.kT, self.fmax_c, self.margin = kT, fmax_coarse, margin
        self.stag_moves = max(1, int(stag_moves))
        self.p_energy, self.p_fragment, self.site_q = p_energy, p_fragment, site_q
        self.p_swap = p_swap if self.multi else 0.0
        self.p_strain = p_strain if p.variable_cell else 0.0
        self.strain, self.p_restart_best, self.n_init = strain, p_restart_best, max(1, int(n_init))
        self.species_aware = species_aware if self.multi else 0.0
        self.best = None
        self.st = dict(moves=0, accepted=0, restarts=0, stage2=0, regrow=0, swap=0,
                       no_site=0, aborted=0, errors=0, kmax=self.kmax)
        ctx.stats.update(self.st)

    def _safe(self, fn, *args):
        """Run a geometry/model helper; internal failures must never end the run."""
        try:
            return fn(*args)
        except BudgetExhausted:
            raise
        except Exception:
            self.st["errors"] += 1
            return None

    # ------------------------------------------------------------------ utils
    def _search_X(self, pos, cell):
        p = self.p
        if self.geo.bulk:
            return np.array(pos, float), np.array(pos[p.search_idx], float)
        pos = ops.unwrap_search(p, pos, cell)
        return pos, np.array(pos[p.search_idx], float)

    def _relax(self, pos, cell, fmax=None):
        ctx = self.ctx
        first = {}

        def grab(i, e, fm):
            if i == 0 and "ea" not in first:
                first["ea"] = ctx.atomic_energies()
            return False
        r = ctx.relax(pos, cell, fmax=fmax, abort=grab)
        return r, first.get("ea")

    def _learn(self, pos, cell, ea, bonds=False):
        if ea is not None:
            self._safe(self._learn_impl, pos, cell, ea, bonds)

    def _learn_impl(self, pos, cell, ea, bonds):
        p, geo = self.p, self.geo
        _, Xs = self._search_X(pos, cell)
        env = geo.build(Xs, geo.sp_s, cell)
        m = env.m
        sp, c, e = env.sp[:m], env.c[:m], ea[p.search_idx]
        if geo.has_sub and len(env.P) > m:
            hb = env.home[m:]
            touch = (env.c[m:] - geo.c_sub[hb]).sum(1) > 0.05
            if np.any(touch):
                sp = np.concatenate([sp, env.sp[m:][touch]])
                c = np.vstack([c, env.c[m:][touch]])
                e = np.concatenate([e, ea[hb[touch]]])
        self.model.add(sp, c, e)
        self.model.fit()
        if bonds:
            geo.learn_bonds(env)

    def _accept_threshold(self, cur):
        return cur.e - self.kT * np.log(max(self.rng.random(), 1e-300))

    # ---------------------------------------------------------------- relaxations
    def _new_random(self):
        p = self.p
        for _ in range(20):
            try:
                pos, cell = p.random_structure(self.rng)
            except BudgetExhausted:
                raise
            except Exception:
                self.st["errors"] += 1
                continue
            r, ea0 = self._relax(pos, cell)
            if r.aborted or r.e >= 1e5:
                continue
            ea = self.ctx.atomic_energies()
            self._learn(r.pos, r.cell, ea, bonds=True)
            return _Min(np.array(r.pos), np.array(r.cell), float(r.e), ea)
        return None

    def _trial(self, pos, cell, thr, grown):
        """Two-stage relaxation; returns (_Min or None, accepted)."""
        p, ctx = self.p, self.ctx
        two = self.fmax_c > p.fmax
        r, ea0 = self._relax(pos, cell, fmax=self.fmax_c if two else None)
        if grown and ea0 is not None:
            self._learn(pos, cell, ea0)
        if r.aborted or r.e >= 1e5:
            self.st["aborted"] += 1
            return None, False
        if two and r.fmax > p.fmax:
            if r.e - self.margin >= thr:
                self._learn(r.pos, r.cell, ctx.atomic_energies())
                return None, False
            self.st["stage2"] += 1
            r, _ = self._relax(r.pos, r.cell)
            if r.aborted or r.e >= 1e5:
                self.st["aborted"] += 1
                return None, False
        ea = ctx.atomic_energies()
        self._learn(r.pos, r.cell, ea, bonds=True)
        new = _Min(np.array(r.pos), np.array(r.cell), float(r.e), ea)
        return new, bool(r.e < thr)

    # ---------------------------------------------------------------- selection
    def _dev(self, ea):
        """Per-atom energy minus the mean of its species (search atoms)."""
        e = np.array(ea[self.p.search_idx], float)
        sp = self.geo.sp_s
        out = np.zeros_like(e)
        for a in np.unique(sp):
            m = sp == a
            out[m] = e[m] - e[m].mean()
        return out

    def _softmax_pick(self, w_dev, k, mask=None):
        n = len(w_dev)
        idx = np.arange(n) if mask is None else np.where(mask)[0]
        k = min(k, len(idx))
        d = w_dev[idx]
        sd = d.std()
        if not np.isfinite(sd) or sd < 1e-9:
            return self.rng.choice(idx, size=k, replace=False)
        w = np.exp(np.clip((d - d.max()) / sd, -50, 0))
        w /= w.sum()
        return self.rng.choice(idx, size=k, replace=False, p=w)

    def _select(self, cur, Xs, k):
        p, rng, geo = self.p, self.rng, self.geo
        u = rng.random()
        if u < self.p_energy:
            return self._softmax_pick(self._dev(cur.ea), k), "energy"
        if u < self.p_energy + self.p_fragment:
            if geo.bulk:
                i0 = rng.integers(self.n)
                d = min_image_distances(Xs[i0], Xs, cur.cell, p.pbc)[0]
                return np.argsort(d + 1e-6 * rng.random(self.n))[:k], "fragment"
            v = ops.random_unit(rng)
            proj = (Xs - Xs.mean(axis=0)) @ v
            return np.argsort(-proj)[:k], "fragment"
        env = geo.build(Xs, geo.sp_s, cur.cell)
        cn = env.c[: env.m].sum(axis=1) + 0.5 * rng.random(self.n)
        return np.argsort(cn)[:k], "lowcn"

    # ---------------------------------------------------------------- moves
    def _pick_rank(self, n):
        r = int(self.rng.geometric(1.0 - self.site_q)) - 1 if self.site_q > 0 else 0
        return min(r, n - 1)

    def _regrow(self, cur, pos, cell, k):
        p, geo, rng = self.p, self.geo, self.rng
        pos, Xs = self._search_X(pos, cell)
        removed, how = self._select(cur, Xs, k)
        removed = np.asarray(removed, int)
        present = np.ones(self.n, bool)
        present[removed] = False
        vac, vac_sp = Xs[removed].copy(), geo.sp_s[removed]
        todo = [int(i) for i in rng.permutation(removed)]
        inv = np.linalg.inv(cell) if geo.bulk else None
        aware = self.species_aware > 0 and rng.random() < self.species_aware
        mu = self._mu(cur) if aware else None
        while todo:
            env = geo.build(Xs[present], geo.sp_s[present], cell)
            left = np.unique(geo.sp_s[todo])
            idx, x = todo[0], None
            if aware and len(left) > 1:
                # joint (site, species) choice: cost relative to the species' mean energy
                XX, SS, AA = [], [], []
                for a in left:
                    X, sc = self._sites(env, a, vac, vac_sp, inv, cell)
                    if X is not None:
                        XX.append(X)
                        SS.append(sc - mu[a])
                        AA.append(np.full(len(X), a))
                if XX:
                    X, sc, A = np.vstack(XX), np.concatenate(SS), np.concatenate(AA)
                    j = np.argsort(sc)[self._pick_rank(len(sc))]
                    x = X[j]
                    idx = next(i for i in todo if geo.sp_s[i] == A[j])
            else:
                X, sc = self._sites(env, geo.sp_s[idx], vac, vac_sp, inv, cell)
                if X is not None:
                    x = X[np.argsort(sc)[self._pick_rank(len(sc))]]
            if x is None:
                x = self._fallback_site(env, geo.sp_s[idx])
                self.st["no_site"] += 1
            if x is not None:
                Xs[idx] = x
            todo.remove(idx)
            present[idx] = True
        pos = np.array(pos)
        pos[p.search_idx] = Xs
        return pos, how

    def _sites(self, env, a, vac, vac_sp, inv, cell):
        """Scored non-overlapping candidate sites for species a (vacated sites tabu)."""
        geo = self.geo
        X = geo.candidates(env, a, self.rng)
        sc = geo.score(env, X, a, self.model)
        ok = np.isfinite(sc)
        if not np.any(ok) and len(X):
            # crowded (dense bulk): accept the tighter sites; relaxation opens them up
            sc = geo.score(env, X, a, self.model, overlap=0.65)
            ok = np.isfinite(sc)
        if not np.any(ok):
            return None, None
        X, sc = X[ok], sc[ok]
        vs = vac[vac_sp == a]
        if len(vs):
            dv = X[:, None, :] - vs[None, :, :]
            if geo.bulk:
                f = dv @ inv
                dv = (f - np.round(f)) @ cell
            tabu = np.linalg.norm(dv, axis=-1).min(axis=1) < 0.5 * geo.bond[a, a]
            if np.any(~tabu):
                X, sc = X[~tabu], sc[~tabu]
        return X, sc

    def _mu(self, cur):
        e = np.asarray(cur.ea, float)[self.p.search_idx]
        mu = np.zeros(self.geo.ns)
        for a in np.unique(self.geo.sp_s):
            mu[a] = e[self.geo.sp_s == a].mean()
        return mu

    def _fallback_site(self, env, a):
        geo, rng = self.geo, self.rng
        if env.m == 0:
            return None
        for _ in range(50):
            j = rng.integers(env.m)
            x = env.P[j] + ops.random_unit(rng) * geo.bond[a, env.sp[j]]
            d = np.linalg.norm(env.P - x, axis=1)
            if np.all(d >= self.p.dmin_factor * geo.d0[a, env.sp]):
                return x
        return None

    def _swap(self, cur, pos):
        p, rng, geo = self.p, self.rng, self.geo
        pos = np.array(pos)
        dev = self._dev(cur.ea)
        sp = geo.sp_s
        for _ in range(1 + int(rng.random() < 0.3)):
            i = self._softmax_pick(dev, 1)[0]
            j = self._softmax_pick(dev, 1, mask=sp != sp[i])[0]
            a, b = p.search_idx[i], p.search_idx[j]
            pos[[a, b]] = pos[[b, a]]
        return pos

    # ------------------------------------------------------------------ main
    def _make_move(self, cur, klev):
        rng = self.rng
        pos, cell = np.array(cur.pos), np.array(cur.cell)
        if self.p_strain and rng.random() < self.p_strain:
            pos, cell = ops.strain_cell(pos, cell, rng, self.strain)
        if self.p_swap and rng.random() < self.p_swap:
            self.st["swap"] += 1
            return self._swap(cur, pos), cell, False
        self.st["regrow"] += 1
        k = int(rng.integers(1, klev + 1))
        return self._regrow(cur, pos, cell, k)[0], cell, True

    def _fallback_move(self, cur):
        pos, cell = np.array(cur.pos), np.array(cur.cell)
        try:
            pos = ops.displace(self.p, pos, self.rng, 0.5 * float(self.geo.bond.max()), cell=cell)
        except Exception:
            self.st["errors"] += 1
        return pos, cell, False

    def loop(self):
        ctx, rng, p = self.ctx, self.rng, self.p
        cur = None
        for _ in range(self.n_init):
            m = self._new_random()
            if m is not None and (cur is None or m.e < cur.e):
                cur = m
        if cur is None:
            return
        self.best = cur
        walk_best, stag, klev = cur.e, 0, 1
        while True:
            mv = self._safe(self._make_move, cur, klev)
            pos, cell, grown = mv if mv is not None else self._fallback_move(cur)
            thr = self._accept_threshold(cur)
            new, acc = self._trial(pos, cell, thr, grown)
            self.st["moves"] += 1
            if new is not None and new.e < self.best.e:
                self.best = new
            if acc:
                cur = new
                self.st["accepted"] += 1
            if acc and new.e < walk_best - 1e-3:
                walk_best, stag, klev = new.e, 0, 1
            else:
                stag += 1
                if stag >= self.stag_moves:
                    stag, klev = 0, klev + 1
            if klev > self.kmax:
                self.st["restarts"] += 1
                if self.p_restart_best > 0 and rng.random() < self.p_restart_best:
                    m = self.best
                else:
                    m = self._new_random()
                if m is not None:
                    cur = m
                    if m.e < self.best.e:
                        self.best = m
                walk_best, stag, klev = cur.e, 0, 1
            self.st["klev"] = klev
            ctx.stats.update(self.st)


def run(ctx, rng, **params):
    kw = dict(DEFAULTS)
    kw.update(params)
    _CRG(ctx, rng, **kw).loop()
