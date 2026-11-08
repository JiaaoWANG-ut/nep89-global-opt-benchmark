"""Energy-guided basin hopping (EGBH).

NEP89 is a sum of per-atom (site) energies and ``ctx.atomic_energies()`` returns
them for free after every evaluation. EGBH uses them in two ways:

* WHICH atoms to move: weakly bound atoms, i.e. high site energy relative to the
  median of their own species (so multi-element systems work), are selected with
  a Boltzmann-like probability.
* WHERE to put them: an online, species-resolved site-energy model (ridge
  regression of the NEP site energies of all relaxed structures on radial-basis
  neighbour counts per neighbour element, substrate elements included) scores
  candidate sites at zero ionic-step cost. Candidate sites are generated in a
  dynamic-lattice fashion around the current structure: tetrahedral apex sites
  above every bonded triangle, bridge sites and atop sites of the cluster, of the
  support/framework next to it, or of the periodic crystal. Removed atoms are
  re-inserted one by one at sampled low-energy sites.

Multi-element systems additionally get model-guided exchange moves (pairs of
unlike atoms whose predicted site energies drop most when swapped), and every
system gets diversification moves: energy-weighted shakes (high-energy atoms are
displaced more), twists of half of a cluster, rigid rotations/translations of
supported/confined clusters and cell strains for bulk. Trial relaxations use a
looser force threshold and are aborted once they are close to a minimum that
cannot be accepted; promising minima are re-tightened to the benchmark fmax.
The walk restarts from a random structure after a size-dependent number of
moves without improvement.
"""
import numpy as np
from scipy.spatial import cKDTree

from .. import ops

LABEL = "Energy-guided basin hopping"
FAMILY = "innovative"
REFERENCE = ("Wales & Doye, J. Phys. Chem. A 101, 5111 (1997) [basin hopping]; "
             "Shao, Cheng & Cai, J. Comput. Chem. 25, 1693 (2004) [dynamic lattice searching]; "
             "Takeuchi, J. Chem. Inf. Model. 46, 2066 (2006) [moving the highest-energy atoms "
             "to the best surface sites]; Rossi & Ferrando, J. Phys.: Condens. Matter 21, "
             "084208 (2009) [exchange moves for nanoalloys]; Behler & Parrinello, PRL 98, "
             "146401 (2007) [atom-centred radial descriptors]")
DESCRIPTION = (
    "Basin hopping whose moves are directed by the per-atom energies that NEP89 returns "
    "for free. Weakly bound atoms (site energy above the median of their species) are "
    "removed preferentially and re-inserted at dynamic-lattice candidate sites (apex "
    "sites above bonded triangles, bridge and atop sites of the cluster, the support or "
    "the crystal) ranked by an online species-resolved site-energy model, a ridge "
    "regression of all observed NEP site energies on radial neighbour counts per element "
    "(substrate elements included), which costs no ionic steps. Alloys get model-guided "
    "exchanges of unlike atoms whose predicted site energies drop when swapped. "
    "Energy-weighted shakes, cluster twists, rigid moves of supported/confined clusters "
    "and cell strains diversify the walk. Trial relaxations use a loose fmax, are aborted "
    "near hopeless minima and are re-tightened only near the best energy; the walk "
    "restarts from random structures after size-dependent stagnation. Unlike basin "
    "hopping (random displacement of every atom) or dynamic lattice searching (fixed "
    "pair-potential lattice energies), it learns where atoms of each element want to sit "
    "from the potential's own energy decomposition.")
DEFAULTS = dict(kT=0.1, fmax_trial=0.1, tighten=0.1, abort_de=0.4, abort_fmax=0.3,
                patience=1.5, patience_min=8, w_reloc=0.5, w_swap=0.3, w_shake=0.15,
                w_twist=0.1, w_rigid=0.1, w_strain=0.3, p_k=0.6, kmax_frac=0.25,
                shake=0.2, strain=0.05, sel_temp=0.5, ridge=0.3, imp_tol=0.01, adapt=1.0,
                site_tmul=1.0, reg_margin=-1.0)

# radial basis for the site-energy model (distances in units of r_cov,i + r_cov,j)
MU = np.array([0.85, 1.0, 1.15, 1.3, 1.5])
WID = 0.12
UCUT = 1.75
BOND = 1.3        # bonded if r < BOND * (r_cov,i + r_cov,j)
OVL = 0.8         # candidate site rejected if r < OVL * (r_cov,i + r_cov,j)
KNN = 48
MAXTRI = 4000


def _fc(u):
    return np.where(u < UCUT, 0.5 * (1.0 + np.cos(np.pi * np.minimum(u, UCUT) / UCUT)), 0.0)


def _softmax_choice(rng, score, temp):
    """Index sampled with p ~ exp(score/temp) (higher score = more likely)."""
    z = (np.asarray(score, float) - np.max(score)) / max(temp, 1e-9)
    w = np.exp(z)
    w /= w.sum()
    return int(rng.choice(len(w), p=w))


class _Env:
    __slots__ = ("xyz", "tid", "own", "srch", "home", "tree")


class _State:
    __slots__ = ("pos", "cell", "e", "ea", "aborted")

    def __init__(self, pos, cell, e, ea, aborted):
        self.pos, self.cell, self.e, self.ea, self.aborted = pos, cell, e, ea, aborted


class _Geo:
    """Neighbour environments, candidate sites and site features."""

    def __init__(self, p):
        self.p = p
        types = sorted(set(p.symbols))
        self.nT = len(types)
        self.tid = np.array([types.index(x) for x in p.symbols], int)
        self.rcT = np.array([p.rcov[np.where(self.tid == t)[0][0]] for t in range(self.nT)])
        self.s = np.asarray(p.search_idx, int)
        self.ns = len(self.s)
        self.tid_s = self.tid[self.s]
        self.bulk = bool(p.variable_cell)
        self.rmax = UCUT * 2 * self.rcT.max()
        self.sub_tree = None
        if p.n_sub and not self.bulk:
            cell = np.asarray(p.cell0, float)
            h = 1.0 / np.linalg.norm(np.linalg.inv(cell), axis=0)
            nrep = [int(min(3, max(1, np.ceil(12.0 / h[k])))) if p.pbc[k] else 0
                    for k in range(3)]
            grids = [np.arange(-m, m + 1) for m in nrep]
            sh = np.array(np.meshgrid(*grids, indexing="ij")).reshape(3, -1).T @ cell
            self.sub_xyz = (np.asarray(p.sub_pos)[None] + sh[:, None]).reshape(-1, 3)
            self.sub_own = np.tile(np.arange(p.n_sub), len(sh))
            self.sub_tree = cKDTree(self.sub_xyz)
            self.sub_rmax = p.dmin_factor * (self.rcT[self.tid_s].max()
                                             + self.rcT[self.tid[:p.n_sub]].max())

    # ------------------------------------------------------------ frames
    def home(self, pos, cell):
        """Contiguous search atoms with their centre in the home cell (bulk: wrapped)."""
        p = self.p
        pos = np.array(pos, float)
        if self.bulk:
            f = pos @ np.linalg.inv(cell)
            f -= np.floor(f)
            return f @ cell
        pos = ops.unwrap_search(p, pos, cell)
        if np.any(p.pbc):
            c = pos[self.s].mean(axis=0)
            f = c @ np.linalg.inv(cell)
            pos[self.s] -= np.where(p.pbc, np.floor(f), 0.0) @ cell
        return pos

    def env(self, pos, cell, excl=()):
        """Environment points (explicit periodic images) of all atoms except search atoms excl."""
        s = self.s
        keep = np.ones(self.ns, bool)
        if len(excl):
            keep[np.asarray(list(excl), int)] = False
        idx = s[keep]
        if self.bulk:
            inv = np.linalg.inv(cell)
            h = 1.0 / np.linalg.norm(inv, axis=0)
            m = np.clip(np.ceil(self.rmax / h).astype(int), 1, 3)
            grids = [np.arange(-k, k + 1) for k in m]
            sh = np.array(np.meshgrid(*grids, indexing="ij")).reshape(3, -1).T.astype(float)
            f = pos[idx] @ inv
            F = f[None] + sh[:, None]
            lim = self.rmax / h
            ok = np.all((F > -lim) & (F < 1 + lim), axis=-1)
            xyz = F[ok] @ cell
            own = np.broadcast_to(idx, F.shape[:2])[ok]
            home = np.broadcast_to(np.all(sh == 0, axis=1)[:, None], F.shape[:2])[ok]
            srch = np.ones(len(xyz), bool)
        else:
            xyz, own = pos[idx], idx
            srch = np.ones(len(idx), bool)
            home = np.ones(len(idx), bool)
            if self.sub_tree is not None:
                c = pos[s].mean(axis=0)
                rad = np.sqrt(((pos[s] - c) ** 2).sum(axis=1).max()) + self.rmax + 0.5
                j = np.asarray(self.sub_tree.query_ball_point(c, rad), int)
                if len(j):
                    xyz = np.vstack([xyz, self.sub_xyz[j]])
                    own = np.concatenate([own, self.sub_own[j]])
                    srch = np.concatenate([srch, np.zeros(len(j), bool)])
                    home = np.concatenate([home, np.zeros(len(j), bool)])
        e = _Env()
        e.xyz, e.own, e.srch, e.home = np.asarray(xyz, float), np.asarray(own, int), srch, home
        e.tid = self.tid[e.own]
        e.tree = cKDTree(e.xyz) if len(e.xyz) else None
        return e

    # ------------------------------------------------------------ features
    def feats(self, Q, tq, env):
        """Radial features G (m, nT*nb), min scaled distance and 'bonded to a search atom'."""
        Q = np.atleast_2d(np.asarray(Q, float))
        m = len(Q)
        nb = len(MU)
        if env.tree is None or m == 0:
            return np.zeros((m, self.nT * nb)), np.full(m, 10.0), np.zeros(m, bool)
        k = int(min(KNN, len(env.xyz)))
        d, j = env.tree.query(Q, k=max(k, 2), distance_upper_bound=self.rmax)
        d = d.reshape(m, -1)
        j = j.reshape(m, -1)
        valid = np.isfinite(d) & (d > 1e-6)
        jj = np.where(valid, j, 0)
        tn = env.tid[jj]
        tq = np.broadcast_to(np.asarray(tq, int), (m,))
        u = np.where(valid, d / (self.rcT[tq][:, None] + self.rcT[tn]), 10.0)
        g = np.exp(-((u[..., None] - MU) / WID) ** 2) * _fc(u)[..., None]
        oh = ((tn[..., None] == np.arange(self.nT)) & valid[..., None]).astype(float)
        G = np.einsum("qkt,qkb->qtb", oh, g).reshape(m, -1)
        umin = u.min(axis=1)
        bonded = np.any(valid & env.srch[jj] & (u < BOND), axis=1)
        return G, umin, bonded

    # ------------------------------------------------------------ sites
    def sites(self, env, t_new, center, rng):
        """Candidate positions for an atom of type t_new around the environment."""
        xyz, tid = env.xyz, env.tid
        if len(xyz) == 0:
            return np.zeros((0, 3))
        if self.bulk:
            anc = np.arange(len(xyz))
            key = env.home
        else:
            S = np.where(env.srch)[0]
            U = np.where(~env.srch)[0]
            if len(U) and len(S):
                dsu = np.sqrt(((xyz[U][:, None] - xyz[S][None]) ** 2).sum(-1)).min(axis=1)
                U = U[dsu < 2.0 * (self.rcT[tid[U]] + self.rcT[t_new]) + 0.5]
            elif len(U):
                U = U[:0]
            anc = np.concatenate([S, U]).astype(int)
            key = np.ones(len(xyz), bool)
        A = xyz[anc]
        ta = tid[anc]
        ka = key[anc]
        rt = self.rcT[ta] + self.rcT[t_new]          # target distance to each anchor
        m = len(anc)
        out = []
        D = np.sqrt(((A[:, None] - A[None]) ** 2).sum(-1))
        adj = (D < BOND * (self.rcT[ta][:, None] + self.rcT[ta][None])) & (D > 1e-6)
        iu, ju = np.nonzero(np.triu(adj, 1))
        # --- apex sites above bonded triangles
        if len(iu):
            com = adj[iu] & adj[ju]
            e_i, c = np.nonzero(com)
            sel = c > ju[e_i]
            a, b, c = iu[e_i[sel]], ju[e_i[sel]], c[sel]
            ok = ka[a] | ka[b] | ka[c]
            a, b, c = a[ok], b[ok], c[ok]
            if len(a) > MAXTRI:
                pick = rng.choice(len(a), MAXTRI, replace=False)
                a, b, c = a[pick], b[pick], c[pick]
            if len(a):
                out.append(_apex(A[a], A[b], A[c], rt[a], rt[b], rt[c]))
        # --- bridge sites
        ok = ka[iu] | ka[ju]
        iu, ju = iu[ok], ju[ok]
        if len(iu):
            P, Qb = A[iu], A[ju]
            mid = 0.5 * (P + Qb)
            ax = Qb - P
            dab = np.linalg.norm(ax, axis=1)
            ax /= dab[:, None]
            r = 0.5 * (rt[iu] + rt[ju])
            rho2 = r ** 2 - (0.5 * dab) ** 2
            good = rho2 > (0.4 * r) ** 2
            if self.bulk or center is None:
                v = ops.random_unit(rng, len(iu))
            else:
                v = mid - center + 0.3 * ops.random_unit(rng, len(iu))
            v -= (v * ax).sum(1)[:, None] * ax
            nv = np.linalg.norm(v, axis=1)
            good &= nv > 1e-6
            if good.any():
                v = v[good] / nv[good][:, None]
                out.append(mid[good] + v * np.sqrt(rho2[good])[:, None])
        # --- atop sites of key anchors (away from their bonded neighbours)
        kk = np.where(ka)[0]
        if len(kk):
            nbm = adj[kk].astype(float)
            cnt = nbm.sum(1)
            cen = (nbm @ A) / np.maximum(cnt, 1)[:, None]
            v = A[kk] - cen
            lone = cnt == 0
            if lone.any():
                v[lone] = ops.random_unit(rng, int(lone.sum())).reshape(-1, 3)
            v = v + 0.25 * ops.random_unit(rng, len(kk)).reshape(-1, 3) \
                * np.linalg.norm(v, axis=1)[:, None]
            nv = np.linalg.norm(v, axis=1)
            good = nv > 1e-6
            if good.any():
                out.append(A[kk][good] + v[good] / nv[good][:, None] * rt[kk][good][:, None])
        if not out:
            return np.zeros((0, 3))
        return np.vstack(out)


def _apex(P1, P2, P3, r1, r2, r3):
    """Both trilateration solutions (points at distances r1,r2,r3 from P1,P2,P3)."""
    ex = P2 - P1
    d = np.linalg.norm(ex, axis=1)
    ex = ex / d[:, None]
    v3 = P3 - P1
    i = (ex * v3).sum(1)
    ey = v3 - i[:, None] * ex
    j = np.linalg.norm(ey, axis=1)
    good = j > 1e-3
    ey = ey / np.where(good, j, 1.0)[:, None]
    ez = np.cross(ex, ey)
    x = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    jj = np.where(good, j, 1.0)
    y = (r1 ** 2 - r3 ** 2 + i ** 2 + j ** 2) / (2 * jj) - (i / jj) * x
    z2 = r1 ** 2 - x ** 2 - y ** 2
    good &= z2 > 0.04 * r1 ** 2
    if not good.any():
        return np.zeros((0, 3))
    z = np.sqrt(np.maximum(z2[good], 0.0))[:, None]
    base = P1[good] + x[good][:, None] * ex[good] + y[good][:, None] * ey[good]
    return np.vstack([base + z * ez[good], base - z * ez[good]])


class _Model:
    """Online ridge regression of NEP site energies, one weight vector per search element."""

    def __init__(self, geo, lam):
        self.D = 1 + geo.nT * len(MU)
        self.types = sorted(set(geo.tid_s.tolist()))
        prior = np.zeros(self.D)
        pr = np.zeros((geo.nT, len(MU)))
        pr[:, MU <= 1.16] = -0.3      # before data: more close neighbours = lower energy
        prior[1:] = pr.ravel()
        self.prior = prior
        self.lam = np.full(self.D, float(lam))
        self.lam[0] = 1e-8
        self.A = {t: np.zeros((self.D, self.D)) for t in self.types}
        self.b = {t: np.zeros(self.D) for t in self.types}
        self.w = {t: prior.copy() for t in self.types}
        self.mse = {t: 0.04 for t in self.types}
        self.n = {t: 0 for t in self.types}

    def predict(self, G, t):
        w = self.w[t]
        return w[0] + G @ w[1:]

    def add(self, G, tq, y):
        for t in self.types:
            m = tq == t
            if not m.any():
                continue
            X = np.hstack([np.ones((int(m.sum()), 1)), G[m]])
            if self.n[t] > 0:
                r = X @ self.w[t] - y[m]
                self.mse[t] = 0.8 * self.mse[t] + 0.2 * float(np.mean(r ** 2))
            self.A[t] += X.T @ X
            self.b[t] += X.T @ y[m]
            self.n[t] += int(m.sum())
            try:
                self.w[t] = np.linalg.solve(self.A[t] + np.diag(self.lam),
                                            self.b[t] + self.lam * self.prior)
            except np.linalg.LinAlgError:
                pass

    def rmse(self, t):
        return float(np.sqrt(self.mse[t]))


class _EGBH:
    def __init__(self, ctx, rng, kw):
        self.ctx, self.rng, self.p = ctx, rng, ctx.p
        self.kw = kw
        p = self.p
        self.geo = _Geo(p)
        self.model = _Model(self.geo, kw["ridge"])
        self.s = self.geo.s
        self.ns = self.geo.ns
        self.bulk = bool(p.variable_cell)
        self.multi = len(set(p.search_symbols)) > 1
        self.patience = max(int(kw["patience_min"]), int(round(kw["patience"] * self.ns)))
        self.bond = 2.0 * float(np.mean(p.rcov[self.s]))
        self.step = kw["shake"] * self.bond
        w = {"reloc": kw["w_reloc"], "shake": kw["w_shake"]}
        if self.multi:
            w["swap"] = kw["w_swap"]
        if not self.bulk and self.ns >= 5:
            w["twist"] = kw["w_twist"]
        if p.category in ("surface", "zeolite") and not self.bulk:
            w["rigid"] = kw["w_rigid"]
        if self.bulk:
            w["strain"] = kw["w_strain"]
        self.moves = list(w)
        pw = np.array([w[k] for k in self.moves], float)
        self.pmove = pw / pw.sum()
        # tries, accepted, improved, steps, new global bests
        self.st = {k: [0, 0, 0, 0, 0] for k in self.moves}
        self.cred = {k: 0.0 for k in self.moves}
        self.n_ep = 0
        self.n_fallback = 0
        self.n_abort = 0
        self.reg = None
        r = getattr(p, "region", None) or {}
        if kw["reg_margin"] >= 0 and p.category in ("surface", "zeolite") and not self.bulk:
            if r.get("type") == "sphere":
                self.reg = ("sphere", np.asarray(r["center"], float),
                            float(r["radius"]) + kw["reg_margin"])
            elif r.get("type") == "box" and p.category == "surface":
                self.reg = ("zrange", float(r["lo"][2]) - kw["reg_margin"],
                            float(r["hi"][2]) + kw["reg_margin"])

    # ------------------------------------------------------------ relax + learn
    def relax(self, pos, cell, e_walk=None):
        ctx, p, kw = self.ctx, self.p, self.kw
        abort = None
        if e_walk is not None and kw["abort_de"] > 0:
            lim = e_walk + kw["abort_de"]
            fab = kw["abort_fmax"]

            def abort(i, e, fm):
                return i >= 2 and fm < fab and e > lim
        fm_trial = max(kw["fmax_trial"], p.fmax)
        r = ctx.relax(pos, cell, fmax=fm_trial, abort=abort)
        if r.aborted:
            self.n_abort += 1
        elif fm_trial > p.fmax and r.e < ctx.best_e + kw["tighten"]:
            r = ctx.relax(r.pos, r.cell)
        ea = ctx.atomic_energies()
        if ea is None or not np.all(np.isfinite(ea)) or r.e > 1e5:
            ea_s = np.zeros(self.ns)
        else:
            ea_s = np.asarray(ea, float)[self.s]
            try:
                self.learn(r.pos, r.cell, ea_s)
            except Exception:
                pass
        return _State(np.array(r.pos, float), np.array(r.cell, float), float(r.e), ea_s,
                      bool(r.aborted))

    def learn(self, pos, cell, ea_s):
        pos = self.geo.home(pos, cell)
        env = self.geo.env(pos, cell)
        G, _, _ = self.geo.feats(pos[self.s], self.geo.tid_s, env)
        self.model.add(G, self.geo.tid_s, ea_s)

    # ------------------------------------------------------------ helpers
    def in_reg(self, X, cell):
        """Candidate sites inside the (margin-extended) generation region."""
        if self.reg is None:
            return np.ones(len(X), bool)
        if self.reg[0] == "zrange":
            return (X[:, 2] >= self.reg[1]) & (X[:, 2] <= self.reg[2])
        v = X - self.reg[1]
        pbc = np.asarray(self.p.pbc, bool)
        if pbc.any():
            f = v @ np.linalg.inv(cell)
            f -= np.where(pbc, np.round(f), 0.0)
            v = f @ cell
        return np.sqrt((v ** 2).sum(1)) <= self.reg[2]

    def delta(self, ea_s):
        """Site energy above the median of the atom's species."""
        d = np.zeros(self.ns)
        for t in set(self.geo.tid_s.tolist()):
            m = self.geo.tid_s == t
            d[m] = ea_s[m] - np.median(ea_s[m])
        return d

    def pick_atoms(self, ea_s, k):
        d = self.delta(ea_s)
        sd = float(np.std(d))
        temp = max(self.kw["sel_temp"] * sd, 1e-3)
        z = (d - d.max()) / temp
        w = np.exp(z)
        w = w / w.sum()
        nz = int((w > 0).sum())
        k = min(k, nz, self.ns)
        return rng_choice(self.rng, self.ns, k, w), d

    def repair(self, pos, cell, iters=40):
        p = self.p
        geo = self.geo
        if geo.sub_tree is None:
            return ops.repair_overlaps(p, pos, cell)
        pos = np.array(pos, float)
        s = self.s
        rc = p.rcov
        lim_ss = p.dmin_factor * (rc[s][:, None] + rc[s][None])
        for _ in range(iters):
            X = pos[s]
            moved = False
            dv = X[:, None] - X[None]
            d = np.sqrt((dv ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            for a, b in zip(*np.nonzero(np.triu(d < lim_ss, 1))):
                v = dv[a, b]
                nv = d[a, b]
                v = v / nv if nv > 1e-8 else ops.random_unit(self.rng)
                push = 0.5 * (lim_ss[a, b] - min(nv, lim_ss[a, b])) + 0.02
                pos[s[a]] += v * push
                pos[s[b]] -= v * push
                moved = True
            hits = geo.sub_tree.query_ball_point(pos[s], geo.sub_rmax)
            for a, h in enumerate(hits):
                for j in h:
                    v = pos[s[a]] - geo.sub_xyz[j]
                    nv = float(np.linalg.norm(v))
                    lim = p.dmin_factor * (rc[s[a]] + rc[geo.sub_own[j]])
                    if nv < lim:
                        v = v / nv if nv > 1e-8 else np.array([0.0, 0.0, 1.0])
                        pos[s[a]] += v * (lim - nv + 0.02)
                        moved = True
            if not moved:
                break
        return pos

    def reseat(self, pos, cell):
        """Re-seat detached search atoms (non-bulk) on the cluster surface."""
        p = self.p
        if self.bulk:
            return pos
        env = self.geo.env(pos, cell)
        X = pos[self.s]
        # attached if any neighbour (search or substrate) within 1.6 * rcov sum
        d, j = env.tree.query(X, k=min(8, len(env.xyz)), distance_upper_bound=self.geo.rmax)
        d = np.atleast_2d(d).reshape(len(X), -1)
        j = np.atleast_2d(j).reshape(len(X), -1)
        valid = np.isfinite(d) & (d > 1e-6)
        jj = np.where(valid, j, 0)
        lim = 1.6 * (self.geo.rcT[self.geo.tid_s][:, None] + self.geo.rcT[env.tid[jj]])
        att = np.any(valid & (d < lim), axis=1)
        lost = self.s[~att]
        if len(lost) and len(lost) < self.ns:
            pos = ops.relocate_atoms(p, pos, self.rng, lost, cell)
        return pos

    # ------------------------------------------------------------ moves
    def mv_reloc(self, cur):
        rng, geo = self.rng, self.geo
        cell = cur.cell
        pos = geo.home(cur.pos, cur.cell)
        kmax = max(1, int(round(self.kw["kmax_frac"] * self.ns)))
        k = int(min(kmax, rng.geometric(self.kw["p_k"])))
        if self.ns - k < 1:
            k = max(1, self.ns - 1)
        if self.ns < 2:
            return None
        rem, d = self.pick_atoms(cur.ea, k)
        rem = sorted(rem, key=lambda i: -d[i])
        pending = set(int(i) for i in rem)
        new = pos.copy()
        for i in rem:
            i = int(i)
            t = int(geo.tid_s[i])
            env = geo.env(new, cell, excl=pending)
            keep = np.array([q not in pending for q in range(self.ns)])
            center = new[self.s][keep].mean(axis=0) if keep.any() else None
            X = geo.sites(env, t, center, rng)
            placed = False
            if len(X):
                G, umin, bonded = geo.feats(X, t, env)
                ok = umin >= OVL
                if not self.bulk:
                    ok &= bonded
                far = np.linalg.norm(X - pos[self.s[i]], axis=1) > 0.6 * self.bond
                ok &= far
                ok &= self.in_reg(X, cell)
                if ok.any():
                    X, G = X[ok], G[ok]
                    E = self.model.predict(G, t)
                    temp = max(self.model.rmse(t), 0.05) * self.kw["site_tmul"]
                    j = _softmax_choice(rng, -E, temp)
                    new[self.s[i]] = X[j]
                    placed = True
            if not placed:
                self.n_fallback += 1
                if self.bulk:
                    res = self.p.place(new, [self.s[i]], rng, cell)
                    if res is None:
                        return None
                    new = res
                else:
                    res = ops.relocate_atoms(self.p, new, rng, [self.s[i]], cell)
                    if res is None:
                        return None
                    new = geo.home(res, cell)
            pending.discard(i)
        return new, cell

    def mv_swap(self, cur):
        rng, geo = self.rng, self.geo
        cell = cur.cell
        pos = geo.home(cur.pos, cur.cell)
        env = geo.env(pos, cell)
        X = pos[self.s]
        ts = geo.tid_s
        types = self.model.types
        Eh = np.zeros((max(types) + 1, self.ns))
        for t in types:
            G, _, _ = geo.feats(X, t, env)
            Eh[t] = self.model.predict(G, t)
        e = cur.ea
        # dE[i, j]: predicted site-energy change when atoms i and j exchange positions
        dE = (Eh[ts[None, :], np.arange(self.ns)[:, None]] - e[:, None]
              + Eh[ts[:, None], np.arange(self.ns)[None, :]] - e[None, :])
        unlike = ts[:, None] < ts[None, :]
        unlike |= (ts[:, None] > ts[None, :])
        ii, jj = np.nonzero(np.triu(unlike, 1))
        if not len(ii):
            return None
        temp = max(np.mean([self.model.rmse(t) for t in types]) * 2, 0.05)
        nsw = 1 + int(rng.random() < 0.25)
        new = pos.copy()
        used = set()
        for _ in range(nsw):
            okp = np.array([(a not in used) and (b not in used) for a, b in zip(ii, jj)])
            if not okp.any():
                break
            ci, cj = ii[okp], jj[okp]
            q = _softmax_choice(rng, -dE[ci, cj], temp)
            a, b = self.s[ci[q]], self.s[cj[q]]
            new[[a, b]] = new[[b, a]]
            used.update((int(ci[q]), int(cj[q])))
        return new, cell

    def mv_shake(self, cur, scale=1.0, strain=0.0):
        rng = self.rng
        cell = np.array(cur.cell, float)
        pos = self.geo.home(cur.pos, cur.cell)
        d = self.delta(cur.ea)
        rank = np.argsort(np.argsort(d)) / max(self.ns - 1, 1)
        amp = self.step * scale * (0.3 + 0.7 * rank)
        r = amp * rng.random(self.ns) ** (1 / 3)
        pos[self.s] += ops.random_unit(rng, self.ns) * r[:, None]
        if strain > 0 and self.bulk:
            pos, cell = ops.strain_cell(pos, cell, rng, strain)
        pos = self.repair(pos, cell)
        return self.reseat(pos, cell), cell

    def mv_strain(self, cur):
        return self.mv_shake(cur, scale=0.6, strain=self.kw["strain"])

    def mv_twist(self, cur):
        rng = self.rng
        cell = cur.cell
        pos = self.geo.home(cur.pos, cur.cell)
        X = pos[self.s]
        c = X.mean(axis=0)
        nv = np.array([0.0, 0.0, 1.0]) if self.p.category == "surface" else ops.random_unit(rng)
        proj = (X - c) @ nv
        thr = np.quantile(proj, rng.uniform(0.35, 0.65))
        up = proj > thr
        if up.sum() < 1 or up.all():
            return None
        R = ops.rotation_about(nv, rng.uniform(0.25, 1.75) * np.pi)
        X[up] = (X[up] - c) @ R.T + c
        pos[self.s] = X
        pos = self.repair(pos, cell)
        return self.reseat(pos, cell), cell

    def mv_rigid(self, cur):
        rng = self.rng
        cell = cur.cell
        pos = self.geo.home(cur.pos, cur.cell)
        X = pos[self.s]
        c = X.mean(axis=0)
        if self.p.category == "surface":
            R = ops.rotation_about([0, 0, 1], rng.uniform(0, 2 * np.pi))
            phi = rng.uniform(0, 2 * np.pi)
            t = np.array([np.cos(phi), np.sin(phi), 0.0]) * rng.uniform(0, 1.0) * self.bond
        else:
            R = ops.random_rotation(rng)
            t = ops.random_unit(rng) * rng.uniform(0, 0.5) * self.bond
        pos[self.s] = (X - c) @ R.T + c + t
        pos = self.repair(pos, cell)
        return self.reseat(pos, cell), cell

    # ------------------------------------------------------------ main loop
    def new_start(self):
        try:
            return self.p.random_structure(self.rng)
        except Exception:
            if self.ctx.best_pos is not None:
                cur = _State(self.ctx.best_pos, self.ctx.best_cell, 0.0, np.zeros(self.ns), False)
                return self.mv_shake(cur, scale=3.0)
            raise

    def move_probs(self):
        """Base move weights rescaled by their observed improvement rate per ionic step."""
        a = float(self.kw["adapt"])
        if a <= 0:
            return self.pmove
        r = np.array([(self.cred[k] + 1.0) / (self.st[k][3] + 100.0) for k in self.moves])
        m = np.clip(r / r.mean(), 0.25, 4.0) ** a
        p = self.pmove * m
        return p / p.sum()

    def episode(self):
        rng, kw = self.rng, self.kw
        tol = kw["imp_tol"]
        pos, cell = self.new_start()
        cur = self.relax(pos, cell, None)
        self.n_ep += 1
        best = cur.e
        stall = 0
        while stall < self.patience:
            k = self.moves[int(rng.choice(len(self.moves), p=self.move_probs()))]
            try:
                out = getattr(self, "mv_" + k)(cur)
            except Exception:
                out = None
            if out is None:
                k = "shake"
                out = self.mv_shake(cur)
            n0 = self.ctx.nsteps
            gb0 = self.ctx.best_e
            trial = self.relax(out[0], out[1], cur.e)
            st = self.st[k]
            st[0] += 1
            st[3] += self.ctx.nsteps - n0
            de = trial.e - cur.e
            if not trial.aborted and de < -tol:
                st[2] += 1
                self.cred[k] += min(1.0, (-de - tol) / kw["kT"])
            if self.ctx.best_e < gb0 - tol:
                st[4] += 1
            if not trial.aborted and (de < 0 or rng.random() < np.exp(-de / kw["kT"])):
                cur = trial
                st[1] += 1
            if not trial.aborted and trial.e < best - tol:
                best = trial.e
                stall = 0
            else:
                stall += 1
            self.ctx.stats.update(episodes=self.n_ep, aborts=self.n_abort,
                                  fallbacks=self.n_fallback,
                                  moves={m: list(v) for m, v in self.st.items()},
                                  pmove=[round(float(x), 3) for x in self.move_probs()],
                                  rmse={str(t): round(self.model.rmse(t), 4)
                                        for t in self.model.types})


def rng_choice(rng, n, k, w):
    return rng.choice(n, size=k, replace=False, p=w)


def run(ctx, rng, **params):
    kw = dict(DEFAULTS)
    kw.update(params)
    alg = _EGBH(ctx, rng, kw)
    while True:
        alg.episode()
