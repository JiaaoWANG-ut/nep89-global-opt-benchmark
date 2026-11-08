"""Symmetry-seeded search (SYMM): symmetric random seeds + bandit-balanced basin hopping."""
import numpy as np
from scipy.sparse.csgraph import connected_components

from .. import ops
from ..context import BudgetExhausted
from . import _symm_groups as SG

LABEL = "Symmetry-seeded search"
FAMILY = "innovative"
REFERENCE = ("Pickard & Needs, J. Phys.: Condens. Matter 23, 053201 (2011) [AIRSS]; "
             "Fredericks, Parrish, Sayre & Zhu, Comput. Phys. Commun. 261, 107810 (2021) [PyXtal]; "
             "Wang, Lv, Zhu & Ma, Phys. Rev. B 82, 094116 (2010); Comput. Phys. Commun. 183, "
             "2063 (2012) [CALYPSO]; Wales & Doye, J. Phys. Chem. A 101, 5111 (1997) [BH]")
DESCRIPTION = (
    "Two-armed search whose exploration arm draws symmetric random structures from "
    "self-written group machinery (no spglib/pyxtal): gas-phase clusters get a random "
    "point group (Cs ... Ih, plus a C1 share) whose orbit classes (centre, rotation axes, "
    "mirror planes, general position) are filled by a bounded-knapsack partition of the "
    "exact composition, then rescaled to bonding distances; supported clusters use a "
    "vertical-axis group (Cn/Cnv) whose axis passes through a random top/bridge/hollow "
    "site, mirror aligned with a surface bond, and are dropped analytically onto the slab; "
    "zeolite clusters are symmetric motifs placed inside the confinement sphere and "
    "checked against the framework; bulk seeds fill Wyckoff orbits of one of 37 space "
    "groups (numerically enumerated site symmetries, conventional or primitive cell) and "
    "are replicated with a random near-isotropic Hermite-normal-form supercell. For "
    "multi-species clusters 30% of symmetric seeds are 'skeletons': the geometry is "
    "symmetric for the total atom count and species are decorated afterwards (core-shell "
    "by covalent radius, random, or greedy swaps maximising unlike-neighbour "
    "electronegativity contrast, always used for ionic compositions). A per-run bandit "
    "allocates seeds between symmetric, C1 motif and plain random seeds by their rate of "
    "good minima per ionic step. The "
    "exploitation arm is a Metropolis basin-hopping walker (adaptive displacement, cell "
    "strain, species swaps and, for gas-phase clusters, a 'symmetrise' move that detects "
    "the approximate point group of the current minimum and projects onto it) that jumps "
    "to better seeds and restarts from a fresh seed on stagnation; the seed/hop share is "
    "set by discounted probability matching on the rate of new best or new elite minima "
    "per ionic step.")
DEFAULTS = dict(kT=0.1, step=0.5, target_accept=0.5, adapt_every=10, strain=0.05,
                p_min=0.2, gamma=0.97, n_init=3, patience=40, p_swap=0.3, p_symz=0.15,
                p_c1=0.0, p_type_min=0.1, alpha=0.5, noise=0.05, pool=8)

# Pauling electronegativities (skeleton decoration: unlike-neighbour preference)
_CHI = dict(H=2.20, Li=0.98, Be=1.57, B=2.04, C=2.55, N=3.04, O=3.44, F=3.98, Na=0.93,
            Mg=1.31, Al=1.61, Si=1.90, P=2.19, S=2.58, Cl=3.16, K=0.82, Ca=1.00, Sc=1.36,
            Ti=1.54, V=1.63, Cr=1.66, Mn=1.55, Fe=1.83, Co=1.88, Ni=1.91, Cu=1.90, Zn=1.65,
            Ga=1.81, Ge=2.01, As=2.18, Se=2.55, Br=2.96, Rb=0.82, Sr=0.95, Y=1.22, Zr=1.33,
            Nb=1.60, Mo=2.16, Ru=2.20, Rh=2.28, Pd=2.20, Ag=1.93, Cd=1.69, In=1.78, Sn=1.96,
            Sb=2.05, Te=2.10, I=2.66, Cs=0.79, Ba=0.89, La=1.10, Ce=1.12, Hf=1.30, Ta=1.50,
            W=2.36, Re=1.90, Os=2.20, Ir=2.20, Pt=2.28, Au=2.54, Hg=2.00, Pb=2.33, Bi=2.02)

_SH = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float)
_SH0 = 13  # index of the zero shift


def _unique(P, tol=1e-3):
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    return P[~np.tril(D < tol, -1).any(axis=1)]


def _unique_frac(F, tol=1e-4):
    F = F % 1.0
    d = F[:, None] - F[None]
    d -= np.round(d)
    keep = ~np.tril(np.abs(d).max(axis=-1) < tol, -1).any(axis=1)
    return F[keep] % 1.0


def _reduce(B):
    """Greedy pairwise (Lagrange-like) basis reduction, right-handed result."""
    B = np.array(B, float)
    for _ in range(60):
        changed = False
        for i in range(3):
            for j in range(3):
                if i != j:
                    mu = round(float(B[i] @ B[j]) / float(B[j] @ B[j]))
                    if mu:
                        B[i] -= mu * B[j]
                        changed = True
        if not changed:
            break
    if np.linalg.det(B) < 0:
        B = -B
    return B


def _heights(B):
    v = abs(np.linalg.det(B))
    return v / np.linalg.norm(np.cross(B[[1, 2, 0]], B[[2, 0, 1]]), axis=1)


def _rotz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _hnfs(k):
    out = []
    for a in range(1, k + 1):
        if k % a:
            continue
        for c in range(1, k // a + 1):
            if (k // a) % c:
                continue
            f = k // (a * c)
            for b in range(a):
                for d in range(a):
                    for e in range(c):
                        out.append(np.array([[a, 0, 0], [b, c, 0], [d, e, f]], float))
    return out


def run(ctx, rng, **params):
    prm = dict(DEFAULTS)
    prm.update(params)
    _Symm(ctx, rng, prm).loop()


class _Symm:
    def __init__(self, ctx, rng, prm):
        self.ctx, self.rng, self.prm = ctx, rng, prm
        p = self.p = ctx.p
        s = self.s = p.search_idx
        sym = np.array(p.symbols)
        self.species = sorted(set(sym[s]))
        self.loc = [np.where(sym[s] == x)[0] for x in self.species]  # local (search-order) indices
        self.counts = [len(x) for x in self.loc]
        self.ns = len(s)
        self.rsp = np.array([p.rcov[s[li[0]]] for li in self.loc])
        self.rs = p.rcov[s]
        self.rbar = float(self.rs.mean())
        self.multi = len(self.species) > 1
        chi = np.array([_CHI.get(x, 1.8) for x in self.species])
        self.chi_w = (chi[:, None] - chi[None, :]) ** 2
        self.ionic = bool(self.chi_w.max() >= 1.0)
        cat = p.category
        if p.variable_cell:
            cat = "bulk"
        elif cat not in ("cluster", "surface", "zeolite"):
            cat = "cluster" if p.n_sub == 0 else (
                "zeolite" if p.region.get("type") == "sphere" else "surface")
        self.cat = cat
        reg = p.region
        if reg.get("type") == "sphere":
            self.center = np.asarray(reg["center"], float)
        elif reg.get("type") == "box":
            self.center = 0.5 * (np.asarray(reg["lo"], float) + np.asarray(reg["hi"], float))
        else:
            self.center = 0.5 * np.ones(3) @ p.cell0
        self.vpa = float(p.vpa) if p.vpa else float(4.19 * self.rbar ** 3 / 0.45)
        self._feas, self._bopts, self._sites_cache, self._surf, self._zeo = {}, {}, {}, None, None
        self._hnf = {}
        self.st = dict(n_seed=0, n_hop=0, n_swap=0, n_symz=0, seed_fail=0, restarts=0)
        self.grp_count = {}

    # ----------------------------------------------------------- utilities
    def base(self):
        pos = np.zeros((self.p.n, 3))
        pos[: self.p.n_sub] = self.p.sub_pos
        return pos

    def put(self, X, lab):
        """Search-order coordinates from motif rows X with species labels lab."""
        out = np.zeros((self.ns, 3))
        for si, li in enumerate(self.loc):
            out[li] = X[lab == si]
        return out

    def fallback(self):
        self.st["seed_fail"] += 1
        for _ in range(5):
            try:
                pos, cell = self.p.random_structure(self.rng)
                return pos, cell, "rand"
            except RuntimeError:
                continue
        raise RuntimeError("symm: could not build any structure")

    def pick_pg(self, names):
        key = tuple(names)
        if key not in self._feas:
            self._feas[key] = [g for g in names
                               if SG.feasible(self.counts, SG.point_group(g).sites)]
        feas = self._feas[key]
        if not feas or self.rng.random() < self.prm["p_c1"]:
            return "C1"
        return feas[self.rng.integers(len(feas))]

    # ----------------------------------------------------- cluster motifs
    def _sample_pg(self, site, R):
        rng = self.rng
        d = site["dim"]
        if d == 0:
            return np.zeros(3)
        if d == 1:
            t = rng.uniform(0.0, R) if site["sym"] else rng.uniform(-R, R)
            return t * site["basis"][0]
        if d == 2:
            r = R * np.sqrt(rng.random())
            a = rng.uniform(0, 2 * np.pi)
            return r * np.cos(a) * site["basis"][0] + r * np.sin(a) * site["basis"][1]
        return SG.random_unit(rng) * R * rng.random() ** (1 / 3)

    def motif(self, gname, aniso=(0.6, 1.6), skeleton=False):
        """Symmetric cluster (search order, centred at 0, principal axis z) or None.

        skeleton: the geometry is symmetric for the total atom count and the species
        are decorated afterwards (radially sorted by covalent radius or at random).
        """
        rng = self.rng
        pg = SG.point_group(gname)
        counts = [self.ns] if skeleton else self.counts
        plan = SG.partition(counts, pg.sites, rng, alpha=self.prm["alpha"])
        if plan is None:
            return None
        R = self.rbar * (self.ns / 0.74) ** (1 / 3) * rng.uniform(0.9, 1.3)
        plan.sort(key=lambda t: (pg.sites[t[1]]["dim"] > 0, -pg.sites[t[1]]["mult"]))
        X = np.zeros((self.ns, 3))
        lab = np.zeros(self.ns, int)
        m0 = 0
        rsp = np.array([self.rbar]) if skeleton else self.rsp
        for si, k in plan:
            site = pg.sites[k]
            m = site["mult"]
            for _ in range(60):
                x = self._sample_pg(site, R)
                orb = _unique(pg.ops @ x)
                if len(orb) != m:
                    continue
                if m > 1 and np.linalg.norm(orb[:, None] - orb[None], axis=-1)[
                        np.triu_indices(m, 1)].min() < 0.75 * 2 * rsp[si]:
                    continue
                if m0 and (np.linalg.norm(orb[:, None] - X[None, :m0], axis=-1)
                           < 0.75 * (rsp[si] + rsp[lab[:m0]])[None, :]).any():
                    continue
                break
            else:
                return None
            X[m0:m0 + m] = orb
            lab[m0:m0 + m] = si
            m0 += m
        axial = pg.order > 1 and gname not in ("T", "Td", "Th", "O", "Oh", "I", "Ih", "Ci")
        if axial and aniso is not None and rng.random() < 0.5:
            X[:, 2] *= np.exp(rng.uniform(np.log(aniso[0]), np.log(aniso[1])))
        D = np.linalg.norm(X[:, None] - X[None], axis=-1)
        np.fill_diagonal(D, np.inf)
        ratio = D / (rsp[lab][:, None] + rsp[lab][None, :])
        sc = rng.uniform(0.92, 1.02) / np.median(ratio.min(axis=1))
        if ratio.min() * sc < 0.7:
            return None
        if self.ns > 1 and connected_components(ratio * sc < 1.3, directed=False)[0] > 1:
            return None
        X = X * sc
        if skeleton:
            base = np.concatenate([np.full(c, si) for si, c in enumerate(self.counts)])
            u = rng.random()
            if self.ionic or u < 0.25:
                lab = self.decorate_mix(ratio * sc < 1.3, rng.permutation(base))
            elif u < 0.75:
                # smaller atoms inside (core-shell prior)
                inner = np.argsort(np.linalg.norm(X, axis=1) + rng.normal(0, 0.05, self.ns))
                lab = np.empty(self.ns, int)
                lab[inner] = np.concatenate([np.full(self.counts[si], si)
                                             for si in np.argsort(self.rsp, kind="stable")])
            else:
                lab = rng.permutation(base)
        return self.put(X, lab)

    def decorate_mix(self, A, lab):
        """Greedy swaps maximising sum over bonded pairs of (chi_a - chi_b)^2."""
        rng, W, n = self.rng, self.chi_w, self.ns
        A = np.asarray(A, float)
        np.fill_diagonal(A, 0.0)
        for _ in range(6 * n):
            i, j = rng.integers(n, size=2)
            a, b = lab[i], lab[j]
            if a == b:
                continue
            wi = A[i].copy()
            wi[j] = 0.0
            wj = A[j].copy()
            wj[i] = 0.0
            g = wi @ (W[b, lab] - W[a, lab]) + wj @ (W[a, lab] - W[b, lab])
            if g > 1e-9 or (g > -1e-9 and rng.random() < 0.5):
                lab[i], lab[j] = b, a
        return lab

    def count(self, tag):
        self.grp_count[tag] = self.grp_count.get(tag, 0) + 1

    def rand_structure(self):
        for _ in range(5):
            try:
                pos, cell = self.p.random_structure(self.rng)
                return pos, cell, "rand"
            except RuntimeError:
                continue
        raise RuntimeError("symm: could not build any structure")

    def _cluster_motif(self, names, kind, aniso=(0.6, 1.6)):
        if kind == "c1":
            return self.motif("C1"), "C1"
        g = self.pick_pg(names)
        skel = self.multi and g != "C1" and self.rng.random() < 0.3
        return self.motif(g, aniso=aniso, skeleton=skel), g + ("k" if skel else "")

    def seed_cluster(self, kind="sym"):
        p, rng = self.p, self.rng
        for _ in range(30):
            X, g = self._cluster_motif(SG.CLUSTER_GROUPS, kind)
            if X is None:
                continue
            pos = self.base()
            pos[self.s] = X @ SG.random_rotation(rng).T + self.center \
                + rng.normal(0, self.prm["noise"], X.shape)
            if p.min_dist_ok(pos, p.cell0):
                return pos, p.cell0.copy(), g
        return self.fallback()

    # ---------------------------------------------------- supported clusters
    def surf_info(self):
        if self._surf is None:
            p = self.p
            sub = p.sub_pos
            ztop = sub[:, 2].max()
            a1, a2 = p.cell0[0], p.cell0[1]
            lat = [i * a1 + j * a2 for i in (-1, 0, 1) for j in (-1, 0, 1)]
            near = np.where(sub[:, 2] > ztop - 4.0)[0]
            P = np.concatenate([sub[near] + t for t in lat])
            rP = np.tile(p.rcov[near], len(lat))
            top = np.where(sub[:, 2] > ztop - 1.0)[0]
            T = sub[top]
            reg = p.region
            if reg.get("type") == "box":
                lo, hi = np.asarray(reg["lo"]), np.asarray(reg["hi"])
                inside = np.all((T[:, :2] >= lo[:2]) & (T[:, :2] <= hi[:2]), axis=1)
                Tin = T[inside] if inside.any() else T
            else:
                Tin = T
            Timg = np.concatenate([T + t for t in lat])
            self._surf = dict(ztop=ztop, P=P, rP=rP, Tin=Tin, Timg=Timg)
        return self._surf

    def surface_site(self):
        info, rng = self.surf_info(), self.rng
        T, Ti = info["Tin"], info["Timg"]
        x = T[rng.integers(len(T))]
        d = np.linalg.norm(Ti[:, :2] - x[:2], axis=1)
        order = np.argsort(d)
        order = order[d[order] > 0.3]
        if len(order) < 2:
            return x[:2].copy(), rng.uniform(0, 2 * np.pi)
        n1 = Ti[order[0]]
        u = rng.random()
        if u < 0.4:
            site = x[:2].copy()
        elif u < 0.7:
            site = 0.5 * (x[:2] + n1[:2])
        else:
            mid = 0.5 * (x[:2] + n1[:2])
            dm = np.linalg.norm(Ti[:, :2] - mid, axis=1)
            dm[np.linalg.norm(Ti[:, :2] - x[:2], axis=1) < 0.3] = np.inf
            dm[np.linalg.norm(Ti[:, :2] - n1[:2], axis=1) < 0.3] = np.inf
            n2 = Ti[np.argmin(dm)]
            a, b, c = x[:2], n1[:2], n2[:2]
            den = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
            if abs(den) < 1e-6:
                site = mid
            else:
                aa, bb, cc = a @ a, b @ b, c @ c
                site = np.array([aa * (b[1] - c[1]) + bb * (c[1] - a[1]) + cc * (a[1] - b[1]),
                                 aa * (c[0] - b[0]) + bb * (a[0] - c[0]) + cc * (b[0] - a[0])]) / den
                if np.linalg.norm(site - mid) > 2.0 * np.linalg.norm(b - a):
                    site = mid
        v = n1[:2] - x[:2]
        phi = np.arctan2(v[1], v[0]) if rng.random() < 0.5 else rng.uniform(0, 2 * np.pi)
        return site, phi

    def drop(self, X, c):
        """Translate X vertically to contact (c * rcov sum) with the substrate."""
        info = self.surf_info()
        P, rP = info["P"], info["rP"]
        rho2 = ((X[:, None, :2] - P[None, :, :2]) ** 2).sum(-1)
        need = (c * (self.rs[:, None] + rP[None, :])) ** 2 - rho2
        mask = need > 0
        if not mask.any():
            return X + [0, 0, info["ztop"] + c * 2 * self.rbar - X[:, 2].min()]
        zreq = np.where(mask, P[None, :, 2] + np.sqrt(np.maximum(need, 0)) - X[:, 2][:, None], -np.inf)
        return X + [0, 0, zreq.max()]

    def seed_surface(self, kind="sym"):
        p, rng = self.p, self.rng
        for _ in range(30):
            X, g = self._cluster_motif(SG.SURFACE_GROUPS, kind, aniso=(0.45, 1.2))
            if X is None:
                continue
            site, phi = self.surface_site()
            if kind == "c1":
                X = X @ SG.random_rotation(rng).T
                X[:, 2] *= rng.uniform(0.45, 1.0)
            X = X @ _rotz(phi).T
            X[:, :2] += site
            X = self.drop(X, rng.uniform(0.95, 1.1))
            pos = self.base()
            pos[self.s] = X + rng.normal(0, self.prm["noise"], X.shape)
            if p.min_dist_ok(pos, p.cell0):
                return pos, p.cell0.copy(), "s" + g
        return self.fallback()

    # ------------------------------------------------- clusters in zeolites
    def zeo_info(self):
        if self._zeo is None:
            p = self.p
            rad = float(p.region.get("radius", 3.0))
            reach = rad + 2.0 * self.rbar * (self.ns / 0.74) ** (1 / 3) + 3.0
            imgs = _SH if np.all(p.pbc) else np.array([[0.0, 0.0, 0.0]])
            P = np.concatenate([p.sub_pos + t @ p.cell0 for t in imgs])
            rP = np.tile(p.rcov[: p.n_sub], len(imgs))
            keep = np.linalg.norm(P - self.center, axis=1) < reach
            self._zeo = dict(P=P[keep], rP=rP[keep], rad=rad)
        return self._zeo

    def seed_zeolite(self, kind="sym"):
        p, rng = self.p, self.rng
        info = self.zeo_info()
        for _ in range(40):
            X, g = self._cluster_motif(SG.CLUSTER_GROUPS, kind)
            if X is None:
                continue
            X = X @ SG.random_rotation(rng).T + self.center \
                + SG.random_unit(rng) * info["rad"] * rng.random() ** (1 / 3)
            X += rng.normal(0, self.prm["noise"], X.shape)
            if len(info["P"]):
                d = np.linalg.norm(X[:, None] - info["P"][None], axis=-1)
                if (d < 0.8 * (self.rs[:, None] + info["rP"][None, :])).any():
                    continue
            pos = self.base()
            pos[self.s] = X
            if p.min_dist_ok(pos, p.cell0):
                return pos, p.cell0.copy(), "z" + g
        return self.fallback()

    # ----------------------------------------------------------------- bulk
    def sites_eff(self, G, prim):
        key = (G.symbol, prim)
        if key not in self._sites_cache:
            self._sites_cache[key] = [dict(s, mult=s["mult"] // G.ncent) for s in G.sites] \
                if prim else G.sites
        return self._sites_cache[key]

    def bulk_options(self, name):
        if name not in self._bopts:
            G = SG.space_group(name)
            out = []
            for k in SG.divisors(SG.gcd_list(self.counts)):
                if k > 16:
                    break
                cm = [c // k for c in self.counts]
                for prim in ((False, True) if G.centring != "P" else (False,)):
                    if SG.feasible(cm, self.sites_eff(G, prim)):
                        out.append((k, prim))
            self._bopts[name] = out
        return self._bopts[name]

    def _pbc_ok(self, orb, si, F, lab, L):
        d = orb[:, None] - orb[None]
        d -= np.round(d)
        r = np.linalg.norm((d[:, :, None, :] + _SH[None, None]) @ L, axis=-1)
        idx = np.arange(len(orb))
        r[idx, idx, _SH0] = np.inf
        if r.min() < 0.75 * 2 * self.rsp[si]:
            return False
        if len(F):
            d = orb[:, None] - F[None]
            d -= np.round(d)
            r = np.linalg.norm((d[:, :, None, :] + _SH[None, None]) @ L, axis=-1).min(axis=2)
            if (r < 0.75 * (self.rsp[si] + self.rsp[lab])[None, :]).any():
                return False
        return True

    def pick_hnf(self, k, L):
        if k == 1:
            return np.eye(3)
        if k not in self._hnf:
            self._hnf[k] = _hnfs(k)
        cands = self._hnf[k]
        sc = np.array([_heights(_reduce(H @ L)).min() for H in cands])
        good = np.where(sc >= 0.85 * sc.max())[0]
        if self.rng.random() < 0.5:
            good = good[sc[good] >= sc.max() - 1e-6]
        return cands[good[self.rng.integers(len(good))]]

    def build_bulk(self, name, k, prim):
        rng = self.rng
        G = SG.space_group(name)
        sites = self.sites_eff(G, prim)
        cm = [c // k for c in self.counts]
        plan = SG.partition(cm, sites, rng, alpha=self.prm["alpha"])
        if plan is None:
            return None
        nm = sum(cm)
        vol = self.vpa * nm * (G.ncent if prim else 1) * rng.uniform(0.9, 1.3)
        A = SG.lattice(G.system, rng, vol)
        if prim:
            L = G.prim @ A
            Pinv = np.linalg.inv(G.prim)
        else:
            L = A
        plan.sort(key=lambda t: -sites[t[1]]["mult"])
        F = np.zeros((nm, 3))
        lab = np.zeros(nm, int)
        m0 = 0
        for si, kk in plan:
            m = sites[kk]["mult"]
            for _ in range(40):
                x = G.sample_site(G.sites[kk], rng)
                orb = G.orbit(x)
                if len(orb) != G.sites[kk]["mult"]:
                    continue
                if prim:
                    orb = _unique_frac(orb @ Pinv)
                    if len(orb) != m:
                        continue
                if self._pbc_ok(orb, si, F[:m0], lab[:m0], L):
                    break
            else:
                return None
            F[m0:m0 + m] = orb
            lab[m0:m0 + m] = si
            m0 += m
        H = self.pick_hnf(k, L)
        S = H @ L
        a, c, f = int(H[0, 0]), int(H[1, 1]), int(H[2, 2])
        T = np.array([(i, j, l) for i in range(a) for j in range(c) for l in range(f)], float)
        cart = (((F[None] + T[:, None]).reshape(-1, 3) @ np.linalg.inv(H)) % 1.0) @ S
        labs = np.tile(lab, len(T))
        S2 = _reduce(S)
        cart = ((cart @ np.linalg.inv(S2)) % 1.0) @ S2
        return self.put(cart, labs), S2

    def seed_bulk(self):
        p, rng = self.p, self.rng
        names = [g[0] for g in SG.SPACE_GROUPS]
        for _ in range(40):
            name = names[rng.integers(len(names))]
            opts = self.bulk_options(name)
            if not opts:
                continue
            k, prim = opts[rng.integers(len(opts))]
            res = self.build_bulk(name, k, prim)
            if res is None:
                continue
            X, cell = res
            eps = rng.normal(0, 0.01, (3, 3))
            cell = cell @ (np.eye(3) + 0.5 * (eps + eps.T))
            frac = X @ np.linalg.inv(res[1])
            pos = self.base()
            pos[self.s] = frac @ cell + rng.normal(0, self.prm["noise"], X.shape)
            if p.min_dist_ok(pos, cell):
                return pos, cell, f"{name}/{k}{'p' if prim else ''}"
        return self.fallback()

    # ------------------------------------------------------------- seeding
    def make_seed(self, kind="sym"):
        try:
            if kind == "rand":
                return self.rand_structure()
            if self.cat == "bulk":
                return self.seed_bulk()
            if self.cat == "surface":
                return self.seed_surface(kind)
            if self.cat == "zeolite":
                return self.seed_zeolite(kind)
            return self.seed_cluster(kind)
        except BudgetExhausted:
            raise
        except Exception:
            return self.fallback()

    # --------------------------------------------------------- exploitation
    def make_hop(self):
        p, rng, prm = self.p, self.rng, self.prm
        pos, cell = np.array(self.cur.pos), np.array(self.cur.cell)
        if self.multi and rng.random() < prm["p_swap"]:
            return ops.swap_unlike(p, pos, rng, nswap=1 + int(rng.random() < 0.3)), cell, "swap"
        if self.cat == "cluster" and prm["p_symz"] > 0 and rng.random() < prm["p_symz"]:
            new = self.symmetrise(pos, cell)
            if new is not None:
                return new, cell, "symz"
        if p.variable_cell:
            pos, cell = ops.strain_cell(pos, cell, rng, prm["strain"])
        pos = ops.displace(p, pos, rng, self.step, cell=cell)
        if self.cat != "bulk":
            lost = ops.detached_atoms(p, pos, cell)
            if len(lost):
                pos = ops.relocate_atoms(p, pos, rng, lost, cell)
        return pos, cell, "hop"

    def symmetrise(self, pos, cell):
        p, rng = self.p, self.rng
        pos = ops.unwrap_search(p, pos, cell)
        X = pos[self.s]
        lab = np.zeros(self.ns, int)
        for si, li in enumerate(self.loc):
            lab[li] = si
        G = SG.detect_ops(X, lab, tol=float(rng.choice([0.2, 0.35, 0.5])), max_axes=60)
        if len(G) <= 1:
            return None
        Y = SG.symmetrize(X, lab, G, tol=0.8)
        if Y is None:
            return None
        pos[self.s] = Y + rng.normal(0, self.prm["noise"], Y.shape)
        return ops.repair_overlaps(p, pos, cell)

    # ------------------------------------------------------------ main loop
    def elite(self, res):
        """2: new best, 1: new distinct elite minimum, 0: otherwise (pool updated)."""
        e = float(res.e)
        pool = self.pool
        if len(pool) >= self.prm["pool"] and e >= pool[-1][0]:
            return 0
        fp = ops.fingerprint(self.p, res.pos, res.cell)
        for i, (e2, fp2) in enumerate(pool):
            if ops.same_structure(self.p, e, fp, e2, fp2):
                if e < e2:
                    pool[i] = (e, fp)
                    pool.sort(key=lambda t: t[0])
                return 0
        best = not pool or e < pool[0][0] - 1e-3
        pool.append((e, fp))
        pool.sort(key=lambda t: t[0])
        del pool[self.prm["pool"]:]
        return 2 if best else 1

    def loop(self):
        ctx, rng, prm = self.ctx, self.rng, self.prm
        self.cur, self.pool, self.step = None, [], prm["step"]
        R = {"seed": 0.0, "hop": 0.0}
        S = {"seed": 0.0, "hop": 0.0}
        avg, ps = 50.0, 1.0
        n_acc = n_try = stall = nrel = 0
        wbest, reset = np.inf, False
        types = ["sym", "rand"] if self.cat == "bulk" else ["sym", "c1", "rand"]
        TS = dict.fromkeys(types, 0.0)
        TN = dict.fromkeys(types, 0.0)
        seed_e = []
        while True:
            if self.cur is None or reset or nrel < prm["n_init"]:
                arm = "seed"
            else:
                rs_ = (R["seed"] + 0.3) / (S["seed"] + 3 * avg)
                rh_ = (R["hop"] + 0.3) / (S["hop"] + 3 * avg)
                ps = float(np.clip(rs_ / (rs_ + rh_), prm["p_min"], 1 - prm["p_min"]))
                arm = "seed" if rng.random() < ps else "hop"
            tag = None
            if arm == "hop":
                try:
                    pos, cell, tag = self.make_hop()
                except BudgetExhausted:
                    raise
                except Exception:
                    arm = "seed"
            if arm == "seed":
                w = np.array([(TS[t] + 1.0) / (TN[t] + 5.0 * avg) for t in types])
                w = np.maximum(w / w.sum(), prm["p_type_min"])
                kind = types[rng.choice(len(types), p=w / w.sum())]
                pos, cell, tag = self.make_seed(kind)
                self.st["n_seed"] += 1
                self.st["n_" + kind] = self.st.get("n_" + kind, 0) + 1
                self.count(tag)
            else:
                self.st["n_" + tag] += 1
            res = ctx.relax(pos, cell)
            nrel += 1
            avg = 0.9 * avg + 0.1 * res.nsteps
            ok = (not res.aborted) and bool(np.isfinite(res.e))
            r, k = 0.0, 0
            if ok:
                k = self.elite(res)
                r = (0.0, 0.3, 1.0)[k]
                if k == 2:
                    self.st["best_by"] = tag
            if arm == "seed":
                good = k > 0 or (ok and len(seed_e) >= 4 and res.e <= np.quantile(seed_e, 0.2))
                TS[kind] += float(good)
                TN[kind] += res.nsteps
                if ok:
                    seed_e.append(float(res.e))
            for a in R:
                R[a] *= prm["gamma"]
                S[a] *= prm["gamma"]
            R[arm] += r
            S[arm] += res.nsteps
            if arm == "seed":
                if ok and (self.cur is None or reset or res.e < self.cur.e):
                    if reset or self.cur is None:
                        wbest, stall = res.e, 0
                    elif res.e < wbest - 1e-3:
                        wbest, stall = res.e, 0
                    self.cur, reset = res, False
            else:
                de = res.e - self.cur.e
                acc = ok and (de < 0 or rng.random() < np.exp(-de / prm["kT"]))
                if acc:
                    self.cur = res
                if ok and res.e < wbest - 1e-3:
                    wbest, stall = res.e, 0
                else:
                    stall += 1
                if tag == "hop":
                    n_try += 1
                    n_acc += int(acc)
                    if n_try % prm["adapt_every"] == 0:
                        self.step *= 1.1 if n_acc / n_try > prm["target_accept"] else 0.9
                        self.step = float(np.clip(self.step, 0.1, 2.0))
                if stall >= prm["patience"]:
                    reset, stall = True, 0
                    self.st["restarts"] += 1
            ctx.stats.update(self.st)
            if nrel % 10 == 0:
                ctx.stats["type_rate"] = {t: round(1000 * (TS[t] + 1) / (TN[t] + 5 * avg), 2)
                                          for t in types}
            ctx.stats.update(p_seed=round(ps, 3), step=round(self.step, 3),
                             accept=round(n_acc / max(n_try, 1), 3))
            if nrel % 25 == 0:
                top = sorted(self.grp_count.items(), key=lambda t: -t[1])[:6]
                ctx.stats["groups"] = dict(top)
