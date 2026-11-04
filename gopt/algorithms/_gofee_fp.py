"""GOFEE helper: batched GOFEE / Valle-Oganov fingerprint with analytic position gradients.

Feature vector = [radial block | angular block]
  radial  f_AB(r_b)  = sum_{pairs (i search, j any) of species AB} fc1(r_ij) / r_ij^2 * g_s1(r_b - r_ij)
  angular f_ABC(t_b) = sum_{i search; j<k neighbours of i} fc2(r_ij) fc2(r_ik) * g_s2(t_b - theta_jik)
Substrate atoms only enter as neighbours of search atoms (host-host terms are constant for a
frozen host). The repulsive GOFEE prior sum (ratio*(rc_i+rc_j)/r)^12 is computed on the same
pair list.
"""
import numpy as np

from .. import ops


def _heights(cell):
    return 1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)


class Fingerprint:
    def __init__(self, p, rc1=6.0, rc2=4.0, binwidth1=0.2, nbins2=30, sigma1=0.2, sigma2=0.2,
                 gamma=2, nsigma=4, skin=1.0, margin=2.0, rep_ratio=0.7, max_move=4.0):
        self.p = p
        self.bulk = bool(p.variable_cell)
        self.rc1, self.rc2, self.gamma = float(rc1), float(rc2), int(gamma)
        self.dr = float(binwidth1)
        self.nb1 = int(np.ceil(rc1 / binwidth1 - 1e-9))
        self.nb2 = int(nbins2)
        self.dth = np.pi / self.nb2
        self.s1, self.s2 = float(sigma1), float(sigma2)
        w1 = int(np.ceil(nsigma * sigma1 / self.dr))
        w2 = int(np.ceil(nsigma * sigma2 / self.dth))
        self.off1 = np.arange(-w1, w1 + 1)
        self.off2 = np.arange(-w2, w2 + 1)
        self.skin, self.margin, self.max_move = float(skin), float(margin), float(max_move)
        self.R1 = self.rc1 + self.skin
        self.R2 = self.rc2 + self.skin
        self.n = p.n_search
        self.pbc = np.array(p.pbc, float)
        sym = p.symbols
        ss = sorted(set(sym[i] for i in p.search_idx))
        sub = sorted(set(p.sub_symbols) - set(ss))
        self.species = ss + sub
        sidx = {s: k for k, s in enumerate(self.species)}
        self.spec_s = np.array([sidx[sym[i]] for i in p.search_idx], int)
        self.rc_s = np.asarray(p.rcov, float)[p.search_idx]
        ns, na = len(ss), len(self.species)
        tr = -np.ones((na, na), int)
        t = 0
        for a in range(ns):
            for b in range(na):
                if tr[a, b] < 0:
                    tr[a, b] = tr[b, a] = t
                    t += 1
        self.type_rad, self.nrt = tr, t
        ta = -np.ones((ns, na, na), int)
        t = 0
        for a in range(ns):
            for b in range(na):
                for c in range(b, na):
                    ta[a, b, c] = ta[a, c, b] = t
                    t += 1
        self.type_ang, self.nat = ta, t
        self.Drad = self.nrt * self.nb1
        self.Dang = self.nat * self.nb2
        self.D = self.Drad + self.Dang
        self.rep_ratio = float(rep_ratio)
        self._sub_spec = np.array([sidx[s] for s in p.sub_symbols], int)
        self._setup_region()
        self._setup_env()

    # ------------------------------------------------------------ region / host
    def _setup_region(self):
        """Allowed zone for surrogate moves (region inflated by `margin`) and wrap vectors."""
        p = self.p
        self.sphere, self.box, self.wrap = None, None, []
        if self.bulk or p.category == "cluster":
            return
        reg = p.region or {"type": "cell"}
        t = reg.get("type", "cell")
        if t == "sphere":
            self.sphere = (np.asarray(reg["center"], float), float(reg["radius"]) + self.margin)
        elif t == "box":
            lo = np.asarray(reg["lo"], float) - self.margin
            hi = np.asarray(reg["hi"], float) + self.margin
            if p.category == "surface":
                c = p.cell0
                self.wrap = [k for k in range(3)
                             if p.pbc[k] and abs(c[k, 2]) < 0.5 * np.linalg.norm(c[k])]
                dims = [2]
            else:
                dims = [0, 1, 2]
            self.box = (dims, lo, hi)
        else:
            self.wrap = [k for k in range(3) if p.pbc[k]]

    def _near(self, x, dist):
        ok = np.ones(len(x), bool)
        if self.sphere is not None:
            c, R = self.sphere
            ok &= np.linalg.norm(x - c, axis=1) <= R + dist
        if self.box is not None:
            dims, lo, hi = self.box
            for k in dims:
                ok &= (x[:, k] >= lo[k] - dist) & (x[:, k] <= hi[k] + dist)
        if self.wrap:
            cell = self.p.cell0
            fr = x @ np.linalg.inv(cell)
            h = _heights(cell)
            for k in self.wrap:
                dl = (dist + self.ext) / h[k]
                ok &= (fr[:, k] >= -dl) & (fr[:, k] <= 1 + dl)
        return ok

    def _setup_env(self, ext=None):
        """Host atoms (with periodic images, Cartesian) that can neighbour a search atom."""
        p = self.p
        if ext is None:
            ext = max(6.0, 2.6 * self.n ** (1 / 3) * float(self.rc_s.max())) + self.max_move
        self.ext = float(ext)
        if self.bulk or p.n_sub == 0:
            self.env = np.zeros((0, 3))
            self.env_spec = np.zeros(0, int)
            self.env_rc = np.zeros(0)
            return
        cell = p.cell0
        h = _heights(cell)
        reach = self.R1 + self.ext + self.margin
        nmax = [int(np.ceil(reach / h[k])) if p.pbc[k] else 0 for k in range(3)]
        grids = np.meshgrid(*[np.arange(-m, m + 1) for m in nmax], indexing="ij")
        sh = np.stack([g.ravel() for g in grids], axis=1).astype(float)
        pos = (p.sub_pos[None, :, :] + (sh @ cell)[:, None, :]).reshape(-1, 3)
        spec = np.tile(self._sub_spec, len(sh))
        rc = np.tile(np.asarray(p.rcov, float)[: p.n_sub], len(sh))
        keep = self._near(pos, self.R1)
        self.env, self.env_spec, self.env_rc = pos[keep], spec[keep], rc[keep]

    def inside(self, X):
        """X (B,n,3) -> bool (B,): all search atoms inside the allowed zone."""
        ok = np.ones(X.shape[0], bool)
        if self.sphere is not None:
            c, R = self.sphere
            ok &= (np.linalg.norm(X - c, axis=-1) <= R).all(axis=-1)
        if self.box is not None:
            dims, lo, hi = self.box
            for k in dims:
                ok &= ((X[..., k] >= lo[k]) & (X[..., k] <= hi[k])).all(axis=-1)
        return ok

    def canonical(self, pos, cell):
        """Equivalent copy of `pos` with the search atoms contiguous and in the host frame."""
        p = self.p
        pos = np.array(pos, float)
        s = p.search_idx
        if self.bulk:
            fr = pos[s] @ np.linalg.inv(cell)
            fr -= np.floor(fr)
            pos[s] = fr @ cell
            return pos
        if not np.any(p.pbc):
            return pos
        cell = p.cell0
        pos = ops.unwrap_search(p, pos, cell)
        c = pos[s].mean(axis=0)
        inv = np.linalg.inv(cell)
        shift = np.zeros(3)
        if self.wrap:
            fc = c @ inv
            for k in self.wrap:
                shift[k] = -np.floor(fc[k])
        ref = None
        if self.sphere is not None:
            ref = self.sphere[0]
        elif self.box is not None:
            ref = 0.5 * (self.box[1] + self.box[2])
        if ref is not None:
            fd = (c - ref) @ inv
            for k in range(3):
                if p.pbc[k] and k not in self.wrap:
                    shift[k] = -np.round(fd[k])
        pos[s] += shift @ cell
        return pos

    def check_extent(self, X):
        """Rebuild the host list if a search set is wider than assumed (rare)."""
        if not len(self.env) or not self.wrap:
            return
        c = X.mean(axis=1, keepdims=True)
        r = float(np.linalg.norm(X - c, axis=-1).max()) + self.max_move
        if r > self.ext:
            self._setup_env(ext=r + 2.0)

    # ------------------------------------------------------------ neighbour list
    def _offsets(self, cells):
        if not np.any(self.p.pbc):
            return np.zeros((1, 3))
        h = _heights(cells[0]) if not self.bulk else np.min([_heights(c) for c in cells], axis=0)
        m = [0, 0, 0]
        for k in range(3):
            if self.p.pbc[k] and h[k] < 2 * self.R1:
                m[k] = int(np.ceil(self.R1 / h[k] + 0.5))
        grids = np.meshgrid(*[np.arange(-q, q + 1) for q in m], indexing="ij")
        return np.stack([g.ravel() for g in grids], axis=1).astype(float)

    def build(self, X, cells):
        """Pair/triplet lists (with skin) for a batch X (B,n,3) with cells (B,3,3)."""
        B, n = X.shape[:2]
        R1s = self.R1 ** 2
        offs = self._offsets(cells)
        o = offs
        lex = (o[:, 0] > 0) | ((o[:, 0] == 0) & ((o[:, 1] > 0) | ((o[:, 1] == 0) & (o[:, 2] > 0))))
        ii = np.arange(n)
        upper = ii[:, None] < ii[None, :]
        diag = ii[:, None] == ii[None, :]
        self_img = np.any(lex)
        kss, iss, jss, nss, kse, ise, ese = [], [], [], [], [], [], []
        env = self.env
        inv0 = None if self.bulk else np.linalg.inv(cells[0])
        for k in range(B):
            x = X[k]
            cell = cells[k]
            if n > 1 or self_img:
                dX = x[None, :, :] - x[:, None, :]
                fr = dX @ (np.linalg.inv(cell) if inv0 is None else inv0)
                base = -np.round(fr) * self.pbc
                nv = base[None] + offs[:, None, None, :]
                vec = dX[None] + nv @ cell
                d2 = np.einsum("sijc,sijc->sij", vec, vec)
                lexm = (nv[..., 0] > 0) | ((nv[..., 0] == 0) & ((nv[..., 1] > 0) | (
                    (nv[..., 1] == 0) & (nv[..., 2] > 0))))
                mask = (d2 < R1s) & (upper[None] | (diag[None] & lexm))
                s_, i, j = np.nonzero(mask)
                kss.append(np.full(len(i), k))
                iss.append(i)
                jss.append(j)
                nss.append(nv[s_, i, j])
            if len(env):
                lo = x.min(axis=0) - self.R1
                hi = x.max(axis=0) + self.R1
                sel = np.nonzero(np.all((env >= lo) & (env <= hi), axis=1))[0]
                if len(sel):
                    dv = env[sel][None, :, :] - x[:, None, :]
                    d2 = np.einsum("ijc,ijc->ij", dv, dv)
                    i, e = np.nonzero(d2 < R1s)
                    kse.append(np.full(len(i), k))
                    ise.append(i)
                    ese.append(sel[e])
        cat = lambda a, dt=int: np.concatenate(a).astype(dt) if a else np.zeros(0, dt)  # noqa: E731
        nl = dict(B=B, n=n)
        nl["kss"], nl["iss"], nl["jss"] = cat(kss), cat(iss), cat(jss)
        nl["nss"] = np.concatenate(nss).astype(float) if nss else np.zeros((0, 3))
        nl["kse"], nl["ise"], nl["ese"] = cat(kse), cat(ise), cat(ese)
        Pss, Pse = len(nl["kss"]), len(nl["kse"])
        nl["Pss"], nl["P"] = Pss, Pss + Pse
        pk = np.concatenate([nl["kss"], nl["kse"]])
        pi = np.concatenate([nl["iss"], nl["ise"]])
        spj = np.concatenate([self.spec_s[nl["jss"]], self.env_spec[nl["ese"]]]).astype(int)
        rcj = np.concatenate([self.rc_s[nl["jss"]], self.env_rc[nl["ese"]]])
        nl["pk"], nl["pi"] = pk, pi
        nl["ptype"] = self.type_rad[self.spec_s[pi], spj] if len(pi) else np.zeros(0, int)
        nl["rep_a"] = self.rep_ratio * (self.rc_s[pi] + rcj)
        nl["same_cell"] = not self.bulk
        nl["cells"] = np.array(cells, float)
        # directed short pairs (centre = search atom) for the angular part
        r, d = self._pair_vectors(X, nl)
        short = d < self.R2
        ps = np.nonzero(short[:Pss])[0]
        pe = Pss + np.nonzero(short[Pss:])[0]
        qp = np.concatenate([ps, ps, pe])
        qs = np.concatenate([np.ones(len(ps)), -np.ones(len(ps)), np.ones(len(pe))])
        qc = np.concatenate([nl["iss"][ps], nl["jss"][ps], pi[pe]]).astype(int)
        qspec = np.concatenate([self.spec_s[nl["jss"][ps]], self.spec_s[nl["iss"][ps]],
                                spj[pe]]).astype(int)
        qk = pk[qp] if len(qp) else np.zeros(0, int)
        nl["qp"], nl["qs"] = qp.astype(int), qs
        Q = len(qp)
        if Q > 1:
            grp = qk * n + qc
            order = np.argsort(grp, kind="stable")
            gs = grp[order]
            ug, start, cnt = np.unique(gs, return_index=True, return_counts=True)
            gid = np.repeat(np.arange(len(ug)), cnt)
            rank = np.arange(Q) - start[gid]
            M = int(cnt.max())
            pad = -np.ones((len(ug), M), int)
            pad[gid, rank] = order
            a, b = np.triu_indices(M, 1)
            t1, t2 = pad[:, a].ravel(), pad[:, b].ravel()
            ok = (t1 >= 0) & (t2 >= 0)
            t1, t2 = t1[ok], t2[ok]
        else:
            t1 = t2 = np.zeros(0, int)
        nl["t1"], nl["t2"] = t1, t2
        nl["tk"] = qk[t1] if len(t1) else np.zeros(0, int)
        nl["ttype"] = (self.type_ang[self.spec_s[qc[t1]], qspec[t1], qspec[t2]]
                       if len(t1) else np.zeros(0, int))
        nl["X0"] = np.array(X, float)
        return nl

    def _pair_vectors(self, X, nl):
        P, Pss = nl["P"], nl["Pss"]
        r = np.empty((P, 3))
        if Pss:
            k, i, j = nl["kss"], nl["iss"], nl["jss"]
            if nl["same_cell"]:
                sh = nl["nss"] @ nl["cells"][0]
            else:
                sh = np.einsum("pc,pcd->pd", nl["nss"], nl["cells"][k])
            r[:Pss] = X[k, j] - X[k, i] + sh
        if P > Pss:
            r[Pss:] = self.env[nl["ese"]] - X[nl["kse"], nl["ise"]]
        d = np.sqrt(np.einsum("pc,pc->p", r, r))
        return r, np.maximum(d, 1e-6)

    # ------------------------------------------------------------ cutoff functions
    def _fc1(self, d):
        """Radial taper: 1 below rc1-1 A, cosine to 0 at rc1."""
        r0 = self.rc1 - 1.0
        x = np.clip(d - r0, 0.0, 1.0)
        fc = 0.5 * (1.0 + np.cos(np.pi * x))
        dfc = np.where((d > r0) & (d < self.rc1), -0.5 * np.pi * np.sin(np.pi * x), 0.0)
        return fc, dfc

    def _fc2(self, d):
        """GOFEE polynomial cutoff 1 + g x^(g+1) - (g+1) x^g, x = r/rc2."""
        x = np.minimum(d / self.rc2, 1.0)
        gm = self.gamma
        fc = 1.0 + gm * x ** (gm + 1) - (gm + 1) * x ** gm
        dfc = gm * (gm + 1) * (x ** gm - x ** (gm - 1)) / self.rc2
        return fc, dfc

    # ------------------------------------------------------------ forward / backward
    def forward(self, X, nl, grad=True):
        """Features F (B,D), repulsive prior energies (B,), and backward(G, grep) -> dE/dX."""
        B, n, D, P = nl["B"], nl["n"], self.D, nl["P"]
        r, d = self._pair_vectors(X, nl)
        pk = nl["pk"]
        # only pairs inside rc1 / triplets with both legs inside rc2 have non-zero weight
        # (and gradient); the skin entries of the neighbour list are skipped
        ra = np.nonzero(d < self.rc1)[0]
        dr_ = d[ra]
        fc, dfc = self._fc1(dr_)
        d2 = dr_ * dr_
        w = fc / d2
        bins = np.floor(dr_ / self.dr).astype(int)[:, None] + self.off1[None, :]
        ok = (bins >= 0) & (bins < self.nb1)
        u = (bins + 0.5) * self.dr - dr_[:, None]
        g = np.exp(-u * u / (2 * self.s1 ** 2)) * ok
        idx = (pk[ra] * D + nl["ptype"][ra] * self.nb1)[:, None] + np.clip(bins, 0, self.nb1 - 1)
        F = np.bincount(idx.ravel(), weights=(w[:, None] * g).ravel(), minlength=B * D)
        x6 = (nl["rep_a"] / d) ** 6
        rep = x6 * x6
        erep = np.bincount(pk, weights=rep, minlength=B)
        t1, t2 = nl["t1"], nl["t2"]
        T = len(t1)
        if T:
            qp, qs = nl["qp"], nl["qs"]
            dq = d[qp]
            inq = dq < self.rc2
            tsel = np.nonzero(inq[t1] & inq[t2])[0]
            t1, t2 = t1[tsel], t2[tsel]
            T = len(t1)
        if T:
            uq = r[qp] * (qs / dq)[:, None]
            f2, df2 = self._fc2(dq)
            u1, u2 = uq[t1], uq[t2]
            cos = np.clip(np.einsum("tc,tc->t", u1, u2), -1.0, 1.0)
            th = np.arccos(cos)
            wt = f2[t1] * f2[t2]
            tb = np.floor(th / self.dth).astype(int)[:, None] + self.off2[None, :]
            tok = (tb >= 0) & (tb < self.nb2)
            tu = (tb + 0.5) * self.dth - th[:, None]
            tg = np.exp(-tu * tu / (2 * self.s2 ** 2)) * tok
            tidx = ((nl["tk"][tsel] * D + self.Drad + nl["ttype"][tsel] * self.nb2)[:, None]
                    + np.clip(tb, 0, self.nb2 - 1))
            F += np.bincount(tidx.ravel(), weights=(wt[:, None] * tg).ravel(), minlength=B * D)
        F = F.reshape(B, D)
        if not grad:
            return F, erep, None

        def backward(G, grep=None):
            Gf = np.asarray(G, float).ravel()
            Gp = Gf[idx] * g
            dw = dfc / d2 - 2.0 * fc / (d2 * dr_)
            dEdd = np.zeros(P)
            dEdd[ra] = dw * Gp.sum(1) + w * (Gp * u).sum(1) / self.s1 ** 2
            if grep is not None:
                dEdd = dEdd - np.asarray(grep, float)[pk] * 12.0 * rep / d
            gpv = (dEdd / d)[:, None] * r
            if T:
                Gt = Gf[tidx] * tg
                dEdth = wt * (Gt * tu).sum(1) / self.s2 ** 2
                dEdw = Gt.sum(1)
                sin = np.sqrt(np.maximum(1.0 - cos * cos, 1e-12))
                cth = -dEdth / sin
                d1, d2_ = dq[t1], dq[t2]
                gd1 = dEdw * df2[t1] * f2[t2]
                gd2 = dEdw * f2[t1] * df2[t2]
                g1 = (cth / d1)[:, None] * (u2 - cos[:, None] * u1) + gd1[:, None] * u1
                g2 = (cth / d2_)[:, None] * (u1 - cos[:, None] * u2) + gd2[:, None] * u2
                Q = len(qp)
                for c in range(3):
                    gq = (np.bincount(t1, g1[:, c], Q) + np.bincount(t2, g2[:, c], Q)) * qs
                    gpv[:, c] += np.bincount(qp, gq, P)
            gX = np.zeros((B * n, 3))
            ai = pk * n + nl["pi"]
            Pss = nl["Pss"]
            aj = nl["kss"] * n + nl["jss"]
            for c in range(3):
                gX[:, c] = -np.bincount(ai, gpv[:, c], B * n)
                if Pss:
                    gX[:, c] += np.bincount(aj, gpv[:Pss, c], B * n)
            return gX.reshape(B, n, 3)

        return F, erep, backward

    def features(self, structures):
        """Features and prior energies of a list of (pos, cell) full-system structures."""
        if not structures:
            return np.zeros((0, self.D)), np.zeros(0)
        s = self.p.search_idx
        X, cells = [], []
        for pos, cell in structures:
            cell = self.p.cell0 if (cell is None or not self.bulk) else np.asarray(cell, float)
            X.append(self.canonical(pos, cell)[s])
            cells.append(cell)
        X = np.array(X)
        cells = np.array(cells)
        self.check_extent(X)
        nl = self.build(X, cells)
        F, erep, _ = self.forward(X, nl, grad=False)
        return F, erep
