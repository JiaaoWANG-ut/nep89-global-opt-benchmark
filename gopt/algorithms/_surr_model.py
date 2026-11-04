"""Online linear moment-tensor surrogate used by ``surr`` (private helper).

Energy model (linear in the weights w, one weight block per search species a)::

    E = E_prior + sum_{i in search} eps_i
    eps_i = w0_a + sum_{b,k} wR_{a,bk} R_bk(i) + sum_{l=0..3} sum_{c<=d} wG_{a,l,cd} G_l[c,d](i)

    R_bk(i)      = sum_{j in b} g_k(r_ij)                       radial Gaussians in r/(rcov_a+rcov_b)
    T_c,alpha(i) = sum_{j in b(c)} f_q(c)(r_ij) m_alpha(u_ij)    Cartesian moment tensors, rank <= 3
    G_l[c,d](i)  = sum_{|alpha|=l} mult_alpha T_c,alpha T_d,alpha = sum_jk f_c f_d cos^l(theta_jik)

Neighbours j are all atoms within the pair cutoff (search atoms, frozen substrate
atoms and periodic images), so the frozen host enters through the environment of
the search atoms. This is a MEAM / low-rank moment-tensor-potential style basis
(quadratic "embedding" in the l=0 densities, 3-body angular terms up to cos^3).
E_prior is the exact short-range guard of the benchmark plus a hard wall below
0.6 x the covalent-radius sum; it is subtracted from the training targets.

Everything is batched: B configurations of the same system are evaluated at once
(fixed cell, or one cell per configuration for variable-cell problems).
"""
import math

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

_EXP = np.array([(0, 0, 0),
                 (1, 0, 0), (0, 1, 0), (0, 0, 1),
                 (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1),
                 (3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1), (1, 2, 0),
                 (0, 2, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1)])
_DEG = _EXP.sum(1)
_MULT = np.array([math.factorial(d) / (math.factorial(a) * math.factorial(b) * math.factorial(c))
                  for (a, b, c), d in zip(_EXP, _DEG)])
_SQM = np.sqrt(_MULT)
NMONO = len(_EXP)
_EM1 = np.maximum(_EXP - 1, 0)


def _monomials(u):
    """Monomials of unit-vector components up to degree 3: m (P,20), dm/du (P,20,3)."""
    P = len(u)
    pw = np.empty((P, 3, 4))
    pw[:, :, 0] = 1.0
    pw[:, :, 1] = u
    pw[:, :, 2] = u * u
    pw[:, :, 3] = pw[:, :, 2] * u
    px = pw[:, 0, _EXP[:, 0]]
    py = pw[:, 1, _EXP[:, 1]]
    pz = pw[:, 2, _EXP[:, 2]]
    m = px * py * pz
    dm = np.empty((P, NMONO, 3))
    dm[:, :, 0] = _EXP[:, 0] * pw[:, 0, _EM1[:, 0]] * py * pz
    dm[:, :, 1] = _EXP[:, 1] * px * pw[:, 1, _EM1[:, 1]] * pz
    dm[:, :, 2] = _EXP[:, 2] * px * py * pw[:, 2, _EM1[:, 2]]
    return m, dm


def _heights(cell):
    """Interplanar spacings of a cell (3,3) or a stack of cells (B,3,3)."""
    return 1.0 / np.linalg.norm(np.linalg.inv(cell), axis=-2)


class PairList:
    """Directed (central search atom -> neighbour) pairs of a batch."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class Surrogate:
    def __init__(self, p, nrad=7, nang=4, zmax=12, lmax=3, rc_fac=2.0, rc_min=4.2,
                 rc_max=6.0, skin=0.7, wall=0.6, wall_k=60.0):
        from ase.data import atomic_numbers, covalent_radii
        self.p = p
        sym = np.array(p.symbols)
        self.species = sorted(set(p.symbols))
        self.nsp = len(self.species)
        sidx = {s: k for k, s in enumerate(self.species)}
        self.atom_sp = np.array([sidx[s] for s in sym])
        self.cspecies = sorted(set(p.search_symbols))
        self.ncs = len(self.cspecies)
        cidx = {s: k for k, s in enumerate(self.cspecies)}
        self.S = np.asarray(p.search_idx)
        self.nS = len(self.S)
        self.cs_of = np.array([cidx[s] for s in sym[self.S]])
        self.sp_search = self.atom_sp[self.S]
        self.vc = bool(p.variable_cell)
        rcs = np.array([covalent_radii[atomic_numbers[s]] for s in self.species])
        rcc = np.array([covalent_radii[atomic_numbers[s]] for s in self.cspecies])
        self.nrad = int(nrad)
        self.nang = int(max(1, min(nang, zmax // self.nsp)))
        self.lmax = int(lmax)
        self.nl = self.lmax + 1
        self.Z = self.nsp * self.nang
        self.iu = np.triu_indices(self.Z)
        self.nG = len(self.iu[0])
        self.ncomp = self.nrad + self.nang * NMONO
        self.Pa = 1 + self.nsp * self.nrad + self.nl * self.nG
        self.P = self.ncs * self.Pa
        self.lidx = [np.where(_DEG == l)[0] for l in range(self.nl)]
        # pair types t = a * nsp + b (a: central search species, b: any species)
        nt = self.ncs * self.nsp
        s = (rcc[:, None] + rcs[None, :]).ravel()
        rc = np.clip(rc_fac * s, rc_min, rc_max)
        xc = rc / s
        self.t_s = s
        self.t_rc = rc
        self.t_mur = np.array([np.linspace(0.75, x - 0.15, self.nrad) for x in xc])
        self.t_sigr = np.array([1.1 * (x - 0.9) / max(self.nrad - 1, 1) for x in xc])
        hi = np.minimum(xc - 0.3, 1.85)
        self.t_mua = np.array([np.linspace(0.92, h, self.nang) if self.nang > 1 else [1.0]
                               for h in hi])
        self.t_siga = np.array([max(0.18, 1.0 * (h - 0.92) / max(self.nang - 1, 1)) for h in hi])
        self.t_g0 = p.guard_factor * s if p.guard_factor > 0 else np.zeros(nt)
        for (ga, gb), v in (getattr(p, "guard_pairs", None) or {}).items():
            # per-pair guard floors of the benchmark (gopt/guard.py)
            for ia, ca in enumerate(self.cspecies):
                for ib, cb in enumerate(self.species):
                    if tuple(sorted((ca, cb))) == (ga, gb) and p.guard_factor > 0:
                        self.t_g0[ia * self.nsp + ib] = max(self.t_g0[ia * self.nsp + ib], v)
        self.gk = float(p.guard_k)
        self.t_w0 = wall * s
        self.wk = float(wall_k) / s ** 2
        self.skin = float(skin)
        self.rcut = float(rc.max())
        self.cell0 = np.array(p.cell0, float)
        self.inv0 = np.linalg.inv(self.cell0)
        self.pbc = np.array(p.pbc, bool)
        # substrate images for the KD-tree (fixed cell problems only)
        self.tree = None
        if p.n_sub and not self.vc:
            sub = np.asarray(p.sub_pos, float)
            fr = sub @ self.inv0
            fr -= np.where(self.pbc, np.floor(fr), 0.0)
            sub = fr @ self.cell0
            h = _heights(self.cell0)
            m = [int(np.ceil((self.rcut + skin) / h[k])) if self.pbc[k] else 0 for k in range(3)]
            grid = np.array([(i, j, k) for i in range(-m[0], m[0] + 1)
                             for j in range(-m[1], m[1] + 1) for k in range(-m[2], m[2] + 1)], float)
            sh = grid @ self.cell0
            self.sub_xyz = (sub[None] + sh[:, None]).reshape(-1, 3)
            self.sub_sp = np.tile(self.atom_sp[: p.n_sub], len(sh))
            self.tree = cKDTree(self.sub_xyz)
        self.w = np.zeros(self.P)

    # ------------------------------------------------------------ neighbours
    def build(self, X, cells):
        """Pair list for search positions X (B,nS,3) and cells (B,3,3), cutoff rcut+skin."""
        B, nS = X.shape[0], self.nS
        rl = self.rcut + self.skin
        conf, ci, nj, csh, nint = [], [], [], [], []
        # --- search-search (incl. periodic images)
        if nS > 1 or self.vc:
            inv = np.linalg.inv(cells)
            d = X[:, None, :, :] - X[:, :, None, :]           # d[b,i,j] = X_j - X_i
            f = np.einsum("bijx,bxy->bijy", d, inv)
            w0 = -np.where(self.pbc, np.round(f), 0.0)
            f = f + w0
            hcs = _heights(cells) if self.vc else _heights(self.cell0)[None]
            hmin = hcs.min()
            if hmin >= 2 * rl:
                grid = np.zeros((1, 3))
            else:
                hs = hcs.min(axis=0)
                m = [int(np.ceil(rl / hs[k] + 0.5)) if self.pbc[k] else 0 for k in range(3)]
                grid = np.array([(i, j, k) for i in range(-m[0], m[0] + 1)
                                 for j in range(-m[1], m[1] + 1)
                                 for k in range(-m[2], m[2] + 1)], float)
            ct = self.cs_of[:, None] * self.nsp + self.sp_search[None, :]
            rl2 = ((self.t_rc[ct] + self.skin) ** 2)[None, None]
            g0 = np.nonzero(~grid.any(axis=1))[0]
            g0 = g0[0] if len(g0) else -1
            step = max(1, int(2e6 // max(B * nS * nS, 1)))
            M = np.einsum("bxk,byk->bxy", cells, cells)          # metric tensor
            Mf = np.einsum("bxy,bijy->bijx", M, f)
            ff = np.einsum("bijx,bijx->bij", f, Mf)
            for k0 in range(0, len(grid), step):
                gr = grid[k0:k0 + step]
                gg = np.einsum("sx,bxy,sy->sb", gr, M, gr)
                r2 = (ff[None] + 2.0 * (gr @ Mf.reshape(-1, 3).T).reshape((len(gr),) + ff.shape)
                      + gg[:, :, None, None])
                msk = r2 < rl2
                if k0 <= g0 < k0 + step:
                    msk[g0 - k0] &= ~np.eye(nS, dtype=bool)[None]
                k, b, i, j = np.nonzero(msk)
                if len(b):
                    conf.append(b)
                    ci.append(i)
                    nj.append(j)
                    nint.append(w0[b, i, j] + gr[k])
        if conf:
            conf = np.concatenate(conf)
            ci = np.concatenate(ci)
            nj = np.concatenate(nj)
            nint = np.concatenate(nint)
        else:
            conf = ci = nj = np.zeros(0, int)
            nint = np.zeros((0, 3))
        if self.vc:
            csh = np.zeros((len(conf), 3))
        else:
            csh = nint @ self.cell0
        isS = np.ones(len(conf), bool)
        nsp_n = self.sp_search[nj]
        # --- search-substrate
        if self.tree is not None:
            pts = X.reshape(-1, 3)
            fr = pts @ self.inv0
            wr = np.where(self.pbc, np.floor(fr), 0.0) @ self.cell0
            hits = self.tree.query_ball_point(pts - wr, rl)
            lens = np.fromiter((len(h) for h in hits), int, count=len(hits))
            if lens.sum():
                ii = np.repeat(np.arange(len(pts)), lens)
                jj = np.fromiter((x for h in hits for x in h), int, count=int(lens.sum()))
                tq = self.cs_of[ii % nS] * self.nsp + self.sub_sp[jj]
                dv = self.sub_xyz[jj] - (pts - wr)[ii]
                keep = (dv * dv).sum(1) < (self.t_rc[tq] + self.skin) ** 2
                ii, jj = ii[keep], jj[keep]
                conf = np.concatenate([conf, ii // nS])
                ci = np.concatenate([ci, ii % nS])
                nj = np.concatenate([nj, np.full(len(ii), -1)])
                csh = np.concatenate([csh, self.sub_xyz[jj] + wr[ii]])
                nint = np.concatenate([nint, np.zeros((len(ii), 3))])
                isS = np.concatenate([isS, np.zeros(len(ii), bool)])
                nsp_n = np.concatenate([nsp_n, self.sub_sp[jj]])
        cg = conf * nS + ci
        ng = np.where(isS, conf * nS + np.maximum(nj, 0), 0)
        tp = self.cs_of[ci] * self.nsp + nsp_n
        P = len(cg)
        nslot = B * nS * self.nsp
        slot = cg * self.nsp + nsp_n
        agg = sparse.csr_matrix((np.ones(P), (slot, np.arange(P))), shape=(nslot, P))
        rows = np.concatenate([cg, ng[isS]])
        cols = np.concatenate([np.arange(P), np.nonzero(isS)[0]])
        vals = np.concatenate([np.ones(P), -np.ones(isS.sum())])
        scat = sparse.csr_matrix((vals, (rows, cols)), shape=(B * nS, P))
        cmat = sparse.csr_matrix((np.ones(P), (conf, np.arange(P))), shape=(B, P))
        return PairList(B=B, conf=conf, cg=cg, ng=ng, isS=isS, tp=tp, spn=nsp_n, csh=csh,
                        nint=nint, slot=slot, agg=agg, scat=scat, cmat=cmat,
                        wpair=np.where(isS, 0.5, 1.0), X0=X.copy(), cells0=cells.copy(),
                        nmax=float(np.sqrt((nint ** 2).sum(axis=1)).max()) if P else 0.0)

    def needs_rebuild(self, pl, X, cells):
        if pl is None or pl.B != len(X):
            return True
        dx = np.sqrt(((X - pl.X0) ** 2).sum(-1)).max() if X.size else 0.0
        tol = 2 * dx
        if self.vc:
            dc = np.sqrt(((cells - pl.cells0) ** 2).sum(axis=(1, 2))).max()
            tol += pl.nmax * dc
        return tol > self.skin

    def vectors(self, pl, X, cells):
        Xf = X.reshape(-1, 3)
        d = -Xf[pl.cg] + pl.csh
        d[pl.isS] += Xf[pl.ng[pl.isS]]
        if self.vc:
            d += np.einsum("pk,pkx->px", pl.nint, cells[pl.conf])
        return d

    # ---------------------------------------------------------- pair terms
    def _pair_terms(self, pl, d):
        r = np.sqrt((d * d).sum(1))
        r = np.maximum(r, 1e-6)
        u = d / r[:, None]
        tp = pl.tp
        s = self.t_s[tp]
        rc = self.t_rc[tp]
        x = r / s
        t = r / rc
        inside = t < 1.0
        tt = np.where(inside, t, 1.0)
        fc = (1 - tt * tt) ** 2
        dfc = np.where(inside, -4 * tt * (1 - tt * tt) / rc, 0.0)
        zr = (x[:, None] - self.t_mur[tp]) / self.t_sigr[tp][:, None]
        gr = np.exp(-zr * zr)
        dgr = gr * (-2 * zr / self.t_sigr[tp][:, None]) / s[:, None]
        R = gr * fc[:, None]
        dR = dgr * fc[:, None] + gr * dfc[:, None]
        za = (x[:, None] - self.t_mua[tp]) / self.t_siga[tp][:, None]
        ga = np.exp(-za * za)
        dga = ga * (-2 * za / self.t_siga[tp][:, None]) / s[:, None]
        Fa = ga * fc[:, None]
        dFa = dga * fc[:, None] + ga * dfc[:, None]
        m, dm = _monomials(u)
        # prior: guard + wall (per directed pair, weight 0.5 for search-search)
        g0 = self.t_g0[tp]
        w0 = self.t_w0[tp]
        xg = np.maximum(g0 - r, 0.0)
        xw = np.maximum(w0 - r, 0.0)
        wk = self.wk[tp]
        ep = pl.wpair * (0.5 * self.gk * xg * xg + wk * xw * xw)
        dep = -pl.wpair * (self.gk * xg + 2 * wk * xw)   # dE/dr
        return dict(r=r, u=u, R=R, dR=dR, Fa=Fa, dFa=dFa, m=m, dm=dm, ep=ep, dep=dep)

    def _moments(self, pl, pt):
        P = len(pl.cg)
        phi = np.empty((P, self.ncomp))
        phi[:, : self.nrad] = pt["R"]
        phi[:, self.nrad:] = (pt["Fa"][:, :, None] * pt["m"][:, None, :]).reshape(P, -1)
        T = pl.agg @ phi                               # (B*nS*nsp, ncomp)
        return T.reshape(-1, self.nsp, self.ncomp)

    def _invariants(self, T):
        N = len(T)
        lin = T[:, :, : self.nrad].reshape(N, -1)
        A = T[:, :, self.nrad:].reshape(N, self.nsp, self.nang, NMONO).reshape(N, self.Z, NMONO)
        As = A * _SQM
        G = []
        for l in range(self.nl):
            Al = As[:, :, self.lidx[l]]
            G.append(np.einsum("nca,nda->ncd", Al, Al)[:, self.iu[0], self.iu[1]])
        feat = np.concatenate([np.ones((N, 1)), lin] + G, axis=1)
        return feat, As

    # ------------------------------------------------------------ predict
    def energy_forces(self, pl, X, cells, w=None, want_virial=False):
        """Surrogate energies (B,), forces (B,nS,3) and virials (B,3,3) (if requested)."""
        w = self.w if w is None else w
        B, nS = X.shape[0], self.nS
        d = self.vectors(pl, X, cells)
        pt = self._pair_terms(pl, d)
        T = self._moments(pl, pt)
        N = B * nS
        feat, As = self._invariants(T)
        W = w.reshape(self.ncs, self.Pa)
        csp = np.tile(self.cs_of, B)
        Wi = W[csp]                                     # (N, Pa)
        eps = (feat * Wi).sum(1)
        E = eps.reshape(B, nS).sum(1) + np.bincount(pl.conf, pt["ep"], minlength=B)
        # backward: dE/dT
        dT = np.zeros((N, self.nsp, self.ncomp))
        dT[:, :, : self.nrad] = Wi[:, 1: 1 + self.nsp * self.nrad].reshape(N, self.nsp, self.nrad)
        off = 1 + self.nsp * self.nrad
        dA = np.zeros((N, self.Z, NMONO))
        for l in range(self.nl):
            wl = W[:, off + l * self.nG: off + (l + 1) * self.nG]      # (ncs, nG)
            M = np.zeros((self.ncs, self.Z, self.Z))
            M[:, self.iu[0], self.iu[1]] += 0.5 * wl
            M[:, self.iu[1], self.iu[0]] += 0.5 * wl
            Ml = M[csp]
            Al = As[:, :, self.lidx[l]]
            dA[:, :, self.lidx[l]] = 2 * np.einsum("ncd,nda->nca", Ml, Al) * _SQM[self.lidx[l]]
        dT[:, :, self.nrad:] = dA.reshape(N, self.nsp, self.nang * NMONO)
        dTp = dT.reshape(N * self.nsp, self.ncomp)[pl.slot]            # (P, ncomp)
        P = len(pl.cg)
        r, u = pt["r"], pt["u"]
        s1 = (dTp[:, : self.nrad] * pt["dR"]).sum(1) + pt["dep"]
        dTa = dTp[:, self.nrad:].reshape(P, self.nang, NMONO)
        s1 += np.einsum("pqa,pq,pa->p", dTa, pt["dFa"], pt["m"])
        cq = np.einsum("pqa,pq->pa", dTa, pt["Fa"])                    # (P, 20)
        v = np.einsum("pa,pax->px", cq, pt["dm"]) / r[:, None]
        g = s1[:, None] * u + v - (u * v).sum(1)[:, None] * u          # dE/dd_p
        F = (pl.scat @ g).reshape(B, nS, 3)
        vir = None
        if want_virial:
            vir = -(pl.cmat @ (d[:, :, None] * g[:, None, :]).reshape(P, 9)).reshape(B, 3, 3)
        return E, F, vir

    # ------------------------------------------------------------ training rows
    def design(self, pl, X, cells, want_virial=False):
        """Feature rows: XE (B,P) energy rows, XF (B,nS,3,P) force rows, XV (B,9,P) virial
        rows (or None), and the prior energies/forces/virials to subtract from targets."""
        B, nS = X.shape[0], self.nS
        d = self.vectors(pl, X, cells)
        pt = self._pair_terms(pl, d)
        T = self._moments(pl, pt)
        N = B * nS
        feat, As = self._invariants(T)
        csp = np.tile(self.cs_of, B)
        XE = np.zeros((B, self.ncs, self.Pa))
        for a in range(self.ncs):
            ma = csp == a
            XE[:, a] = feat[ma].reshape(B, -1, self.Pa).sum(1) if ma.any() else 0.0
        XE = XE.reshape(B, self.P)
        P = len(pl.cg)
        r, u = pt["r"], pt["u"]
        # per-pair gradient of every feature of the central atom: D (P, Pa, 3)
        D = np.zeros((P, self.Pa, 3))
        lin0 = 1
        cols = lin0 + pl.spn[:, None] * self.nrad + np.arange(self.nrad)[None, :]
        D[np.arange(P)[:, None], cols, :] = pt["dR"][:, :, None] * u[:, None, :]
        # dPhi_{q,alpha}/dd = dFa m u + Fa (dm - (u.dm) u)/r
        dm = pt["dm"]
        udm = np.einsum("pax,px->pa", dm, u)
        tang = (dm - udm[:, :, None] * u[:, None, :]) / r[:, None, None]       # (P,20,3)
        dPhi = (pt["dFa"][:, :, None, None] * pt["m"][:, None, :, None] * u[:, None, None, :]
                + pt["Fa"][:, :, None, None] * tang[:, None, :, :])             # (P,nang,20,3)
        dPhi *= _SQM[None, None, :, None]
        Ac = As[pl.cg]                                                          # (P,Z,20)
        off = 1 + self.nsp * self.nrad
        rows = pl.spn[:, None] * self.nang + np.arange(self.nang)[None, :]      # (P,nang)
        ar = np.arange(P)[:, None]
        for l in range(self.nl):
            li = self.lidx[l]
            H = np.einsum("pda,pqax->pqdx", Ac[:, :, li], dPhi[:, :, li, :])     # (P,nang,Z,3)
            Gp = np.zeros((P, self.Z, self.Z, 3))
            Gp[ar, rows] = H
            Gs = Gp + Gp.transpose(0, 2, 1, 3)
            D[:, off + l * self.nG: off + (l + 1) * self.nG, :] = Gs[:, self.iu[0], self.iu[1], :]
        # forces: F_k = -dE/dx_k ; dE/dx_ci = -sum D ; dE/dx_nj = +sum D
        XF = np.zeros((B * nS, 3, self.ncs, self.Pa))
        csp_p = self.cs_of[pl.cg % nS]
        for a in range(self.ncs):
            ma = csp_p == a
            if not ma.any():
                continue
            Da = np.where(ma[:, None, None], D, 0.0).transpose(0, 2, 1).reshape(P, 3 * self.Pa)
            XF[:, :, a, :] = (pl.scat @ Da).reshape(B * nS, 3, self.Pa)
        XF = XF.reshape(B, nS, 3, self.P)
        # prior
        gp = (pt["dep"][:, None]) * u
        Fp = (pl.scat @ gp).reshape(B, nS, 3)
        Ep = np.bincount(pl.conf, pt["ep"], minlength=B)
        XV = Vp = None
        if want_virial:
            XV = np.zeros((B, 9, self.ncs, self.Pa))
            for a in range(self.ncs):
                ma = csp_p == a
                if not ma.any():
                    continue
                for b in range(B):
                    sel = ma & (pl.conf == b)
                    if sel.any():
                        XV[b, :, a, :] = -np.einsum("px,pky->xyk", d[sel], D[sel]).reshape(9, self.Pa)
            XV = XV.reshape(B, 9, self.P)
            Vp = -(pl.cmat @ (d[:, :, None] * gp[:, None, :]).reshape(P, 9)).reshape(B, 9)
        return XE, XF, XV, Ep, Fp, Vp


class RidgeAccumulator:
    """Weighted normal equations with standardised columns; several ridge strengths
    are kept and the one with the best out-of-sample (next-data) error is used."""

    def __init__(self, P, lams=(1e-7, 1e-5, 1e-3, 1e-1)):
        self.P = P
        self.A = np.zeros((P, P))
        self.b = np.zeros(P)
        self.n = 0
        self.lams = list(lams)
        self.W = None
        self.err = np.zeros(len(lams))
        self.nerr = 0.0
        self.best = len(lams) // 2

    def add(self, X, y, sw):
        Xw = X * sw[:, None]
        self.A += Xw.T @ Xw
        self.b += Xw.T @ (y * sw)
        self.n += len(y)

    def validate(self, X, y, sw, decay=0.97):
        if self.W is None:
            return
        res = (X @ self.W.T - y[:, None]) * sw[:, None]
        self.err = decay * self.err + (res ** 2).sum(0)
        self.nerr = decay * self.nerr + float((sw ** 2).sum())
        self.best = int(np.argmin(self.err))

    def solve(self):
        d = np.sqrt(np.maximum(np.diag(self.A), 0.0))
        ok = d > 1e-12 * max(d.max(), 1e-300)
        dd = np.where(ok, d, 1.0)
        As = self.A / np.outer(dd, dd)
        bs = self.b / dd
        As[~ok, :] = 0.0
        As[:, ~ok] = 0.0
        bs[~ok] = 0.0
        ev, V = np.linalg.eigh(As)
        ev = np.maximum(ev, 0.0)
        q = V.T @ bs
        scale = max(float(np.mean(ev)), 1e-12)
        W = []
        for lam in self.lams:
            u = V @ (q / (ev + lam * scale))
            W.append(np.where(ok, u / dd, 0.0))
        self.W = np.array(W)
        return self.W[self.best]


def relax_batch(m, X, cells, fmax=0.1, steps=150, maxstep=0.2, memory=10, h0=1.0 / 70.0,
                erise=1e-4, w=None, corr=None, trust_total=None):
    """Batched trust-region L-BFGS on the surrogate.

    X (B,nS,3) search positions, cells (B,3,3). Variable-cell problems use the
    deformation-gradient parameterisation of gopt.context._relax_cell (cell rows
    scaled by n). Steps that raise the energy are rejected and halve the per-config
    trust radius. ``corr=(dG, dE)`` adds a linear correction dE - dG.(x - x_start)
    (generalised coordinates) so that the model reproduces a given (NEP) energy and
    gradient at the start; ``trust_total`` caps the total displacement (max row norm).
    Returns (X, cells, E, fmax, ok, n_batched_evaluations).
    """
    B, nS = X.shape[:2]
    vc = m.vc
    cell0 = np.array(cells, float)
    cf = float(nS)
    if vc:
        x = np.concatenate([np.array(X, float), np.repeat(cf * np.eye(3)[None], B, axis=0)], axis=1)
    else:
        x = np.array(X, float)
    nd = x.shape[1]
    D = nd * 3
    vol0 = np.abs(np.linalg.det(cell0))

    def unpack(xx, idx):
        if not vc:
            return xx, cell0[idx], None
        F = xx[:, nS:] / cf
        pos = np.einsum("bij,bkj->bik", xx[:, :nS], F)
        cel = np.einsum("bij,bkj->bik", cell0[idx], F)
        return pos, cel, F

    state = {"pl": None, "idx": None}

    def fun(xx, idx):
        pos, cel, F = unpack(xx, idx)
        bad = np.zeros(len(idx), bool)
        if vc:
            vol = np.abs(np.linalg.det(cel))
            bad |= ~np.isfinite(vol) | (vol < 0.25 * vol0[idx]) | (vol > 4.0 * vol0[idx])
            if (~bad).any():
                hm = np.full(len(idx), np.inf)
                hm[~bad] = _heights(cel[~bad]).min(axis=1)
                bad |= hm < 0.2 * np.cbrt(np.where(bad, 1.0, vol))
            if bad.any():  # evaluate a harmless placeholder for broken cells
                pos = pos.copy()
                cel = cel.copy()
                pos[bad] = (x[idx[bad], :nS] / 1.0)
                cel[bad] = cell0[idx[bad]]
                F = F.copy()
                F[bad] = np.eye(3)
        if (state["pl"] is None or not np.array_equal(state["idx"], idx)
                or m.needs_rebuild(state["pl"], pos, cel)):
            state["pl"] = m.build(pos, cel)
            state["idx"] = idx.copy()
        E, Fo, vir = m.energy_forces(state["pl"], pos, cel, w, want_virial=vc)
        if vc:
            vir = 0.5 * (vir + vir.transpose(0, 2, 1))
            vir = np.linalg.solve(F, vir.transpose(0, 2, 1)).transpose(0, 2, 1)
            G = np.concatenate([np.einsum("bij,bjk->bik", Fo, F), vir / cf], axis=1)
        else:
            G = Fo
        if corr is not None:
            E = E + corr[1][idx] - (corr[0][idx] * (xx - xs[idx])).sum(axis=(1, 2))
            G = G + corr[0][idx]
        bad |= ~np.isfinite(E) | ~np.all(np.isfinite(G), axis=(1, 2))
        G = np.where(np.isfinite(G), G, 0.0)
        return np.where(bad, np.inf, E), G, bad

    xs = x.copy()
    idx = np.arange(B)
    Ecur, Gcur, bad = fun(x, idx)
    nev = 1
    ok = ~bad
    fm = np.sqrt((Gcur ** 2).sum(-1)).max(-1)
    done = bad | (fm < fmax)
    S = np.zeros((B, memory, D))
    Y = np.zeros((B, memory, D))
    RHO = np.zeros((B, memory))
    trust = np.full(B, float(maxstep))
    ptr = 0
    for _ in range(steps):
        act = np.nonzero(~done)[0]
        if not len(act):
            break
        if len(act) <= 0.5 * len(idx):
            idx = act
        nb = len(idx)
        g = -Gcur[idx].reshape(nb, D)
        q = g.copy()
        order = [(ptr - 1 - k) % memory for k in range(memory)]
        al = np.zeros((nb, memory))
        for k in order:
            a = RHO[idx, k] * (S[idx, k] * q).sum(1)
            q -= a[:, None] * Y[idx, k]
            al[:, k] = a
        gam = np.full(nb, h0)
        found = np.zeros(nb, bool)
        for k in order:
            v = (RHO[idx, k] > 0) & ~found
            if v.any():
                yy = (Y[idx, k] ** 2).sum(1)
                sy = (S[idx, k] * Y[idx, k]).sum(1)
                gam[v] = sy[v] / np.maximum(yy[v], 1e-30)
                found |= v
        gam = np.clip(gam, h0 * 0.01, h0 * 100)
        z = gam[:, None] * q
        for k in reversed(order):
            bb = RHO[idx, k] * (Y[idx, k] * z).sum(1)
            z += S[idx, k] * (al[:, k] - bb)[:, None]
        d = -z
        notdesc = (d * g).sum(1) >= 0
        if notdesc.any():
            RHO[idx[notdesc]] = 0.0
            d[notdesc] = -h0 * g[notdesc]
        d = d.reshape(nb, nd, 3)
        lmax = np.sqrt((d ** 2).sum(-1)).max(-1)
        d *= np.minimum(1.0, trust[idx] / np.maximum(lmax, 1e-12))[:, None, None]
        frozen = done[idx]
        d[frozen] = 0.0
        xn = x[idx] + d
        edge = np.zeros(nb, bool)
        if trust_total is not None:
            tot = np.sqrt(((xn - xs[idx]) ** 2).sum(-1)).max(-1)
            edge = tot > trust_total
            if edge.any():
                xn[edge] = xs[idx[edge]] + (xn[edge] - xs[idx[edge]]) * (
                    trust_total / tot[edge])[:, None, None]
                d = xn - x[idx]
        En, Gn, badn = fun(xn, idx)
        nev += 1
        acc = (En <= Ecur[idx] + erise) & ~badn & ~frozen
        rej = ~acc & ~frozen
        s = d.reshape(nb, D)
        y = (Gcur[idx] - Gn).reshape(nb, D)
        sy = (s * y).sum(1)
        upd = acc & (sy > 1e-12)
        S[idx[upd], ptr] = s[upd]
        Y[idx[upd], ptr] = y[upd]
        RHO[idx[upd], ptr] = 1.0 / sy[upd]
        RHO[idx[~upd], ptr] = 0.0
        ptr = (ptr + 1) % memory
        ia = idx[acc]
        x[ia] = xn[acc]
        Ecur[ia] = En[acc]
        Gcur[ia] = Gn[acc]
        trust[ia] = np.minimum(trust[ia] * 1.5, maxstep)
        ir = idx[rej]
        trust[ir] *= 0.5
        RHO[ir] = 0.0
        fm[idx] = np.sqrt((Gcur[idx] ** 2).sum(-1)).max(-1)
        done |= (fm < fmax) | (trust < 1e-3)
        done[idx[acc & edge]] = True
    pos, cel, _ = unpack(x, np.arange(B))
    return pos, cel, Ecur, fm, ok, nev
