"""Zero-cost bond-stiffness model Hessian and soft-mode jumps for softmode (v2+).

Pair springs k_ij = exp(-A (r_ij / r0_ij - 1)) with r0_ij = rcov_i + rcov_j for
r_ij < rcut * r0_ij (anisotropic network model with Lindh/Packwood-style
exponential decay); each spring acts along the bond with a small isotropic part.
Only search atoms are degrees of freedom; frozen substrate atoms anchor them.
Variable-cell problems add the six strain components (scaled by L = V^(1/3) so
that they are measured in Angstrom like the atomic displacements).
"""
import numpy as np
from scipy.spatial import cKDTree

from .. import ops
from ..problem import _images

_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


class ModelHessian:
    def __init__(self, p, A=3.0, rcut=1.6, c_iso=0.05):
        self.p = p
        self.A, self.rcut, self.c = float(A), float(rcut), float(c_iso)
        s = p.search_idx
        self.s = s
        self.ns = len(s)
        rc = p.rcov
        self.rc = rc
        self.r0_ss = rc[s][:, None] + rc[s][None, :]
        self.iu = np.triu_indices(self.ns, 1)
        self.r0_iu = self.r0_ss[self.iu]
        self.bulk = bool(p.variable_cell)
        self.cluster = (p.category == "cluster") and not self.bulk
        self.tree = None
        self.inv0 = np.linalg.inv(p.cell0)
        if p.n_sub and not self.bulk:
            shifts = _images(p.pbc) @ p.cell0
            xyz = (p.sub_pos[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
            self.tree_idx = np.tile(np.arange(p.n_sub), len(shifts))
            self.tree_shift = np.repeat(shifts, p.n_sub, axis=0)
            self.tree = cKDTree(xyz)
            self.rmax_sub = self.rcut * (rc[s].max() + rc[: p.n_sub].max())
        if self.bulk:
            self.rmax = self.rcut * 2 * rc[s].max()
            self._grids = {}

    # ------------------------------------------------------------- pair lists
    def _pairs_fixed(self, pos):
        """Search-search pairs (a<b, local indices) and search-substrate anchors."""
        p = self.p
        xs = pos[self.s]
        a, b = self.iu
        d = xs[a] - xs[b]
        if np.any(p.pbc):
            fr = d @ self.inv0
            fr -= np.where(p.pbc, np.round(fr), 0.0)
            d = fr @ p.cell0
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        m = r < self.rcut * self.r0_iu
        ss = (a[m], b[m], d[m], r[m], self.r0_iu[m])
        anchors = None
        if self.tree is not None:
            fr = xs @ self.inv0
            sw = np.where(p.pbc, np.floor(fr), 0.0) @ p.cell0
            xw = xs - sw
            hits = self.tree.query_ball_point(xw, self.rmax_sub)
            cnt = np.fromiter((len(h) for h in hits), int, len(hits))
            if cnt.sum():
                ii = np.repeat(np.arange(self.ns), cnt)
                jj = np.fromiter((x for h in hits for x in h), int, int(cnt.sum()))
                js = self.tree_idx[jj]
                dv = xw[ii] - (pos[js] + self.tree_shift[jj])
                rr = np.sqrt(np.einsum("ij,ij->i", dv, dv))
                r0 = self.rc[self.s[ii]] + self.rc[js]
                m = (rr < self.rcut * r0) & (rr > 1e-6)
                anchors = (ii[m], dv[m], rr[m], r0[m])
        return ss, anchors

    def _pairs_bulk(self, pos, cell):
        inv = np.linalg.inv(cell)
        frac = pos[self.s] @ inv
        frac -= np.floor(frac)
        heights = 1.0 / np.linalg.norm(inv, axis=0)
        key = tuple(np.minimum(np.ceil(self.rmax / heights).astype(int), 3))
        g = self._grids.get(key)
        n = self.ns
        if g is None:
            grids = [np.arange(-m, m + 1) for m in key]
            sh = np.array(np.meshgrid(*grids, indexing="ij")).reshape(3, -1).T.astype(float)
            pos_half = (sh[:, 0] > 0) | ((sh[:, 0] == 0) & (sh[:, 1] > 0)) | \
                ((sh[:, 0] == 0) & (sh[:, 1] == 0) & (sh[:, 2] > 0))
            ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            once = (ii < jj)[None] | ((ii == jj)[None] & pos_half[:, None, None])
            g = self._grids[key] = (sh, once)
        sh, once = g
        d = ((frac[:, None, :] - frac[None, :, :])[None] - sh[:, None, None, :]) @ cell
        r2 = np.einsum("kijx,kijx->kij", d, d)
        cut2 = (self.rcut * self.r0_ss) ** 2
        k, a, b = np.nonzero((r2 < cut2[None]) & once & (r2 > 1e-12))
        return a, b, d[k, a, b], np.sqrt(r2[k, a, b]), self.r0_ss[a, b]

    # ---------------------------------------------------------------- Hessian
    def _blocks(self, d, r, r0):
        k = np.exp(-self.A * (r / r0 - 1.0))
        u = d / r[:, None]
        blk = (1.0 - self.c) * u[:, :, None] * u[:, None, :] + self.c * np.eye(3)[None]
        return k, u, k[:, None, None] * blk

    def hessian(self, pos, cell):
        """Model Hessian over search DOF (3 ns [+ 6 strain for bulk]) and the strain scale L."""
        ns = self.ns
        H4 = np.zeros((ns, ns, 3, 3))
        L = 1.0
        if self.bulk:
            a, b, d, r, r0 = self._pairs_bulk(pos, cell)
        else:
            (a, b, d, r, r0), anchors = self._pairs_fixed(pos)
        k, u, blk = self._blocks(d, r, r0)
        off = a != b
        if off.any():
            ao, bo, bk = a[off], b[off], blk[off]
            np.add.at(H4, (ao, ao), bk)
            np.add.at(H4, (bo, bo), bk)
            np.add.at(H4, (ao, bo), -bk)
            np.add.at(H4, (bo, ao), -bk)
        if not self.bulk and anchors is not None and len(anchors[0]):
            ia, dv, rr, r0a = anchors
            _, _, bka = self._blocks(dv, rr, r0a)
            np.add.at(H4, (ia, ia), bka)
        H = H4.transpose(0, 2, 1, 3).reshape(3 * ns, 3 * ns)
        if not self.bulk:
            return H, L
        L = abs(np.linalg.det(cell)) ** (1.0 / 3.0)
        # strain coupling of the stretch coordinate u.(dv @ eps) (Voigt, scaled by 1/L)
        g = np.empty((len(k), 6))
        for v, (x, y) in enumerate(_VOIGT):
            g[:, v] = d[:, x] * u[:, y] if x == y else d[:, x] * u[:, y] + d[:, y] * u[:, x]
        g /= L
        kk = (1.0 - self.c) * k
        Hxe = np.zeros((ns, 3, 6))
        t = kk[:, None, None] * u[:, :, None] * g[:, None, :]
        np.add.at(Hxe, a, t)
        np.add.at(Hxe, b, -t)
        Hee = (kk[:, None] * g).T @ g
        Hee += 1e-3 * np.trace(Hee) / 6.0 * np.eye(6)
        Hf = np.zeros((3 * ns + 6, 3 * ns + 6))
        Hf[: 3 * ns, : 3 * ns] = H
        Hf[: 3 * ns, 3 * ns:] = Hxe.reshape(3 * ns, 6)
        Hf[3 * ns:, : 3 * ns] = Hxe.reshape(3 * ns, 6).T
        Hf[3 * ns:, 3 * ns:] = Hee
        return Hf, L

    def rigid_basis(self, pos, cell):
        """Orthonormal basis (D x k) of zero modes to project out."""
        ns = self.ns
        D = 3 * ns + (6 if self.bulk else 0)
        vecs = []
        if self.bulk or self.cluster:
            for x in range(3):
                t = np.zeros((ns, 3))
                t[:, x] = 1.0
                v = np.zeros(D)
                v[: 3 * ns] = t.ravel()
                vecs.append(v)
        if self.cluster:
            xs = ops.unwrap_search(self.p, pos, cell)[self.s]
            xs = xs - xs.mean(axis=0)
            for x in range(3):
                e = np.zeros(3)
                e[x] = 1.0
                v = np.zeros(D)
                v[: 3 * ns] = np.cross(e[None, :], xs).ravel()
                vecs.append(v)
        if not vecs:
            return np.zeros((D, 0))
        M = np.array(vecs).T
        U, sv, _ = np.linalg.svd(M, full_matrices=False)
        return U[:, sv > 1e-6 * sv.max()]

    def _project(self, H, R):
        if R.shape[1]:
            big = 10.0 * (np.abs(np.diag(H)).max() + 1.0)
            PR = R @ (R.T @ H)
            H = H - PR - PR.T + R @ (R.T @ H @ R) @ R.T + big * (R @ R.T)
        lam, V = np.linalg.eigh(0.5 * (H + H.T))
        return lam, V

    def soft_modes(self, pos, cell):
        """Soft modes of the current minimum.

        Returns (lam, V, L, slam, SV): ascending eigenpairs of the internal atomic
        model Hessian (rigid modes removed; for bulk at fixed cell) and, for bulk,
        the eigenpairs of the relaxed-ion strain stiffness (Schur complement), i.e.
        the soft elastic directions in Voigt-eta units (eta = L * strain).
        """
        H, L = self.hessian(pos, cell)
        R = self.rigid_basis(pos, cell)
        n3 = 3 * self.ns
        if not self.bulk:
            lam, V = self._project(H, R)
            m = len(lam) - R.shape[1]
            return lam[:m], V[:, :m], L, None, None
        R = R[:n3]
        lam, V = self._project(H[:n3, :n3], R)
        m = len(lam) - R.shape[1]
        Hxe = H[:n3, n3:]
        T = V[:, :m].T @ Hxe
        S = H[n3:, n3:] - T.T @ (T / np.maximum(lam[:m], 1e-9 * max(lam[m - 1], 1e-9))[:, None])
        slam, SV = np.linalg.eigh(0.5 * (S + S.T))
        return lam[:m], V[:, :m], L, np.maximum(slam, 0.0), SV

    # ------------------------------------------------------------------ jumps
    def direction(self, lam, V, rng, n_modes, beta):
        K = int(min(max(n_modes, 1), len(lam)))
        ref = max(lam[K - 1], 1e-9)
        lk = np.maximum(lam[:K], 1e-3 * ref)
        w = rng.normal(size=K) * (lk[0] / lk) ** (0.5 * beta)
        return V[:, :K] @ w

    def jump(self, pos, cell, modes, rng, amp, n_modes, beta, noise, strain=0.0):
        """Random soft-mode combination with max atomic shift = amp (Angstrom).

        Bulk: atomic soft modes at fixed cell plus a soft elastic strain with
        Frobenius norm ``strain`` (capped at 0.3)."""
        p = self.p
        ns = self.ns
        lam, V, L, slam, SV = modes
        pos = np.array(pos, float)
        cell = np.array(cell, float)
        if len(lam):
            da = self.direction(lam, V, rng, n_modes, beta).reshape(ns, 3)
            da = da / max(np.sqrt((da ** 2).sum(axis=1)).max(), 1e-12)
        else:
            da = np.zeros((ns, 3))
        if noise > 0:
            da = da + noise * rng.normal(size=da.shape) / np.sqrt(3.0)
        da *= amp / max(np.sqrt((da ** 2).sum(axis=1)).max(), 1e-12)
        if self.bulk and strain > 0:
            eta = self.direction(slam, SV, rng, 6, beta)
            eps = np.zeros((3, 3))
            for v, (x, y) in enumerate(_VOIGT):
                eps[x, y] = eps[y, x] = eta[v]
            eps *= min(strain, 0.3) / max(np.linalg.norm(eps), 1e-12)
            pos = pos @ (np.eye(3) + eps)
            cell = cell @ (np.eye(3) + eps)
        pos[self.s] += da
        return ops.repair_overlaps(p, pos, cell), cell
