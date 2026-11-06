"""Stochastic surface walking (SSW) global optimisation (Shang & Liu 2013)."""
import numpy as np

from .. import ops
from ..context import BAD_ENERGY, lbfgs

LABEL = "Stochastic surface walking"
FAMILY = "baseline"
REFERENCE = ("Shang & Liu, J. Chem. Theory Comput. 9, 1838 (2013); Shang & Liu, J. Chem. Theory "
             "Comput. 8, 2215 (2012) (BP-CBD); Zhang, Shang & Liu, J. Chem. Theory Comput. 9, 3252 "
             "(2013); Shang, Zhang & Liu, Phys. Chem. Chem. Phys. 16, 17845 (2014) (SSW-crystal); "
             "Huang et al., WIREs Comput. Mol. Sci. 9, e1415 (2019) (LASP)")
DESCRIPTION = (
    "Metropolis walk between minima; each SSW step climbs out of the current minimum R0 with "
    "Gaussian bias potentials and relaxes into a neighbouring basin. (1) Random direction "
    "N0 = (Ng + lam*Nl)/|..|, Ng = normalised Maxwell-Boltzmann velocities (300 K), Nl = bond "
    "formation mode of two random atoms > 3 A apart, lam ~ U[0.1, 1.5]. (2) For every Gaussian "
    "n = 1..H, N0 is refined by a biased dimer rotation (dimer length 0.005 A, bias "
    "-(a/2)((R1-R0).N0)^2 with a = C_e(N0), at most 6 force calls, stop when the BP-CBD "
    "rotational force |dF_perp| = 2|(F1-F0)_perp| < 0.01 eV/A). (3) A Gaussian "
    "w*exp(-((R-Rt).N)^2/(2 ds^2)) is centred at the current "
    "structure Rt, the structure moves to Rt + ds*N and w is set on the fly (BP-CBD biased "
    "translation) so that the total force along N at that inflection point is +0.1 eV/A. "
    "(4) L-BFGS on the modified PES (<= 25 steps, fmax 0.1 eV/A). (5) Climbing stops after H "
    "Gaussians or when the real energy drops below E(R0). (6) Biases removed, full L-BFGS "
    "relaxation (ctx.relax). (7) Metropolis acceptance at T. Parameters: ds = 0.6 A, H = 10 "
    "(SSW-crystal/LASP standard), T = 1000 K (SSW 2013 molecular example; SSW-crystal used "
    "1200-4000 K, LASP 10-500 K). Bias energies/forces are analytic; every force call (dimer, "
    "climbing, relaxation) is one ionic step. Deviations/simplifications: the CBD (Broyden) "
    "rotation is replaced by exact minimisation of the biased curvature in the dimer plane "
    "(N, rotational force) using finite-difference Hessian-vector products (1 force call per "
    "rotation); the papers quote e.g. 0.1 eV/A for the rotation criterion, but with dR = "
    "0.005 A that means |(H N)_perp| < 10 eV/A^2, which a random N0 already satisfies on "
    "NEP89 metal clusters, so the rotation would never start; 0.01 eV/A (the VTST dimer "
    "default) is used instead; a is floored at 0 at non-minimum points; translations (and rotations for "
    "gas-phase clusters) are projected out of N, frozen atoms never move; if no pair is "
    "> 3 A apart the most distant pair is used; one extra force call per accepted minimum gives "
    "the exact reference force of the dimer. Crystals: instead of the alternating CBD-cell "
    "lattice cycles of SSW-crystal, the LASP generalised-coordinate variant is used: N spans "
    "atoms (reference-frame coordinates) and the cell deformation (scaled by the RMS lattice "
    "vector length, i.e. in A of lattice displacement, rigid rotations projected out), the "
    "lattice part of N0 is a random symmetric strain with the same weight as the atomic part, "
    "the lattice is always passed to the potential in a pairwise (Gauss/Minkowski-type) reduced "
    "basis of the same lattice (free; avoids spurious flat-cell failures after large shears), "
    "and the climb stops if the cell degenerates (volume outside 0.25-4x, reduced cell "
    "thinner than 0.2 V^(1/3)); atoms are wrapped into the cell at the start of every step "
    "and after the climb (same periodic structure; keeps fractional coordinates bounded, "
    "which the NEP neighbour search needs to stay fast). As in bh, "
    "detached atoms of non-bulk systems are re-seated on the cluster before the final "
    "relaxation.")
DEFAULTS = dict(T=1000.0, ds=0.6, n_gauss=10, lam_min=0.1, lam_max=1.5, pair_min=3.0,
                dimer_dr=0.005, rot_evals=6, rot_ftol=0.01, push=0.1, climb_steps=25,
                climb_fmax=0.1, cell_weight=1.0)

_SQRT_E = float(np.exp(0.5))


def _min_height(cell):
    return float(1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1).max())


def _wrap(pos, cell):
    """Atoms mapped into the cell (fractional coordinates in [0, 1)); same periodic structure."""
    frac = np.asarray(pos, float) @ np.linalg.inv(cell)
    return (frac - np.floor(frac)) @ cell


def _reduced(cell):
    """Same lattice in a pairwise (Gauss/Minkowski-type) reduced basis (unimodular, det > 0)."""
    c = np.array(cell, float)
    for _ in range(60):
        g = c @ c.T
        q = g / np.diag(g)[None, :]
        np.fill_diagonal(q, 0.0)
        i, j = divmod(int(np.argmax(np.abs(q))), 3)
        m = np.rint(q[i, j])
        if m == 0.0:
            break
        c[i] -= m * c[j]
    return c


class _Walker:
    """State of one SSW trajectory (coordinates, bias potentials, statistics)."""

    def __init__(self, ctx, rng, prm):
        self.ctx, self.rng, self.prm = ctx, rng, prm
        p = self.p = ctx.p
        self.n = p.n
        self.vc = bool(p.variable_cell)
        self.rows = p.n + (3 if self.vc else 0)
        self.free_atoms = np.where(p.free_mask)[0]
        free_rows = np.zeros(self.rows, bool)
        free_rows[:p.n] = p.free_mask
        free_rows[p.n:] = True
        self.frozen_rows = ~free_rows
        all_free = bool(p.free_mask.all())
        self.proj_trans = all_free
        self.proj_rot = all_free and not self.vc and p.category == "cluster"
        self.reseat = not self.vc and p.category != "bulk"
        self.ds = float(prm["ds"])
        self.kT = ops.KB * float(prm["T"])
        self.cref = None
        self.cf = 1.0
        self.vref = 1.0
        self._clear_bias()
        self.st = dict(moves=0, accepted=0, gauss=0, climb_evals=0, rot_evals=0, relax_evals=0,
                       crossed=0, broken=0)

    # ------------------------------------------------------------ coordinates
    def set_ref(self, pos, cell):
        """Generalised coordinates X (rows x 3) of (pos, cell); F = 1 for variable cell."""
        pos = np.array(pos, float)
        if not self.vc:
            return pos
        self.cref = _reduced(cell)
        self.vref = abs(np.linalg.det(self.cref))
        self.cf = float(np.linalg.norm(self.cref) / np.sqrt(3.0))
        return np.vstack([_wrap(pos, self.cref), self.cf * np.eye(3)])

    def unpack(self, X):
        if not self.vc:
            return X, None, None
        F = X[self.n:] / self.cf
        return X[:self.n] @ F.T, self.cref @ F.T, F

    def real(self, X):
        """Real energy and generalised force (one ionic step); None for a degenerate cell."""
        if not self.vc:
            return self.ctx.evaluate(X)
        pos, cell, F = self.unpack(X)
        vol = abs(np.linalg.det(cell))
        if not 0.25 * self.vref < vol < 4.0 * self.vref:
            return None
        cell = _reduced(cell)
        if _min_height(cell) < 0.2 * vol ** (1 / 3):
            return None
        e, f = self.ctx.evaluate(pos, cell)
        vir = self.ctx.last_virial
        vir = np.linalg.solve(F, (0.5 * (vir + vir.T)).T).T
        return e, np.vstack([f @ F, vir / self.cf])

    # ------------------------------------------------------------ projections
    def ext_basis(self, X):
        """Orthonormal rows spanning the external (zero) modes at X, or None."""
        n, vecs = self.n, []
        if self.proj_trans:
            for k in range(3):
                v = np.zeros((self.rows, 3))
                v[:n, k] = 1.0
                vecs.append(v.ravel())
        if self.proj_rot:
            d = ops.unwrap_search(self.p, X[:n])
            d = d - d.mean(axis=0)
            for k in range(3):
                v = np.zeros((self.rows, 3))
                v[:n] = np.cross(np.eye(3)[k], d)
                vecs.append(v.ravel())
        if self.vc:
            F = X[n:] / self.cf
            for i, j in ((0, 1), (0, 2), (1, 2)):
                om = np.zeros((3, 3))
                om[i, j], om[j, i] = 1.0, -1.0
                v = np.zeros((self.rows, 3))
                v[n:] = self.cf * om @ F
                vecs.append(v.ravel())
        if not vecs:
            return None
        q, r = np.linalg.qr(np.array(vecs).T)
        d = np.abs(np.diag(r))
        return q[:, d > 1e-8 * max(d.max(), 1e-12)].T

    def project(self, v, Q):
        v = v.reshape(self.rows, 3).copy()
        v[self.frozen_rows] = 0.0
        v = v.ravel()
        if Q is not None:
            v -= Q.T @ (Q @ v)
        return v

    # ------------------------------------------------------------ random mode
    def bond_mode(self, X):
        """Bond-formation mode between two random free atoms more than pair_min apart."""
        idx = self.free_atoms
        if len(idx) < 2:
            return None
        pos, cell, _ = self.unpack(X)
        cell = self.p.cell0 if cell is None else cell
        a, b = np.triu_indices(len(idx), 1)
        dv = pos[idx[b]] - pos[idx[a]]
        if np.any(self.p.pbc):
            fr = dv @ np.linalg.inv(cell)
            fr[:, self.p.pbc] -= np.round(fr[:, self.p.pbc])
            dv = fr @ cell
        d = np.linalg.norm(dv, axis=1)
        cand = np.where(d > self.prm["pair_min"])[0]
        k = int(self.rng.choice(cand)) if len(cand) else int(np.argmax(d))
        nl = np.zeros((self.rows, 3))
        nl[idx[a[k]]] = dv[k]
        nl[idx[b[k]]] = -dv[k]
        nrm = np.linalg.norm(nl)
        return nl / nrm if nrm > 1e-12 else None

    def random_direction(self, X, Q):
        prm, rng = self.prm, self.rng
        ng = np.zeros((self.rows, 3))
        ng[:self.n] = ops.mb_velocities(self.p, 300.0, rng)
        ng /= max(np.linalg.norm(ng), 1e-300)
        nl = self.bond_mode(X)
        N = ng if nl is None else ng + rng.uniform(prm["lam_min"], prm["lam_max"]) * nl
        N = N / max(np.linalg.norm(N), 1e-300)
        if self.vc:
            eps = rng.normal(size=(3, 3))
            eps = 0.5 * (eps + eps.T)
            N[self.n:] = prm["cell_weight"] * eps / np.linalg.norm(eps)
        N = self.project(N, Q)
        nrm = np.linalg.norm(N)
        if nrm < 1e-8:
            N = self.project(rng.normal(size=self.rows * 3), Q)
            nrm = np.linalg.norm(N)
        return N / nrm

    # ------------------------------------------------------------ dimer
    def rotate(self, X, G0, N0, Q):
        """Biased dimer rotation of N0 at X (real force G0). Returns (N, evals)."""
        prm = self.prm
        dr = float(prm["dimer_dr"])
        n0 = N0
        r1 = self.real(X + dr * N0.reshape(self.rows, 3))
        if r1 is None:
            return N0, 0
        g0 = G0.ravel()
        HN = -(r1[1].ravel() - g0) / dr
        a = max(float(n0 @ HN), 0.0)
        N, MN = n0.copy(), HN - a * n0
        ev = 1
        while ev < prm["rot_evals"]:
            res = self.project(MN - (N @ MN) * N, Q)
            if 2.0 * dr * np.linalg.norm(res) < prm["rot_ftol"]:
                break
            T = res - (res @ N) * N
            tn = np.linalg.norm(T)
            if tn < 1e-12:
                break
            T /= tn
            r2 = self.real(X + dr * T.reshape(self.rows, 3))
            ev += 1
            if r2 is None:
                break
            HT = -(r2[1].ravel() - g0) / dr
            MT = HT - a * (n0 @ T) * n0
            off = 0.5 * (N @ MT + T @ MN)
            _, vec = np.linalg.eigh(np.array([[N @ MN, off], [off, T @ MT]]))
            c1, c2 = vec[:, 0]
            if c1 < 0:
                c1, c2 = -c1, -c2
            N, MN = c1 * N + c2 * T, c1 * MN + c2 * MT
            nrm = np.linalg.norm(N)
            N, MN = N / nrm, MN / nrm
            if N @ n0 < 0:
                N, MN = -N, -MN
        return N, ev

    # ------------------------------------------------------------ bias
    def _clear_bias(self):
        self.gn = np.zeros((0, self.rows * 3))
        self.gcn = np.zeros(0)
        self.gw = np.zeros(0)

    def add_gaussian(self, center, N, w):
        self.gn = np.vstack([self.gn, N])
        self.gcn = np.append(self.gcn, N @ center.ravel())
        self.gw = np.append(self.gw, w)

    def bias(self, X):
        if not len(self.gw):
            return 0.0, 0.0
        ds2 = self.ds * self.ds
        s = self.gn @ X.ravel() - self.gcn
        g = self.gw * np.exp(-0.5 * s * s / ds2)
        return float(g.sum()), ((g * s / ds2) @ self.gn).reshape(X.shape)

    def relax_biased(self, X, first):
        """L-BFGS on the modified PES from X (first = cached real (e, G) at X)."""
        prm = self.prm
        st = dict(first=first, last=(X, first[0], first[1]), bad=False, ev=0)

        def fun(x):
            if st["first"] is not None:
                r, st["first"] = st["first"], None
            else:
                r = self.real(x)
                if r is None:
                    st["bad"] = True
                    return BAD_ENERGY, np.zeros_like(x)
                st["ev"] += 1
            e, G = r
            if not e < 0.5 * BAD_ENERGY:
                st["bad"] = True
                return e, G
            st["last"] = (x, e, G)
            eb, fb = self.bias(x)
            return e + eb, G + fb

        def stop(i, e, fm):
            return st["bad"] or st["ev"] >= prm["climb_steps"] - 1

        lbfgs(fun, X, prm["climb_fmax"], prm["climb_steps"], abort=stop)
        return st["last"], st["bad"], st["ev"]

    # ------------------------------------------------------------ one SSW step
    def climb(self, X0, E0, G0):
        """Climbing phase from the minimum X0; returns the last structure (pos, cell)."""
        prm, ds = self.prm, self.ds
        Q = self.ext_basis(X0)
        N0 = self.random_direction(X0, Q)
        self._clear_bias()
        X, G = X0, G0
        for n in range(int(prm["n_gauss"])):
            if n:
                Q = self.ext_basis(X)
                N0 = self.project(N0, Q)
                N0 /= max(np.linalg.norm(N0), 1e-300)
            N, ev = self.rotate(X, G, N0, Q)
            self.st["rot_evals"] += ev
            self.st["climb_evals"] += ev
            Xi = X + ds * N.reshape(self.rows, 3)
            r = self.real(Xi)
            if r is None:
                self.st["broken"] += 1
                break
            self.st["climb_evals"] += 1
            ei, Gi = r
            _, fb = self.bias(Xi)
            fN = float((Gi + fb).ravel() @ N)
            w = max(0.0, prm["push"] - fN) * ds * _SQRT_E
            self.add_gaussian(X, N, w)
            self.st["gauss"] += 1
            (X, e, G), bad, ev = self.relax_biased(Xi, (ei, Gi))
            self.st["climb_evals"] += ev
            if bad:
                self.st["broken"] += 1
                break
            if e < E0:
                self.st["crossed"] += 1
                break
        self._clear_bias()
        pos, cell, _ = self.unpack(X)
        if self.vc:
            cell = _reduced(cell)
            pos = _wrap(pos, cell)
        if self.reseat:
            lost = ops.detached_atoms(self.p, pos, cell)
            if len(lost):
                pos = ops.relocate_atoms(self.p, pos, self.rng, lost, cell)
        return np.array(pos), (None if cell is None else np.array(cell))

    def stats(self):
        s = self.st
        m = max(s["moves"], 1)
        g = max(s["gauss"], 1)
        return dict(moves=s["moves"], accept_rate=round(s["accepted"] / m, 4),
                    gauss_per_move=round(s["gauss"] / m, 2),
                    climb_evals_per_move=round(s["climb_evals"] / m, 1),
                    relax_evals_per_move=round(s["relax_evals"] / m, 1),
                    rot_evals_per_gauss=round(s["rot_evals"] / g, 2),
                    crossed=s["crossed"], broken=s["broken"])


def run(ctx, rng, **prm):
    prm = {**DEFAULTS, **prm}
    p = ctx.p
    W = _Walker(ctx, rng, prm)
    cur = None
    while cur is None:
        pos, cell = p.random_structure(rng)
        res = ctx.relax(pos, cell)
        if not res.aborted:
            cur = res
    ref = None  # (X0, G0) of the current minimum
    while True:
        if ref is None:
            X0 = W.set_ref(cur.pos, cur.cell)
            r = W.real(X0)
            if r is None or not r[0] < 0.5 * BAD_ENERGY:
                return  # degenerate current minimum: restart the walk
            ref = (X0, r[1])
        else:
            W.set_ref(cur.pos, cur.cell)
        X0, G0 = ref
        pos, cell = W.climb(X0, cur.e, G0)
        trial = ctx.relax(pos, cell)
        W.st["relax_evals"] += trial.nsteps
        W.st["moves"] += 1
        de = trial.e - cur.e
        if not trial.aborted and trial.e < 0.5 * BAD_ENERGY and \
                (de < 0 or rng.random() < np.exp(-de / W.kT)):
            cur = trial
            ref = None
            W.st["accepted"] += 1
        ctx.stats.update(W.stats())
