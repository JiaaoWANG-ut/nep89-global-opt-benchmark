"""Model-Hessian preconditioned L-BFGS for softmode (fixed and variable cell).

Same convergence test, step cap (0.2 A), energy-rise backtracking and
variable-cell parameterisation (positions in the reference frame + cell_factor
* deformation gradient, cell_factor = n) as ``gopt.context``; the only
difference is the initial inverse Hessian H0 = gamma * P^-1 with P = mu *
(model Hessian + reg * I) instead of a scaled identity. gamma is updated from
the latest secant pair and mu is carried over between relaxations. Every
energy/force call goes through ``ctx.evaluate`` (one ionic step each).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ..context import RelaxResult, _min_height


def _rowmax(v):
    v = v.reshape(-1, 3)
    return float(np.sqrt((v * v).sum(axis=1).max())) if len(v) else 0.0


class PreconRelaxer:
    def __init__(self, ctx, model, memory=20, maxstep=0.2, maxerise=1e-4, reg=0.1, rebuild=10,
                 mu=10.0, abort_f=0.5):
        self.ctx, self.p, self.model = ctx, ctx.p, model
        self.memory, self.maxstep, self.maxerise = memory, maxstep, maxerise
        self.reg, self.rebuild, self.mu = reg, rebuild, mu
        self.abort_f = abort_f
        self.ceiling = None
        p = self.p
        free = np.where(p.free_mask)[0]
        self.other = free[~np.isin(free, p.search_idx)]
        self.free = np.concatenate([self.other, p.search_idx])
        self.n_other3 = 3 * len(self.other)

    # ----------------------------------------------------------- precondition
    def _factor(self, pos, cell):
        ns3 = 3 * self.model.ns
        try:
            H, L = self.model.hessian(pos, cell)
            Hx = H[:ns3, :ns3]
            dmean = float(np.mean(np.diag(Hx)))
            if not np.isfinite(dmean) or dmean <= 1e-6:
                raise ValueError("degenerate model Hessian")
            cfac = cho_factor(Hx + self.reg * dmean * np.eye(ns3))
            if self.p.variable_cell:
                cf = float(self.p.n)
                # strain stiffness converted from eta = L*eps to cell-DOF units cf*F
                ck = max(float(np.mean(np.diag(H[ns3:, ns3:]))) * L * L / (cf * cf), 0.05 * dmean)
                if not np.isfinite(ck):
                    raise ValueError("degenerate strain block")
                self._cell_k = ck
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, IndexError):
            dmean = 1.0
            cfac = cho_factor(np.eye(ns3))
            self._cell_k = 1.0
        self._dmean = dmean
        self._cf = cfac

    def _solve(self, v):
        """P^-1 v without mu (mu is folded into gamma)."""
        out = np.empty_like(v)
        no = self.n_other3
        if self.p.variable_cell:
            n3 = 3 * self.p.n
            out[:n3] = cho_solve(self._cf, v[:n3])
            out[n3:] = v[n3:] / self._cell_k
            return out
        if no:
            out[:no] = v[:no] / self._dmean
        out[no:] = cho_solve(self._cf, v[no:])
        return out

    # ------------------------------------------------------------------ core
    def _lbfgs(self, fun, x, fmax, steps, stop, refactor):
        e, g = fun(x)
        nst = 1
        S, Y, RHO = [], [], []
        gamma = 1.0 / self.mu
        since = 0
        ceil = self.ceiling
        for _ in range(steps):
            gm = _rowmax(g)
            if gm < fmax:
                self.mu = float(np.clip(1.0 / gamma, 0.1, 1e3))
                return x, e, g, True, nst, False
            if stop is not None and stop():
                return x, e, g, False, nst, True
            if ceil is not None and e > ceil and gm < self.abort_f:
                self.n_ceiling = getattr(self, "n_ceiling", 0) + 1
                return x, e, g, False, nst, True
            q = g.copy()
            alphas = []
            for s, y, rho in zip(reversed(S), reversed(Y), reversed(RHO)):
                a = rho * np.dot(s, q)
                q -= a * y
                alphas.append(a)
            r = gamma * self._solve(q)
            for s, y, rho, a in zip(S, Y, RHO, reversed(alphas)):
                r += s * (a - rho * np.dot(y, r))
            if np.dot(r, g) <= 0:
                S, Y, RHO = [], [], []
                r = gamma * self._solve(g)
            d = -r
            lm = _rowmax(d)
            if lm > self.maxstep:
                d *= self.maxstep / lm
            for _ in range(6):
                xn = x + d
                en, gn = fun(xn)
                nst += 1
                if en <= e + self.maxerise:
                    break
                d *= 0.5
                S, Y, RHO = [], [], []
            sk = xn - x
            yk = gn - g
            sy = float(np.dot(sk, yk))
            if sy > 1e-12:
                S.append(sk)
                Y.append(yk)
                RHO.append(1.0 / sy)
                if len(S) > self.memory:
                    S.pop(0)
                    Y.pop(0)
                    RHO.pop(0)
                gamma = float(np.clip(sy / max(np.dot(yk, self._solve(yk)), 1e-30),
                                      1e-3 / self.mu, 1e3 / self.mu))
            x, e, g = xn, en, gn
            since += 1
            if since >= self.rebuild:
                since = 0
                refactor(x)
        return x, e, g, _rowmax(g) < fmax, nst, False

    def _finish(self, res):
        ctx = self.ctx
        ctx.n_relax += 1
        if not res.aborted:
            ctx.minima.append(round(float(res.e), 5))
        return res

    def relax(self, pos, cell=None, fmax=None, steps=None, ceiling=None):
        """Like ctx.relax; with ``ceiling`` the relaxation is abandoned (aborted=True) once
        the force is below abort_f while the energy is still above the ceiling."""
        p = self.p
        self.ceiling = ceiling
        fmax = p.fmax if fmax is None else fmax
        steps = p.max_relax_steps if steps is None else steps
        if p.variable_cell:
            return self._finish(self._relax_vc(pos, p.cell0 if cell is None else cell, fmax, steps))
        return self._finish(self._relax_fixed(pos, fmax, steps))

    def _relax_fixed(self, pos, fmax, steps):
        ctx, p = self.ctx, self.p
        base = np.array(pos, float)
        free = self.free
        cell = p.cell0

        def fun(x):
            q = base.copy()
            q[free] = x.reshape(-1, 3)
            e, f = ctx.evaluate(q)
            return e, -f[free].ravel()

        def refactor(x):
            q = base.copy()
            q[free] = x.reshape(-1, 3)
            self._factor(q, cell)

        self._factor(base, cell)
        x, e, g, conv, nst, ab = self._lbfgs(fun, base[free].ravel(), fmax, steps, None, refactor)
        out = base.copy()
        out[free] = x.reshape(-1, 3)
        return RelaxResult(out, cell.copy(), e, conv, nst, _rowmax(g), ab)

    def _relax_vc(self, pos, cell, fmax, steps):
        ctx, p = self.ctx, self.p
        cell0 = np.array(cell, float)
        n = p.n
        cf = float(n)
        v0 = abs(np.linalg.det(cell0))
        bad = {"flag": False}

        def unpack(x):
            X = x.reshape(-1, 3)
            F = X[n:] / cf
            return X[:n] @ F.T, cell0 @ F.T, F

        def fun(x):
            q, c, F = unpack(x)
            vol = abs(np.linalg.det(c))
            if not (0.25 * v0 < vol < 4.0 * v0) or _min_height(c) < 0.2 * vol ** (1 / 3):
                bad["flag"] = True
            e, f = ctx.evaluate(q, c)
            vir = ctx.last_virial
            vir = 0.5 * (vir + vir.T)
            vir = np.linalg.solve(F, vir.T).T
            return e, -np.vstack([f @ F, vir / cf]).ravel()

        def refactor(x):
            q, c, _ = unpack(x)
            self._factor(q, c)

        x0 = np.vstack([np.array(pos, float), cf * np.eye(3)]).ravel()
        self._factor(np.array(pos, float), cell0)
        x, e, g, conv, nst, ab = self._lbfgs(fun, x0, fmax, steps, lambda: bad["flag"], refactor)
        q, c, _ = unpack(x)
        return RelaxResult(q, c, e, conv, nst, _rowmax(g), ab)
