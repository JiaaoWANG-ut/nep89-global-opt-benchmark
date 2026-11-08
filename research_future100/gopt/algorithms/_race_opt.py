"""Resumable L-BFGS for the `race` algorithm.

Same algorithm and parameters as ``gopt.context.lbfgs`` / ``_relax_cell`` (step-length
cap, GMIN-style backtracking, UnitCellFilter-type cell variables), but written as an
object that advances one iteration at a time so that the caller can inspect the
partial trajectory (positions, forces, quasi-Newton model) and stop or resume it.
Every energy/force call goes through ``ctx.evaluate`` (one ionic step each).
"""
import numpy as np


def _min_height(cell):
    return float(1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1).max())


class Relaxer:
    __slots__ = ("ctx", "vc", "cell0", "cf", "v0", "n", "x", "e", "f", "nst", "it",
                 "s_hist", "y_hist", "rho_hist", "memory", "maxstep", "h0", "maxerise",
                 "bad", "pred", "fm", "e_hist")

    def __init__(self, ctx, pos, cell=None, memory=20, maxstep=0.2, h0=1.0 / 70.0,
                 maxerise=1e-4):
        self.ctx = ctx
        p = ctx.p
        self.vc = bool(p.variable_cell)
        self.memory, self.maxstep, self.h0, self.maxerise = memory, maxstep, h0, maxerise
        self.bad = False
        pos = np.array(pos, float)
        self.n = len(pos)
        if self.vc:
            self.cell0 = np.array(p.cell0 if cell is None else cell, float)
            self.cf = float(self.n)
            self.v0 = abs(np.linalg.det(self.cell0))
            x0 = np.vstack([pos, self.cf * np.eye(3)])
        else:
            self.cell0 = p.cell0
            x0 = pos
        self.s_hist, self.y_hist, self.rho_hist = [], [], []
        self.nst = 0
        self.it = 0
        self.pred = np.inf
        self.e_hist = []
        self.x = x0
        self.e, self.f = self._fun(x0)
        self.nst = 1
        self.fm = self._fmax(self.f)
        self.e_hist.append(self.e)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _fmax(f):
        return float(np.sqrt((f * f).sum(axis=1).max())) if len(f) else 0.0

    def _unpack(self, x):
        if not self.vc:
            return x, self.cell0, None
        F = x[self.n:] / self.cf
        return x[:self.n] @ F.T, self.cell0 @ F.T, F

    def _fun(self, x):
        if not self.vc:
            return self.ctx.evaluate(x)
        p, c, F = self._unpack(x)
        vol = abs(np.linalg.det(c))
        if not (0.25 * self.v0 < vol < 4.0 * self.v0) or _min_height(c) < 0.2 * vol ** (1 / 3):
            self.bad = True
        e, f = self.ctx.evaluate(p, c)
        vir = self.ctx.last_virial
        vir = 0.5 * (vir + vir.T)
        vir = np.linalg.solve(F, vir.T).T
        return e, np.vstack([f @ F, vir / self.cf])

    @property
    def pos(self):
        return self._unpack(self.x)[0]

    @property
    def cell(self):
        return self._unpack(self.x)[1]

    @property
    def atom_forces(self):
        return self.f[:self.n]

    def converged(self, fmax):
        return self.fm < fmax

    # -------------------------------------------------------------------- step
    def step(self):
        """One L-BFGS iteration (1-6 ionic steps). Sets ``pred`` = quasi-Newton
        estimate of the remaining energy decrease at the start of the iteration."""
        f = self.f
        g = -f.ravel()
        q = g.copy()
        alphas = []
        for s, y, rho in zip(reversed(self.s_hist), reversed(self.y_hist), reversed(self.rho_hist)):
            a = rho * np.dot(s, q)
            q -= a * y
            alphas.append(a)
        if self.s_hist:
            gamma = np.dot(self.s_hist[-1], self.y_hist[-1]) / np.dot(self.y_hist[-1], self.y_hist[-1])
            gamma = min(max(gamma, self.h0 * 0.01), self.h0 * 100)
        else:
            gamma = self.h0
        z = gamma * q
        for s, y, rho, a in zip(self.s_hist, self.y_hist, self.rho_hist, reversed(alphas)):
            b = rho * np.dot(y, z)
            z += s * (a - b)
        d = -z.reshape(-1, 3)
        gd = float(np.dot(d.ravel(), g))
        if gd >= 0:  # not a descent direction: reset
            self.s_hist, self.y_hist, self.rho_hist = [], [], []
            d = self.h0 * f
            gd = float(np.dot(d.ravel(), g))
        self.pred = -0.5 * gd
        lmax = np.sqrt((d ** 2).sum(axis=1).max())
        if lmax > self.maxstep:
            d *= self.maxstep / lmax
        x, e = self.x, self.e
        for _ in range(6):
            xn = x + d
            en, fn = self._fun(xn)
            self.nst += 1
            if en <= e + self.maxerise:
                break
            d *= 0.5
            self.s_hist, self.y_hist, self.rho_hist = [], [], []
        sk = (xn - x).ravel()
        yk = (f - fn).ravel()
        sy = np.dot(sk, yk)
        if sy > 1e-12:
            self.s_hist.append(sk)
            self.y_hist.append(yk)
            self.rho_hist.append(1.0 / sy)
            if len(self.s_hist) > self.memory:
                self.s_hist.pop(0)
                self.y_hist.pop(0)
                self.rho_hist.pop(0)
        self.x, self.e, self.f = xn, en, fn
        self.fm = self._fmax(fn)
        self.it += 1
        self.e_hist.append(en)
