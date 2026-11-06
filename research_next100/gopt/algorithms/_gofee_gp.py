"""GOFEE helper: Gaussian-process regression with the GOFEE double-Gaussian kernel.

k(x, x') = A [exp(-d^2 / 2 l1^2) + w exp(-d^2 / 2 l2^2)] (+ A * noise on the diagonal),
d = |S(x) - S(x')|, S = [radial features, eta * angular features]  (GOFEE: w = 0.01,
noise = 1e-5 relative to A, eta / l1 / l2 / A fitted by maximum marginal likelihood).
Prior mean = repulsive prior + bias, bias = mean(E - E_rep) (as in GOFEE).
The amplitude A is profiled analytically (its exact ML optimum for given l1, l2, eta).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize


class GP:
    def __init__(self, D, Drad, weight=0.01, noise=1e-5, nmax=1000, nfit=300):
        self.D, self.Drad = int(D), int(Drad)
        self.w, self.noise = float(weight), float(noise)
        self.nmax, self.nfit = int(nmax), int(nfit)
        self.F = np.zeros((0, self.D))
        self.y = np.zeros(0)
        self.erep = np.zeros(0)
        self.tag = np.zeros(0, int)
        self.count = 0
        self.l1 = self.l2 = self.eta = self.eta0 = None
        self.A = 1.0
        self.c0 = 0.0  # centring shift of the scaled features (limits d^2 round-off)
        self.dref0 = None
        self.ready = False
        self.L = None
        self.nfits = 0
        self.nfallback = 0

    def __len__(self):
        return len(self.y)

    # ---------------------------------------------------------------- kernel
    def scale(self, F):
        S = np.array(F, float)
        S[:, self.Drad:] *= self.eta
        return S

    def _ktilde(self, S1, S2, n1=None, n2=None):
        """Unit-amplitude kernel matrix and its d^2."""
        n1 = np.einsum("ij,ij->i", S1, S1) if n1 is None else n1
        n2 = np.einsum("ij,ij->i", S2, S2) if n2 is None else n2
        d2 = np.maximum(n1[:, None] + n2[None, :] - 2.0 * (S1 @ S2.T), 0.0)
        e1 = np.exp(-0.5 * d2 / self.l1 ** 2)
        e2 = np.exp(-0.5 * d2 / self.l2 ** 2)
        return e1 + self.w * e2, e1, e2

    def k0(self):
        return self.A * (1.0 + self.w)

    def similarity(self, F1, F2):
        """Normalised kernel similarity (GOFEE population criterion)."""
        if self.l1 is None:
            return np.zeros((len(F1), len(F2)))
        k, _, _ = self._ktilde(self.scale(F1) - self.c0, self.scale(F2) - self.c0)
        return k / (1.0 + self.w)

    # ---------------------------------------------------------------- data
    def add(self, F, y, erep):
        F = np.atleast_2d(np.asarray(F, float))
        y = np.atleast_1d(np.asarray(y, float))
        erep = np.atleast_1d(np.asarray(erep, float))
        m = len(y)
        ok = np.isfinite(y) & np.all(np.isfinite(F), axis=1) & np.isfinite(erep)
        if not np.all(ok):
            F, y, erep = F[ok], y[ok], erep[ok]
            m = len(y)
        if m == 0:
            return
        N0 = len(self.y)
        self.F = np.vstack([self.F, F])
        self.y = np.concatenate([self.y, y])
        self.erep = np.concatenate([self.erep, erep])
        self.tag = np.concatenate([self.tag, self.count + np.arange(m)])
        self.count += m
        if not self.ready:
            return
        if len(self.y) > self.nmax:
            self.prune()
            self.rebuild()
            return
        # block append to the Cholesky factor (stable Schur complement via L^-1 B)
        try:
            Snew = self.scale(F) - self.c0
            B, _, _ = self._ktilde(self.S, Snew, self.Snorm)
            C, _, _ = self._ktilde(Snew, Snew)
            C[np.diag_indices(m)] += self.noise
            Z = solve_triangular(self.L, B, lower=True, check_finite=False)
            L22 = cholesky(C - Z.T @ Z, lower=True, check_finite=False)
            L = np.zeros((N0 + m, N0 + m), order="F")
            L[:N0, :N0] = self.L
            L[N0:, :N0] = Z.T
            L[N0:, N0:] = L22
            self.L = L
            self.S = np.vstack([self.S, Snew])
            self.Snorm = np.einsum("ij,ij->i", self.S, self.S)
            self._update_alpha()
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            self.nfallback += 1
            self.rebuild()

    def prune(self):
        """Keep the lowest-energy half and the most recent half of the cap."""
        N = len(self.y)
        keep = np.zeros(N, bool)
        keep[np.argsort(self.y, kind="stable")[: self.nmax // 2]] = True
        rec = np.argsort(-self.tag, kind="stable")
        for i in rec:
            if keep.sum() >= self.nmax:
                break
            keep[i] = True
        self.F, self.y, self.erep, self.tag = (self.F[keep], self.y[keep], self.erep[keep],
                                              self.tag[keep])

    def _update_alpha(self):
        self.bias = float(np.mean(self.y - self.erep))
        r = self.y - self.erep - self.bias
        self.alpha = cho_solve((self.L, True), r, check_finite=False)
        self.r = r

    def rebuild(self):
        """Full recomputation of the Cholesky factor (fixed hyperparameters).

        Never raises: if the kernel matrix cannot be factorised even with a diagonal
        jitter of 1e-2 (relative to the unit amplitude) the GP is marked not ready and
        the caller refits / samples randomly."""
        self.ready = False
        if self.l1 is None or len(self.y) == 0:
            return
        S = self.scale(self.F)
        c0 = S.mean(axis=0)
        S = S - c0
        Sn = np.einsum("ij,ij->i", S, S)
        K, _, _ = self._ktilde(S, S, Sn, Sn)
        K[np.diag_indices_from(K)] += self.noise
        if not np.all(np.isfinite(K)):
            self.nfallback += 1
            return
        diag = np.diag_indices_from(K)
        c = None
        for jitter in (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
            try:
                if jitter:
                    K[diag] += jitter
                c = cho_factor(K, lower=True, check_finite=False)
                break
            except (np.linalg.LinAlgError, ValueError):
                if jitter:
                    K[diag] -= jitter
                self.nfallback += 1
        if c is None:
            return
        self.L = np.asfortranarray(np.tril(c[0]))
        self.S, self.Snorm, self.c0 = S, Sn, c0
        self._update_alpha()
        self.ready = bool(np.all(np.isfinite(self.alpha)))

    # ---------------------------------------------------------------- prediction
    def predict(self, F, erep, grad=False, std=True):
        """mu, sigma (B,) and optionally dmu/dF, dsigma/dF (B, D) (dmu/derep = 1)."""
        F = np.atleast_2d(F)
        Sx = self.scale(F) - self.c0
        kt, e1, e2 = self._ktilde(Sx, self.S, None, self.Snorm)
        A = self.A
        mu = kt @ self.alpha + self.bias + erep
        sig = dmu = dsig = None
        if std:
            Z = solve_triangular(self.L, kt.T, lower=True, check_finite=False)
            s2r = 1.0 + self.w - np.einsum("ij,ij->j", Z, Z)
            floor = s2r < self.noise
            sig = np.sqrt(A * np.maximum(s2r, self.noise))
        if grad:
            kd = -(e1 / self.l1 ** 2 + self.w * e2 / self.l2 ** 2)
            W = kd * self.alpha[None, :]
            dmu = Sx * W.sum(1)[:, None] - W @ self.S
            if std:
                V = solve_triangular(self.L, Z, lower=True, trans="T", check_finite=False).T
                Wv = kd * V
                ds2 = -2.0 * A * (Sx * Wv.sum(1)[:, None] - Wv @ self.S)
                dsig = ds2 / (2.0 * sig[:, None])
                dsig[floor] = 0.0
                dsig[:, self.Drad:] *= self.eta
            dmu[:, self.Drad:] *= self.eta
        return mu, sig, dmu, dsig

    # ---------------------------------------------------------------- hyperparameters
    def _subset(self):
        N = len(self.y)
        if N <= self.nfit:
            return np.arange(N)
        keep = np.zeros(N, bool)
        keep[np.argsort(self.y, kind="stable")[: self.nfit // 2]] = True
        for i in np.argsort(-self.tag, kind="stable"):
            if keep.sum() >= self.nfit:
                break
            keep[i] = True
        return np.nonzero(keep)[0]

    def fit(self, maxiter=60):
        """Maximise the (amplitude-profiled) log marginal likelihood over l1, l2, eta."""
        idx = self._subset()
        N = len(idx)
        if N < 2:
            return False
        F = self.F[idx]
        r = self.y[idx] - self.erep[idx]
        r = r - r.mean()
        Fr, Fa = F[:, : self.Drad], F[:, self.Drad:]

        def sqd(X):
            X = X - X.mean(axis=0)
            n = np.einsum("ij,ij->i", X, X)
            return np.maximum(n[:, None] + n[None, :] - 2.0 * X @ X.T, 0.0)

        Dr, Da = sqd(Fr), sqd(Fa)
        off = ~np.eye(N, dtype=bool)
        if self.eta0 is None:
            mr, ma = Dr[off].mean(), Da[off].mean()
            self.eta0 = float(np.sqrt(mr / ma)) if ma > 0 and mr > 0 else 1.0
            self.eta = self.eta0
        D2 = Dr + self.eta ** 2 * Da
        pos = D2[off][D2[off] > 0]
        if self.dref0 is None:
            # length-scale bounds are fixed by the first fit (diverse random structures);
            # a data-dependent scale collapses once the data concentrate on one minimum
            self.dref0 = float(np.sqrt(np.median(pos))) if len(pos) else 1.0
        dref = self.dref0
        if self.l1 is None:
            starts = [(dref, 0.1 * dref), (dref, 10.0 * dref)]
        else:
            starts = [(self.l1, self.l2)]
        lo = np.log([1e-2 * dref, 1e-2 * dref, 0.1 * self.eta0])
        hi = np.log([1e2 * dref, 1e2 * dref, 10.0 * self.eta0])
        w, noise = self.w, self.noise
        I = np.eye(N)

        def nll(p):
            l1, l2, eta = np.exp(p)
            D2 = Dr + eta ** 2 * Da
            e1 = np.exp(-0.5 * D2 / l1 ** 2)
            e2 = np.exp(-0.5 * D2 / l2 ** 2)
            K = e1 + w * e2 + noise * I
            try:
                c = cho_factor(K, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                return 1e10, np.zeros(3)
            a = cho_solve(c, r, check_finite=False)
            A = max(float(r @ a) / N, 1e-12)
            f = 0.5 * N * np.log(A) + np.sum(np.log(np.diag(c[0])))
            Ki = cho_solve(c, I, check_finite=False)
            M = Ki - np.outer(a, a) / A
            g1 = e1 * D2 / l1 ** 2
            g2 = w * e2 * D2 / l2 ** 2
            g3 = -(e1 / l1 ** 2 + w * e2 / l2 ** 2) * (eta ** 2 * Da)
            g = 0.5 * np.array([np.sum(M * g1), np.sum(M * g2), np.sum(M * g3)])
            if not np.isfinite(f) or not np.all(np.isfinite(g)):
                return 1e10, np.zeros(3)
            return f, g

        best = None
        for l1, l2 in starts:
            x0 = np.clip(np.log([l1, l2, self.eta]), lo, hi)
            try:
                res = minimize(nll, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                               options={"maxiter": maxiter})
                fx, xx = float(res.fun), res.x
            except Exception:
                fx, xx = nll(x0)[0], x0
            if best is None or fx < best[0]:
                best = (fx, xx)
        old = (self.l1, self.l2, self.eta)
        self.l1, self.l2, self.eta = (float(v) for v in np.exp(best[1]))
        self.nfits += 1
        self.rebuild()
        if not self.ready and old[0] is not None:
            self.l1, self.l2, self.eta = old
            self.rebuild()
        if self.ready:
            self.A = max(float(self.r @ self.alpha) / len(self.y), 1e-8)
        return True
