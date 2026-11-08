"""Site-directed relocation for metabh: move an atom to the most coordinated free site.

Candidate sites are the hollows above every bonded triangle of atoms near the search set
(search atoms plus nearby substrate atoms), found by trilateration at covalent bond length.
Sites that overlap any atom or leave the search region are dropped; the remaining ones are
scored by their smooth coordination number and one of the best few is chosen at random.
"""
import numpy as np

from .. import ops

BOND, ACC, OVER = 1.25, 1.02, 0.85


def _trilaterate(p1, p2, p3, r1, r2, r3):
    """Both points at distances r1, r2, r3 from p1, p2, p3 (vectorised over rows)."""
    ex = p2 - p1
    d = np.linalg.norm(ex, axis=1)
    ex /= d[:, None]
    a = p3 - p1
    i = np.einsum("ij,ij->i", ex, a)
    ey = a - i[:, None] * ex
    j = np.linalg.norm(ey, axis=1)
    ok = j > 1e-6
    ey /= np.maximum(j, 1e-9)[:, None]
    ez = np.cross(ex, ey)
    x = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    y = (r1 ** 2 - r3 ** 2 + i ** 2 + j ** 2) / (2 * np.maximum(j, 1e-9)) - i / np.maximum(j, 1e-9) * x
    z2 = r1 ** 2 - x ** 2 - y ** 2
    ok &= z2 > -0.1 * r1 ** 2
    z = np.sqrt(np.clip(z2, 0.0, None))
    base = p1 + x[:, None] * ex + y[:, None] * ey
    pts = np.concatenate([base + z[:, None] * ez, base - z[:, None] * ez])
    return pts[np.concatenate([ok, ok])]


class SiteFinder:
    def __init__(self, p, tree=None, sub_rc=None):
        self.p = p
        self.tree = tree
        self.sub_rc = sub_rc
        self.inv0 = np.linalg.inv(p.cell0)
        self.pbc = np.asarray(p.pbc, bool)
        self.dmf = float(getattr(p, "dmin_factor", 0.7))

    def _wrap(self, x):
        fr = x @ self.inv0
        fr -= np.where(self.pbc, np.floor(fr), 0.0)
        return fr @ self.p.cell0

    def _sub_near(self, x, cut, shift):
        """Substrate image positions within cut of any of x (frame of x)."""
        if self.tree is None:
            return np.zeros((0, 3)), np.zeros(0)
        lst = self.tree.query_ball_point(x - shift, cut)
        idx = np.unique(np.concatenate([np.asarray(a, int) for a in lst])) if len(lst) else []
        if len(idx) == 0:
            return np.zeros((0, 3)), np.zeros(0)
        return self.tree.data[idx] + shift, self.sub_rc[idx]

    def place(self, pos, i, rng, top=3, max_tri=3000):
        """New position array with search atom i moved to a well-coordinated site, or None."""
        p = self.p
        pos = ops.unwrap_search(p, pos, p.cell0)
        s = p.search_idx
        rest = s[s != i]
        if len(rest) < 2:
            return None
        xr = pos[rest]
        rr = p.rcov[rest]
        ri = float(p.rcov[i])
        c = xr.mean(axis=0)
        shift = c - self._wrap(c[None])[0]
        rsub = self.sub_rc.max() if self.tree is not None else 0.0
        xs, rs = self._sub_near(xr, BOND * (rr.max() + rsub), shift)
        xa = np.concatenate([xr, xs])
        ra = np.concatenate([rr, rs])
        isr = np.zeros(len(xa), bool)
        isr[: len(xr)] = True
        d = np.linalg.norm(xa[:, None] - xa[None], axis=2)
        adj = d < BOND * (ra[:, None] + ra[None])
        np.fill_diagonal(adj, False)
        # triangles with at least one search atom
        ia, ja = np.nonzero(np.triu(adj, 1))
        keep = isr[ia] | isr[ja]
        ia, ja = ia[keep], ja[keep]
        if len(ia) == 0:
            return None
        com = adj[ia] & adj[ja]
        com &= np.arange(len(xa))[None, :] > ja[:, None]
        e, l = np.nonzero(com)
        if len(e) == 0:
            return None
        a, b, c = ia[e], ja[e], l
        if len(a) > max_tri:
            sel = rng.choice(len(a), max_tri, replace=False)
            a, b, c = a[sel], b[sel], c[sel]
        f = ACC * rng.uniform(0.97, 1.03)
        cand = _trilaterate(xa[a], xa[b], xa[c], f * (ri + ra[a]), f * (ri + ra[b]), f * (ri + ra[c]))
        if len(cand) == 0:
            return None
        # overlaps with search atoms and nearby substrate atoms
        dc = np.linalg.norm(cand[:, None] - xa[None], axis=2)
        lim = ri + ra[None]
        ok = np.all(dc >= OVER * lim, axis=1)
        cand, dc = cand[ok], dc[ok]
        if len(cand) == 0:
            return None
        if self.tree is not None:
            dd, ii = self.tree.query(cand - shift, k=min(4, self.tree.n))
            dd = np.asarray(dd).reshape(len(cand), -1)
            ii = np.asarray(ii).reshape(len(cand), -1)
            ok = np.all(dd >= OVER * (ri + self.sub_rc[ii]), axis=1)
            cand, dc = cand[ok], dc[ok]
            if len(cand) == 0:
                return None
        xr_ = dc / (ra[None] + ri)
        coord = np.clip((1.4 - xr_) / (1.4 - 1.1), 0.0, 1.0).sum(axis=1)
        coord += 1e-3 * rng.random(len(coord))
        order = np.argsort(-coord)[: max(1, int(top))]
        k = order[int(rng.integers(len(order)))]
        out = np.array(pos)
        out[i] = cand[k]
        return out
