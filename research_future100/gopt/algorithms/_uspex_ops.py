"""Private helpers of the USPEX-style evolutionary algorithm (``uspex.py``).

No energy calls here: variation operators, the bond-hardness spring model used
for softmutation, cell utilities and the Oganov-Valle fingerprint.
"""
import numpy as np
from ase.geometry import cell_to_cellpar, cellpar_to_cell, minkowski_reduce
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import erf

from .. import ops
from ..problem import min_image_distances

BV_B = 0.37      # Brown bond-valence softness parameter (A)
SPRING_CUT = 1.6  # springs between atoms closer than 1.6 x (sum of covalent radii)


# ------------------------------------------------------------------ basics
def n_valence(z):
    """Valence-electron count from the periodic-table group (s+p; s+d for d-block; 3 for f-block)."""
    z = int(z)
    if z <= 2:
        return z
    nobles = [2, 10, 18, 36, 54, 86, 118]
    for k in range(len(nobles) - 1):
        if z <= nobles[k + 1]:
            pos = z - nobles[k]
            plen = nobles[k + 1] - nobles[k]
            if plen == 8:
                return pos
            if plen == 18:
                return pos if pos <= 12 else pos - 10
            if pos <= 2:
                return pos
            if pos <= 17:
                return 3
            return pos - 14 if pos <= 26 else pos - 24
    return 3


def is_gas_cluster(p):
    return p.n_sub == 0 and not p.variable_cell and p.category != "bulk"


def is_bulk(p):
    return p.variable_cell or p.category == "bulk"


def heights(cell):
    """Interplanar spacings of the three lattice-plane families spanned by the cell rows."""
    return 1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)


def lattice_shifts(cell, pbc, rcut):
    """Integer lattice translations covering all images within rcut of a min-image vector."""
    pbc = np.asarray(pbc, bool)
    if not pbc.any():
        return np.zeros((1, 3))
    nrep = np.where(pbc, np.ceil(rcut / heights(cell) - 1e-9).astype(int), 0)
    nrep = np.minimum(nrep, 8)
    g = [np.arange(-k, k + 1) for k in nrep]
    return np.array(np.meshgrid(*g, indexing="ij")).reshape(3, -1).T.astype(float)


def volume(cell):
    return float(abs(np.linalg.det(cell)))


def reduced(cell):
    try:
        rc, _ = minkowski_reduce(np.asarray(cell, float))
        return np.asarray(rc, float)
    except Exception:
        return np.asarray(cell, float)


def standardize(pos, cell):
    """Minkowski-reduce the cell and rotate it to ASE's standard (lower-triangular) setting.

    Fractional coordinates are kept (wrapped into [0,1)), so the result is the same crystal
    up to a rigid rotation (or reflection). Returns (pos, cell, frac).
    """
    rc = reduced(cell)
    frac = np.asarray(pos, float) @ np.linalg.inv(rc)
    frac -= np.floor(frac)
    std = cellpar_to_cell(cell_to_cellpar(rc))
    return frac @ std, std, frac


def cell_ok(cell, min_len, min_height=0.35):
    """USPEX hard constraints for cells (on the reduced cell): no flat cells, no short vectors."""
    if not np.all(np.isfinite(cell)):
        return False
    v = volume(cell)
    if v < 1e-6:
        return False
    rc = reduced(cell)
    if heights(rc).min() < min_height * v ** (1 / 3):
        return False
    return bool(np.linalg.norm(rc, axis=1).min() >= min_len)


def rescale_volume(pos, cell, v_target):
    f = (v_target / volume(cell)) ** (1 / 3)
    return np.asarray(pos) * f, np.asarray(cell) * f


# ------------------------------------------------------------------ repair
def components(p, pos, cell, factor=1.6):
    s = p.search_idx
    d = min_image_distances(pos[s], pos[s], cell, p.pbc)
    adj = d < factor * (p.rcov[s][:, None] + p.rcov[s][None, :])
    n, lab = connected_components(coo_matrix(adj), directed=False)
    return n, lab


def recenter(p, pos, cell):
    """Gas-phase clusters: put the search-atom centroid at the region centre (energy-invariant)."""
    pos = ops.unwrap_search(p, pos, cell)
    r = p.region
    if r.get("type") == "sphere":
        c = np.asarray(r["center"], float)
    elif r.get("type") == "box":
        c = 0.5 * (np.asarray(r["lo"], float) + np.asarray(r["hi"], float))
    else:
        c = 0.5 * np.asarray(cell).sum(axis=0)
    s = p.search_idx
    pos[s] += c - pos[s].mean(axis=0)
    return pos


def finalize(p, pos, cell, rng):
    """Hard constraints before relaxation: push apart overlapping atoms; for clusters
    (gas phase / supported / in zeolites) re-seat detached atoms and fragments."""
    pos = ops.repair_overlaps(p, pos, cell)
    if is_bulk(p):
        return pos
    for _ in range(4):
        changed = False
        lost = ops.detached_atoms(p, pos, cell)
        if len(lost):
            new = ops.relocate_atoms(p, pos, rng, lost, cell)
            if new is not None:
                pos = new
            changed = True
        if p.connected and p.n_search > 1:
            nc, lab = components(p, pos, cell)
            if nc > 1:
                big = np.bincount(lab).argmax()
                new = ops.relocate_atoms(p, pos, rng, p.search_idx[lab != big], cell)
                if new is not None:
                    pos = new
                changed = True
        if not changed:
            break
    if is_gas_cluster(p):
        pos = recenter(p, pos, cell)
    return pos


# ------------------------------------------------- bond-hardness spring model
def substrate_bv_sums(p):
    """Per-substrate-atom sums of exp(-r/b) over substrate neighbours (fixed, computed once)."""
    S0 = np.zeros(p.n_sub)
    if p.n_sub == 0:
        return S0
    rc = p.rcov[: p.n_sub]
    rmax = SPRING_CUT * 2 * rc.max()
    shifts = lattice_shifts(p.cell0, p.pbc, rmax) @ p.cell0
    img = (p.sub_pos[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    img_idx = np.tile(np.arange(p.n_sub), len(shifts))
    m = cKDTree(img).sparse_distance_matrix(cKDTree(p.sub_pos), rmax, output_type="coo_matrix")
    i = img_idx[m.row]
    j = m.col
    r = m.data
    ok = (r > 1e-6) & (r < SPRING_CUT * (rc[i] + rc[j]))
    np.add.at(S0, j[ok], np.exp(-r[ok] / BV_B))
    return S0


def neighbor_pairs(p, pos, cell):
    """Directed pairs (a, j, d): a indexes search atoms, j any atom (all images), d = x_j - x_i,
    |d| < SPRING_CUT * (r_i + r_j); an atom paired with its own image is included."""
    s = p.search_idx
    pos = np.asarray(pos, float)
    cell = np.asarray(cell, float)
    cut = SPRING_CUT * (p.rcov[s][:, None] + p.rcov[None, :])
    pbc = np.asarray(p.pbc, bool)
    inv = np.linalg.inv(cell)
    shifts = lattice_shifts(cell, pbc, cut.max()) @ cell
    A, J, D = [], [], []
    for a, i in enumerate(s):
        d0 = pos - pos[i]
        if pbc.any():
            f = d0 @ inv
            f[:, pbc] -= np.round(f[:, pbc])
            d0 = f @ cell
        d = d0[None, :, :] + shifts[:, None, :]
        r2 = np.einsum("mnk,mnk->mn", d, d)
        mi, jj = np.nonzero((r2 < cut[a][None, :] ** 2) & (r2 > 1e-10))
        A.append(np.full(len(jj), a))
        J.append(jj)
        D.append(d[mi, jj])
    return np.concatenate(A), np.concatenate(J), np.concatenate(D)


def hardness_hessian(p, pos, cell, nval, S0):
    """Force-constant matrix (3ns x 3ns) of springs E = 1/2 sum_k H_k (r_k - r_k0)^2 at rest.

    Bond hardness H_k after Lyakhov & Oganov, PRB 84, 092103 (2011) / Li et al. PRL 100, 235504
    (2008): chi_i = 0.481 n_i / (R_i + Delta_k/2), CN_i^k = v_i / s_i^k with Brown bond valences
    s ~ exp(-r/0.37), X_k = sqrt(chi_i chi_j / (CN_i CN_j)), f_k = |chi_i-chi_j| / (4 sqrt(chi_i chi_j)),
    H_k = X_k exp(-2.7 f_k). Frozen substrate atoms act as fixed spring anchors.
    """
    s = p.search_idx
    ns = len(s)
    a, j, d = neighbor_pairs(p, pos, cell)
    if len(a) == 0:
        return np.zeros((3 * ns, 3 * ns))
    r = np.linalg.norm(d, axis=1)
    i = s[a]
    w = np.exp(-r / BV_B)
    S = np.zeros(p.n)
    np.add.at(S, i, w)
    sub = j < p.n_sub
    if p.n_sub:
        np.add.at(S, j[sub], w[sub])
        S[: p.n_sub] += S0
    R = p.rcov
    delta = r - R[i] - R[j]
    chi_i = 0.481 * nval[i] / np.maximum(R[i] + 0.5 * delta, 0.1)
    chi_j = 0.481 * nval[j] / np.maximum(R[j] + 0.5 * delta, 0.1)
    X = w * np.sqrt(chi_i * chi_j / (S[i] * S[j]))
    fion = np.abs(chi_i - chi_j) / (4 * np.sqrt(chi_i * chi_j))
    H = X * np.exp(-2.7 * fion)
    u = d / r[:, None]
    K = H[:, None, None] * u[:, :, None] * u[:, None, :]
    Phi = np.zeros((ns, ns, 3, 3))
    if sub.any():
        np.add.at(Phi, (a[sub], a[sub]), K[sub])
    ss = (~sub) & (i < j)  # each search-search bond once; self-images vanish at Gamma
    if ss.any():
        aa, bb, Ks = a[ss], j[ss] - p.n_sub, K[ss]
        np.add.at(Phi, (aa, aa), Ks)
        np.add.at(Phi, (bb, bb), Ks)
        np.add.at(Phi, (aa, bb), -Ks)
        np.add.at(Phi, (bb, aa), -Ks)
    return Phi.transpose(0, 2, 1, 3).reshape(3 * ns, 3 * ns)


def rigid_vectors(p, pos, cell):
    """Zero modes of the isolated system: translations (bulk, gas-phase) and rotations (gas-phase)."""
    ns = p.n_search
    vecs = []
    if is_bulk(p) or is_gas_cluster(p):
        for k in range(3):
            t = np.zeros((ns, 3))
            t[:, k] = 1.0
            vecs.append(t.ravel())
    if is_gas_cluster(p):
        x = ops.unwrap_search(p, pos, cell)[p.search_idx]
        x = x - x.mean(axis=0)
        for k in range(3):
            e = np.zeros(3)
            e[k] = 1.0
            vecs.append(np.cross(e, x).ravel())
    return np.array(vecs).T if vecs else np.zeros((3 * ns, 0))


def soft_modes(p, pos, cell, nval, S0, rel_tol=1e-3):
    """Normal modes of the spring model, softest first, grouped by degeneracy.

    Returns a list of (eigenvalue, vectors[3ns, g]) with rigid-body motions projected out.
    """
    Phi = hardness_hessian(p, pos, cell, nval, S0)
    n3 = Phi.shape[0]
    Rv = rigid_vectors(p, pos, cell)
    if Rv.shape[1]:
        Uu, sv, _ = np.linalg.svd(Rv, full_matrices=True)
        rank = int((sv > 1e-8 * sv.max()).sum())
        B = Uu[:, rank:]
    else:
        B = np.eye(n3)
    lam, w = np.linalg.eigh(B.T @ Phi @ B)
    V = B @ w
    scale = max(abs(lam).max(), 1e-12)
    groups = []
    k = 0
    while k < len(lam):
        m = k + 1
        while m < len(lam) and lam[m] - lam[k] <= rel_tol * scale:
            m += 1
        groups.append((float(lam[k]), V[:, k:m]))
        k = m
    return groups


def mode_displacement(p, group, amp, rng):
    """Random unit combination inside a degenerate mode group, scaled so that the largest
    atomic displacement equals amp, with a random sign."""
    V = group[1]
    v = V @ rng.normal(size=V.shape[1]) if V.shape[1] > 1 else V[:, 0].copy()
    v = v.reshape(-1, 3)
    m = np.linalg.norm(v, axis=1).max()
    if m < 1e-12:
        v = ops.random_unit(rng, len(v))
        m = 1.0
    return v * (amp / m) * (1.0 if rng.random() < 0.5 else -1.0)


# ------------------------------------------------------------ heredity
def heredity_cluster(p, pos_a, pos_b, rng, cell, frac_lo=0.25, frac_hi=0.75):
    """Spatially coherent heredity for clusters (gas phase, supported, in pores).

    A random plane (vertical for supported clusters) cuts both parents; the child takes a random
    fraction (frac_lo..frac_hi) of atoms from one side of parent A and completes the composition
    with the atoms on the opposite side of parent B (B translated onto A's centre; laterally only
    for supported clusters; gas-phase B is also randomly rotated).
    """
    s = p.search_idx
    ns = len(s)
    sym = np.array(p.symbols)[s]
    a = ops.unwrap_search(p, pos_a, cell)
    xa = a[s]
    xb = ops.unwrap_search(p, pos_b, cell)[s]
    ca, cb = xa.mean(axis=0), xb.mean(axis=0)
    if is_gas_cluster(p):
        xb = (xb - cb) @ ops.random_rotation(rng).T + cb
    shift = ca - cb
    nrm = ops.random_unit(rng)
    if p.category == "surface":
        shift[2] = 0.0
        nrm[2] = 0.0
        if np.linalg.norm(nrm) < 1e-6:
            nrm = np.array([1.0, 0.0, 0.0])
        nrm /= np.linalg.norm(nrm)
    xb = xb + shift
    pa = (xa - ca) @ nrm
    pb = (xb - ca) @ nrm
    n_a = int(np.clip(round(rng.uniform(frac_lo, frac_hi) * ns), 1, ns - 1))
    need = {x: int((sym == x).sum()) for x in set(sym)}
    picked = []
    for k in np.argsort(-pa)[:n_a]:
        picked.append((sym[k], xa[k]))
        need[sym[k]] -= 1
    for k in np.argsort(pb):
        if need[sym[k]] > 0:
            picked.append((sym[k], xb[k]))
            need[sym[k]] -= 1
    slots = {x: list(s[sym == x]) for x in set(sym)}
    child = np.array(a)
    for x, xyz in picked:
        child[slots[x].pop(0)] = xyz
    return child


def heredity_bulk(p, pos_a, cell_a, pos_b, cell_b, rng, v_target, min_len,
                  frac_lo=0.25, frac_hi=0.75, variable_cell=True, tries=10):
    """Oganov-Glass heredity: slab (fractional coordinates along a random lattice vector, random
    position and thickness, both parents randomly shifted) of A + the complementary slab of B;
    lattice = random weighted average of the parents' standardized lattices; composition repaired
    with atoms from the unused parts of the parents; volume rescaled to v_target.
    Returns (pos, cell) or None.
    """
    s = p.search_idx
    sym = np.array(p.symbols)[s]
    if variable_cell:
        _, la, fa0 = standardize(np.asarray(pos_a)[s], cell_a)
        _, lb, fb0 = standardize(np.asarray(pos_b)[s], cell_b)
    else:
        inv = np.linalg.inv(p.cell0)
        fa0 = (np.asarray(pos_a)[s] @ inv) % 1.0
        fb0 = (np.asarray(pos_b)[s] @ inv) % 1.0
    for _ in range(tries):
        fa = (fa0 + rng.random(3)) % 1.0
        fb = (fb0 + rng.random(3)) % 1.0
        k = int(rng.integers(3))
        t = rng.uniform(frac_lo, frac_hi)
        if variable_cell:
            w = rng.random()
            lat = w * la + (1 - w) * lb
            lat *= (v_target / volume(lat)) ** (1 / 3)
            if not cell_ok(lat, min_len):
                continue
        else:
            lat = np.asarray(p.cell0, float)
        in_a = fa[:, k] < t
        in_b = fb[:, k] >= t
        frac = []
        spec = []
        for x in sorted(set(sym)):
            need = int((sym == x).sum())
            have = [fa[i] for i in np.where(in_a & (sym == x))[0]] + \
                   [fb[i] for i in np.where(in_b & (sym == x))[0]]
            if len(have) > need:
                keep = rng.choice(len(have), need, replace=False)
                have = [have[i] for i in keep]
            spare = [fa[i] for i in np.where(~in_a & (sym == x))[0]] + \
                    [fb[i] for i in np.where(~in_b & (sym == x))[0]]
            while len(have) < need and spare:
                # missing atoms: take the unused parent atom farthest from the current child atoms
                cur = np.array(frac + have) if (frac or have) else None
                if cur is None:
                    pick = int(rng.integers(len(spare)))
                else:
                    dd = min_image_distances(np.array(spare) @ lat, cur @ lat, lat, p.pbc).min(axis=1)
                    pick = int(np.argmax(dd))
                have.append(spare.pop(pick))
            if len(have) < need:
                break
            frac.extend(have)
            spec.extend([x] * need)
        else:
            frac = np.array(frac)
            spec = np.array(spec)
            out = np.zeros((p.n, 3))
            out[: p.n_sub] = p.sub_pos
            for x in set(sym):
                out[s[sym == x]] = frac[spec == x] @ lat
            return out, lat
    return None


# ------------------------------------------------------------ lattice mutation
def strain_matrix(rng, sigma):
    """Oganov & Glass (2006) Eq. (2): symmetric (I + eps), eps_i ~ N(0, sigma) clipped to [-1, 1]."""
    e = np.clip(rng.normal(0.0, sigma, 6), -1.0, 1.0)
    return np.array([[1 + e[0], e[5] / 2, e[4] / 2],
                     [e[5] / 2, 1 + e[1], e[3] / 2],
                     [e[4] / 2, e[3] / 2, 1 + e[2]]])


def lattice_mutation(p, pos, cell, rng, sigma, v_target, min_len, tries=20):
    """Strain the Minkowski-reduced cell with a random symmetric strain, atoms keep fractional
    coordinates, volume rescaled to v_target. Returns (pos, cell) or None."""
    s = p.search_idx
    pos = np.array(pos, float)
    cell = reduced(cell)
    frac = pos[s] @ np.linalg.inv(cell)
    sgn = np.sign(np.linalg.det(cell))
    for _ in range(tries):
        new = np.asarray(cell) @ strain_matrix(rng, sigma)
        if sgn * np.linalg.det(new) <= 1e-6:
            continue
        new *= (v_target / volume(new)) ** (1 / 3)
        if cell_ok(new, min_len):
            out = pos.copy()
            out[s] = frac @ new
            return out, new
    return None


# ------------------------------------------------------------ fingerprints
def ov_fingerprint(p, pos, cell, rmax=10.0, delta=0.08, sigma=0.03):
    """Oganov-Valle (2009) fingerprint F_AB(R) = sum delta(R-R_ij) / (4 pi R_ij^2 N_A N_B/V Delta) - 1,
    Gaussian-smeared, concatenated over species pairs with weights sqrt(w_AB), w_AB ~ N_A N_B,
    and normalised so that cos distances are dot products."""
    s = p.search_idx
    x = np.asarray(pos, float)[s]
    cell = reduced(cell)
    sym = np.array(p.symbols)[s]
    species = sorted(set(sym))
    spec_id = np.array([species.index(t) for t in sym])
    ns_sp = len(species)
    cnt = np.bincount(spec_id, minlength=ns_sp).astype(float)
    nb = int(round(rmax / delta))
    edges = np.arange(nb + 1) * delta
    vol = volume(cell)
    inv = np.linalg.inv(cell)
    shifts = lattice_shifts(cell, [True, True, True], rmax) @ cell
    hist = np.zeros((ns_sp, ns_sp, nb))
    for i in range(len(x)):
        d0 = x - x[i]
        f = d0 @ inv
        f -= np.round(f)
        d0 = f @ cell
        d = d0[None, :, :] + shifts[:, None, :]
        r = np.sqrt(np.einsum("mnk,mnk->mn", d, d))
        mi, jj = np.nonzero((r < rmax - 3 * sigma) & (r > 1e-6))
        rr = r[mi, jj]
        sj = spec_id[jj]
        b = np.minimum((rr / delta).astype(int), nb - 1)
        lo = 0.5 * (1 + erf((edges[b] - rr) / (np.sqrt(2) * sigma)))
        hi = 0.5 * (1 - erf((edges[b + 1] - rr) / (np.sqrt(2) * sigma)))
        wgt = 1.0 / (4 * np.pi * rr ** 2)
        si = spec_id[i]
        for bins, frac in ((b, 1 - lo - hi), (b - 1, lo), (b + 1, hi)):
            ok = (bins >= 0) & (bins < nb)
            np.add.at(hist[si], (sj[ok], bins[ok]), wgt[ok] * frac[ok])
    vecs = []
    tot = cnt.sum() ** 2
    for A in range(ns_sp):
        for B in range(A, ns_sp):
            F = hist[A, B] / (cnt[A] * cnt[B] / vol * delta) - 1.0
            wab = (1.0 if A == B else 2.0) * cnt[A] * cnt[B] / tot
            vecs.append(np.sqrt(wab) * F)
    v = np.concatenate(vecs)
    return v / (np.linalg.norm(v) + 1e-12)


def cos_distance(f1, f2):
    return float(0.5 * (1.0 - np.dot(f1, f2)))
