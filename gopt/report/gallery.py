"""Pictures of the putative global minima (ASE matplotlib renderer)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.data import covalent_radii  # noqa: E402
from ase.data.colors import jmol_colors  # noqa: E402
from ase.visualize.plot import plot_atoms  # noqa: E402

VIEW = {"cluster": "10x,20y,0z", "surface": "-72x,0y,0z", "zeolite": "15x,20y,0z",
        "bulk": "10x,20y,0z"}


def _mic(v, cell, pbc):
    if not np.any(pbc):
        return v
    f = v @ np.linalg.inv(cell)
    f[..., pbc] -= np.round(f[..., pbc])
    return f @ cell


def scene(at, p):
    """Atoms to draw (search atoms unwrapped + nearby substrate) and per-atom 'is search'."""
    cat = p.category
    if cat == "bulk":
        a = at.copy()
        a.wrap()
        if len(a) < 24:
            a = a.repeat((2, 2, 2))
        return a, np.ones(len(a), bool), True
    cell, pbc = at.cell.array, p.pbc
    pos = at.positions
    s = p.search_idx
    sp = pos[s[0]] + _mic(pos[s] - pos[s[0]], cell, pbc)
    c = sp.mean(0)
    keep_pos, keep_num, is_s = [sp], [at.numbers[s]], [np.ones(len(s), bool)]
    if p.n_sub:
        v = _mic(pos[: p.n_sub] - c, cell, pbc)
        d = np.linalg.norm(v[:, :2], axis=1) if cat == "surface" else np.linalg.norm(v, axis=1)
        m = d < (8.0 if cat == "surface" else 5.5)
        keep_pos.append(c + v[m])
        keep_num.append(at.numbers[: p.n_sub][m])
        is_s.append(np.zeros(m.sum(), bool))
    a = Atoms(numbers=np.concatenate(keep_num), positions=np.concatenate(keep_pos))
    return a, np.concatenate(is_s), False


def render(at, p, path, title=None):
    a, is_s, cellshow = scene(at, p)
    col = jmol_colors[a.numbers].copy()
    lum = col @ np.array([0.30, 0.59, 0.11])
    dark = is_s & (lum > 0.6)
    col[dark] *= (0.6 / lum[dark])[:, None]
    if not is_s.all():
        col[~is_s] = 0.3 * col[~is_s] + 0.7
    rad = covalent_radii[a.numbers] * np.where(is_s, 0.95, 0.6)
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    plot_atoms(a, ax, rotation=VIEW.get(p.category, "10x,20y,0z"), radii=rad,
               colors=[tuple(c) for c in col], show_unit_cell=2 if cellshow else 0)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8)
    fig.savefig(path, dpi=90, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


def min_dist_ratio(at, p):
    """min over pairs with a search atom of d / (r_cov,i + r_cov,j)."""
    from ..problem import min_image_distances
    s = p.search_idx
    d = min_image_distances(at.positions[s], at.positions, at.cell.array, p.pbc)
    d[np.arange(len(s)), s] = np.inf
    return float((d / (p.rcov[s][:, None] + p.rcov[None, :])).min())
