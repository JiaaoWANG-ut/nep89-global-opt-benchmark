"""Figures of the report (matplotlib, PNG)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from .data import CATS, DEV, LOG_RANGE  # noqa: E402

TYPE_COLOR = {"baseline": "#4C72B0", "innovative": "#DD8452"}
TYPE_STYLE = {"baseline": "--", "innovative": "-"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path, dpi=110):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def alg_colors(n):
    cm = [plt.get_cmap("tab20")(i) for i in range(20)] + [plt.get_cmap("tab20b")(i) for i in range(20)]
    return [cm[(2 * i) % 20 + (i // 10) % 2] if i < 20 else cm[i] for i in range(n)]


def _type_legend(ax, loc="lower right"):
    ax.legend(handles=[Patch(color=c, label=t) for t, c in TYPE_COLOR.items()], loc=loc, frameon=False)


def rank_bars(names, types, score, lo, hi, path, xlabel):
    order = np.argsort(-np.nan_to_num(score, nan=-1))
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(order) + 1))
    y = np.arange(len(order))[::-1]
    err = np.clip(np.nan_to_num([score[order] - lo[order], hi[order] - score[order]]), 0, None)
    ax.barh(y, score[order], color=[TYPE_COLOR[types[k]] for k in order],
            xerr=err, capsize=2, error_kw=dict(lw=0.8))
    for yy, k in zip(y, order):
        if np.isfinite(score[k]):
            x = max(score[k], hi[k] if np.isfinite(hi[k]) else score[k])
            ax.text(x + 1.2, yy, f"{score[k]:.1f}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([names[k] for k in order])
    ax.set_xlim(0, 110)
    ax.set_xlabel(xlabel)
    _type_legend(ax)
    _save(fig, path)


def heatmap(st, order, names, path):
    e = np.where(np.isnan(st.ert), np.inf, st.ert)
    best = e.min(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log10(e / best[:, None])
    L = lr[:, order]
    M = np.ma.masked_where(~np.isfinite(L), np.clip(L, 0, LOG_RANGE))
    nb, na = M.shape
    fig, ax = plt.subplots(figsize=(0.42 * na + 3, 0.27 * nb + 1.6))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#3a3a3a")
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=LOG_RANGE, aspect="auto")
    for i in range(nb):
        for jj, j in enumerate(order):
            v = st.ert[i, j]
            if np.isnan(v):
                txt, col = "", "w"
            elif not np.isfinite(v):
                txt, col = "x", "#bbbbbb"
            else:
                r = v / best[i]
                txt = "1" if r < 1.005 else (f"{r:.1f}" if r < 9.95 else f"{r:.0f}")
                col = "w" if np.log10(r) > 0.6 * LOG_RANGE else "k"
            ax.text(jj, i, txt, ha="center", va="center", fontsize=5.5, color=col)
    cats = [st.probs[b].category for b in st.benches]
    for i in range(1, nb):
        if cats[i] != cats[i - 1]:
            ax.axhline(i - 0.5, color="k", lw=1.2)
    ax.set_yticks(range(nb))
    ax.set_yticklabels([b + (" *" if b in DEV else "") for b in st.benches], fontsize=7)
    ax.set_xticks(range(na))
    ax.set_xticklabels([names[j] for j in order], rotation=60, ha="right", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("log10(ERT / best ERT)   (dark grey x: never found)")
    ax.spines[:].set_visible(False)
    _save(fig, path, dpi=120)


def _legend(ax, others):
    h, l = ax.get_legend_handles_labels()
    h, l = h[::-1], l[::-1]
    if others:
        h.append(Line2D([], [], color="#c8c8c8", lw=0.8))
        l.append("others")
    ax.legend(h, l, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)


def perf_profile(st, subset, order, names, types, colors, path, hl=None):
    rows = st.rows(subset)
    e = np.where(np.isnan(st.ert), np.inf, st.ert)[rows]
    r = e / e.min(1)[:, None]
    tau = np.logspace(0, LOG_RANGE, 400)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for j in order[::-1]:
        on = hl is None or j in hl
        ax.step(tau, (r[:, j][:, None] <= tau[None, :]).mean(0), where="post",
                color=colors[j] if on else "#c8c8c8", ls=TYPE_STYLE[types[j]], lw=1.5 if on else 0.8,
                label=names[j] if on else None, zorder=3 if on else 1)
    _legend(ax, hl is not None and len(hl) < len(order))
    ax.set_xscale("log")
    ax.set_xlabel("tau  (ERT within a factor tau of the best algorithm)")
    ax.set_ylabel("fraction of benchmarks")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, which="both")
    _save(fig, path)


def ecdf(st, subset, order, names, types, colors, path, hl=None):
    rows = np.flatnonzero(st.rows(subset))
    x = np.logspace(-3, 0, 400)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for j in order[::-1]:
        F = np.mean([(st.hitfrac[i, j][:, None] <= x[None, :]).mean(0) for i in rows], 0)
        on = hl is None or j in hl
        ax.plot(x, F, color=colors[j] if on else "#c8c8c8", ls=TYPE_STYLE[types[j]], lw=1.5 if on else 0.8,
                label=names[j] if on else None, zorder=3 if on else 1)
    ax.set_xscale("log")
    ax.set_xlabel("ionic steps / budget")
    ax.set_ylabel("fraction of runs that found the global minimum")
    ax.set_ylim(0, 1.02)
    _legend(ax, hl is not None and len(hl) < len(order))
    ax.grid(alpha=0.3, which="both")
    _save(fig, path)


def category_matrix(st, order, names, path):
    cats = [c for c in CATS if st.rows(c).any()]
    cols = cats + ["test", "all"]
    M = np.array([st.scores(c) for c in cols]).T[order]
    fig, ax = plt.subplots(figsize=(0.9 * len(cols) + 2.2, 0.27 * len(order) + 1))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for k in range(M.shape[1]):
            if np.isfinite(M[i, k]):
                ax.text(k, i, f"{M[i, k]:.0f}", ha="center", va="center", fontsize=7,
                        color="k" if M[i, k] > 55 else "w")
    labels = [f"{c}\n({st.rows(c).sum()})" for c in cols]
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[j] for j in order], fontsize=7.5)
    ax.axvline(len(cats) - 0.5, color="w", lw=2)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("efficiency score")
    ax.spines[:].set_visible(False)
    _save(fig, path)


def cd_diagram(avg, cd, names, path, title=""):
    k = len(avg)
    order = np.argsort(avg)
    r = avg[order]
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(9, 0.18 * k + 1.6))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(-(k // 2 + 3) * 0.6, 1.2)
    ax.axis("off")
    ax.hlines(0, lo, hi, color="k", lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0, 0.15, color="k", lw=0.8)
        ax.text(t, 0.25, str(t), ha="center", fontsize=7)
    half = (k + 1) // 2
    for n, (j, v) in enumerate(zip(order, r)):
        left = n < half
        y = -(n if left else k - 1 - n) * 0.6 - 1.4
        xt = lo - 0.4 if left else hi + 0.4
        ax.plot([v, v, xt], [0, y, y], color="k", lw=0.7)
        ax.text(xt + (-0.05 if left else 0.05), y, f"{names[j]} ({v:.1f})", va="center",
                ha="right" if left else "left", fontsize=7)
    cliques, last = [], -1
    for a in range(k):
        b = a
        while b + 1 < k and r[b + 1] - r[a] < cd:
            b += 1
        if b > a and b > last:
            cliques.append((a, b))
            last = b
    for n, (a, b) in enumerate(cliques):
        y = -0.35 - 0.18 * (n % 6)
        ax.hlines(y, r[a] - 0.05, r[b] + 0.05, color="#c0392b", lw=2.5)
    ax.hlines(0.9, lo, lo + cd, color="#c0392b", lw=2)
    ax.text(lo + cd / 2, 1.0, f"CD = {cd:.2f}", ha="center", fontsize=7, color="#c0392b")
    if title:
        ax.set_title(title, fontsize=9)
    _save(fig, path)


def overhead_bars(over, ms_bh, order, names, types, path):
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(order) + 1))
    y = np.arange(len(order))[::-1]
    vals = np.array([over[j] for j in order])
    ax.barh(y, vals - 1, left=1, color=[TYPE_COLOR[types[j]] for j in order])
    for yy, v in zip(y, vals):
        ax.annotate(f"{v:.2f}x", (max(v, 1), yy), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(min(vals.min(), 1) / 1.1, max(vals.max(), 1) * 1.5)
    ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels([names[j] for j in order])
    ax.axvline(1, color="k", lw=0.8)
    ax.set_xlabel(f"wall time per ionic step relative to basin hopping (BH median {ms_bh:.2f} ms/step)")
    _type_legend(ax)
    _save(fig, path)


def sr_curves(st, cat, order, names, types, colors, path):
    rows = [i for i, b in enumerate(st.benches) if st.probs[b].category == cat]
    if not rows:
        return False
    nc = 4
    nr = (len(rows) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.4 * nr + 0.9), squeeze=False)
    for ax in axes.flat[len(rows):]:
        ax.axis("off")
    for ax, i in zip(axes.flat, rows):
        b = st.benches[i]
        budget = st.probs[b].budget
        t = np.logspace(1, np.log10(budget), 200)
        for j in order:
            if (i, j) not in st.hitfrac:
                continue
            h = st.hitfrac[i, j] * budget
            ax.plot(t, (h[:, None] <= t[None, :]).mean(0), color=colors[j],
                    ls=TYPE_STYLE[types[j]], lw=1.0)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_title(b + (" *" if b in DEV else ""), fontsize=8)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("ionic steps", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("success rate", fontsize=7)
    handles = [Line2D([], [], color=colors[j], ls=TYPE_STYLE[types[j]], label=names[j]) for j in order]
    fig.legend(handles=handles, loc="lower center", ncol=min(8, len(order)), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02 - 0.012 * (len(order) // 8)))
    fig.tight_layout(rect=(0, 0.04 + 0.012 * (len(order) // 8), 1, 1))
    _save(fig, path, dpi=100)
    return True
