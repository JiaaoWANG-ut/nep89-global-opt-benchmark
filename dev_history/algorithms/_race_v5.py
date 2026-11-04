"""Racing basin hopping: BH whose local relaxations race against the acceptance threshold."""
import numpy as np

from .. import ops
from ._race_moves import rigid_move, site_relocate
from ._race_opt import Relaxer

LABEL = "Racing basin hopping"
FAMILY = "innovative"
REFERENCE = ("Wales & Doye, J. Phys. Chem. A 101, 5111 (1997) (basin hopping); "
             "Maron & Moore, NIPS 6 (1994) / Artif. Intell. Rev. 11, 193 (1997) (racing); "
             "Jamieson & Talwalkar, AISTATS (2016) and Li et al., JMLR 18, 1 (2018) "
             "(successive halving / Hyperband early stopping); Iwamatsu & Okabe, "
             "Chem. Phys. Lett. 399, 396 (2004) (BH with restarts/jumps)")
DESCRIPTION = (
    "Basin hopping in which every local relaxation is a sequential race against the "
    "Metropolis acceptance threshold. The uniform random number of the Metropolis test "
    "is drawn BEFORE relaxing, so the trial is accepted iff its relaxed energy ends below "
    "E_thr = E_cur + kT*Exp(1). A resumable L-BFGS (identical to the shared one) is "
    "advanced one iteration at a time and, after each iteration, an optimistic lower "
    "bound of the final energy, LB = E_k - (a0 + c*max(quasi-Newton predicted decrease, "
    "energy drop over the last 3 iterations)), is compared with the threshold: trials "
    "with LB > E_thr are abandoned (they would be rejected anyway); trials that are "
    "certainly accepted (E_k < E_thr) but cannot become a new global best (LB > E_best) "
    "stop at a loose force threshold; relaxations that converge back into the parent "
    "basin (aligned max deviation < r_ret) are cut. Only potential new best minima are "
    "polished to the problem fmax. Because losing trials are cheap, the driver uses "
    "bolder moves (all-atom displacement with adaptive step, relocation of high-energy "
    "atoms, swaps for alloys, cell strain for bulk) and mixes in raced random-restart "
    "trials (fresh random structures that replace the walker only if they beat its "
    "threshold; rate 0.5 for bulk, min(0.7, 4/n_search) for supported and confined "
    "clusters, min(0.7, 2/n_search) for gas-phase clusters); the walker is restarted "
    "after 15+3*n_search non-improving trials.")
DEFAULTS = dict(kT=0.1, step=0.6, target_accept=0.5, adapt_every=10, strain=0.05,
                a0=0.3, c_est=8.0, kmin=4, f_loose=0.15, r_ret=0.12, f_ret=0.5,
                p_jump=0.2, p_swap=0.15, stag_base=15, stag_per_atom=3.0, rs_cluster=2.0,
                rs_support=4.0, rs_max=0.7, rs_bulk=0.5, eswap=0, p_ejump=0.5,
        jump_sites=40, p_rigid=0.1, q_abort=0.95, n_cal=12, cal_max=400)


class _Walker:
    __slots__ = ("pos", "cell", "e", "ea", "best", "since")


def _parent_dev(p, pos, cell, ppos, pcell, inv0):
    """Max deviation of search atoms from the parent minimum (aligned for clusters)."""
    s = p.search_idx
    if p.variable_cell:
        d = pos[s] @ np.linalg.inv(cell) - ppos[s] @ np.linalg.inv(pcell)
        d -= np.round(d)
        d -= d.mean(axis=0)
        d = d @ cell
        return float(np.sqrt((d * d).sum(axis=1).max()))
    d = pos[s] - ppos[s]
    if inv0 is not None:
        fr = d @ inv0
        fr -= np.round(fr)
        d = fr @ p.cell0
    if p.category == "cluster" and len(s) > 2:
        b = ppos[s]
        a = b + d
        A = a - a.mean(axis=0)
        B = b - b.mean(axis=0)
        u, _, vt = np.linalg.svd(A.T @ B)
        if np.linalg.det(vt.T @ u.T) < 0:
            vt[-1] *= -1
        d = A @ (vt.T @ u.T).T - B
    return float(np.sqrt((d * d).sum(axis=1).max()))


def run(ctx, rng, kT=0.1, step=0.6, target_accept=0.5, adapt_every=10, strain=0.05,
        a0=0.3, c_est=8.0, kmin=4, f_loose=0.15, r_ret=0.12, f_ret=0.5,
        p_jump=0.2, p_swap=0.15, stag_base=15, stag_per_atom=3.0, rs_cluster=2.0,
        rs_support=4.0, rs_max=0.7, rs_bulk=0.5, eswap=0, p_ejump=0.5,
        jump_sites=40, p_rigid=0.1, q_abort=0.95, n_cal=12, cal_max=400, audit=0):
    p = ctx.p
    fmax = p.fmax
    maxit = p.max_relax_steps
    s_idx = p.search_idx
    multi = len(p.species) > 1
    sym_s = np.array(p.symbols)[s_idx]
    vc = p.variable_cell
    rigid_ok = p.category in ("surface", "zeolite") and not vc
    reseat = p.category != "bulk" and not vc
    inv0 = np.linalg.inv(p.cell0) if np.any(p.pbc) else None
    stag = int(stag_base + stag_per_atom * p.n_search)
    # probability of a raced random-restart trial: random sampling pays off for small
    # systems, small-cell crystals and clusters templated by a substrate
    rs_size = rs_cluster if p.category == "cluster" else rs_support
    p_rs = rs_bulk if vc else min(rs_max, rs_size / max(p.n_search, 1))
    st = ctx.stats
    for k in ("trials", "accepted", "aborted", "returned", "early", "restarts", "bad",
              "steps_trials", "rs_trials", "rs_accepted"):
        st.setdefault(k, 0)

    # online calibration: per fmax bin, distribution of the remaining drop
    # D = E_k - E_final observed on this run's completed relaxations
    edges = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
    nbin = len(edges) + 1
    cal = [[] for _ in range(nbin)]
    qcal = np.full(nbin, np.nan)

    def calibrate(eh, fh, e_final):
        b = np.searchsorted(edges, fh)
        mx = np.full(nbin, -np.inf)
        np.maximum.at(mx, b, np.asarray(eh) - e_final)
        for j in np.flatnonzero(np.isfinite(mx)):
            L = cal[j]
            L.append(float(mx[j]))
            if len(L) > cal_max:
                del L[0]
            if len(L) >= n_cal:
                qcal[j] = np.quantile(L, q_abort)

    def race(R, thr, par):
        """Advance relaxation R until its race is decided. Returns a status string."""
        eh = R.e_hist
        fh = [R.fm]
        while True:
            if R.fm < fmax or R.it >= maxit:
                calibrate(eh, fh, R.e)
                return "conv" if R.fm < fmax else "maxit"
            if R.bad:
                return "bad"
            R.step()
            fh.append(R.fm)
            e = R.e
            q = qcal[np.searchsorted(edges, R.fm)]
            if q == q:
                lb = e - q
            else:
                d3 = eh[-4] - e if len(eh) >= 4 else eh[0] - e
                lb = e - (a0 + c_est * max(R.pred, d3, 0.0))
            if thr is not None:
                if R.it >= kmin and lb > thr:
                    return "abort"
                if par is not None and R.fm < f_ret and e - par.e < 0.5:
                    if _parent_dev(p, R.pos, R.cell, par.pos, par.cell, inv0) < r_ret:
                        return "return"
            if R.fm < f_loose and (thr is None or e < thr) and lb > ctx.best_e:
                return "early"

    def finish(R, status):
        ctx.n_relax += 1
        if status in ("conv", "maxit"):
            ctx.minima.append(round(float(R.e), 5))

    def random_start():
        for _ in range(5):
            try:
                return p.random_structure(rng)
            except Exception:
                pass
        if ctx.best_pos is None:
            raise RuntimeError("cannot generate a structure")
        cell = np.array(ctx.best_cell)
        return ops.displace(p, ctx.best_pos, rng, 1.0, cell=cell), cell

    def new_walker():
        pos, cell = random_start()
        R = Relaxer(ctx, pos, cell)
        status = race(R, None, None)
        finish(R, status)
        w = _Walker()
        w.pos, w.cell, w.e = np.array(R.pos), np.array(R.cell), float(R.e)
        w.ea = ctx.atomic_energies()
        w.best, w.since = w.e, 0
        return w

    def move(w):
        if rng.random() < p_rs:
            pos, cell = random_start()
            return "rs", pos, cell
        pos, cell = np.array(w.pos), np.array(w.cell)
        r = rng.random()
        kind = "disp"
        if multi and r < p_swap:
            kind = "swap"
            ea = w.ea
            if eswap and ea is not None and rng.random() < eswap:
                # unlike pair of atoms that are both weakly bound for their species
                es = ea[s_idx]
                z = np.zeros(len(s_idx))
                for sp in p.species:
                    m = sym_s == sp
                    z[m] = (es[m] - es[m].mean()) / max(es[m].std(), 1e-3)
                wgt = np.exp(np.clip(z, -10, 10))
                i = rng.choice(len(s_idx), p=wgt / wgt.sum())
                cand = np.flatnonzero(sym_s != sym_s[i])
                wc = wgt[cand]
                j = cand[rng.choice(len(cand), p=wc / wc.sum())]
                a, b = s_idx[i], s_idx[j]
                pos[[a, b]] = pos[[b, a]]
            else:
                pos = ops.swap_unlike(p, pos, rng, 1)
            pos = ops.displace(p, pos, rng, 0.1, cell=cell)
        elif not vc and r < p_swap + p_jump and p.n_search > 2:
            kind = "jump"
            u = rng.random()
            nj = 1 if u < 0.6 else (2 if u < 0.85 or p.n_search < 7 else 3)
            ea = w.ea
            if ea is not None and rng.random() < p_ejump:
                es = ea[s_idx]
                z = np.zeros(len(s_idx))
                for sp in p.species:
                    m = sym_s == sp
                    z[m] = (es[m] - es[m].mean()) / max(es[m].std(), 1e-3)
                wgt = np.exp(np.clip(z, -10, 10))
                idx = rng.choice(s_idx, size=nj, replace=False, p=wgt / wgt.sum())
            else:
                idx = rng.choice(s_idx, size=nj, replace=False)
            new = site_relocate(p, pos, rng, idx, cell, jump_sites) if jump_sites else None
            pos = ops.relocate_atoms(p, pos, rng, idx, cell) if new is None else new
            pos = ops.repair_overlaps(p, pos, cell)
        elif rigid_ok and r < p_swap + p_jump + p_rigid:
            kind = "rigid"
            pos = rigid_move(p, pos, rng, cell)
        else:
            if vc:
                pos, cell = ops.strain_cell(pos, cell, rng, strain)
            pos = ops.displace(p, pos, rng, step, cell=cell)
        if reseat:
            lost = ops.detached_atoms(p, pos, cell)
            if len(lost):
                pos = ops.relocate_atoms(p, pos, rng, lost, cell)
        return kind, pos, cell

    w = new_walker()
    n_acc = n_try = 0
    while True:
        kind, pos, cell = move(w)
        thr = w.e + kT * rng.exponential()
        n0 = ctx.nsteps
        b0 = ctx.best_e
        R = Relaxer(ctx, pos, cell)
        status = race(R, thr, None if kind == "rs" else w)
        if audit and status == "abort":
            ek = R.e
            while R.fm >= fmax and R.it < maxit and not R.bad:
                R.step()
            au = st.setdefault("audit", [0, 0, 0])
            au[0] += 1
            au[1] += int(R.e < thr)
            au[2] += int(R.e < b0 - 1e-3)
        finish(R, status)
        kst = st.setdefault("k_" + kind, [0, 0, 0, 0])
        kst[0] += 1
        kst[1] += int(status in ("conv", "maxit", "early") and R.e < thr)
        kst[2] += int(ctx.best_e < b0 - 1e-3)
        kst[3] += ctx.nsteps - n0
        st["trials"] += 1
        if kind == "rs":
            st["rs_trials"] += 1
        st["steps_trials"] += ctx.nsteps - n0
        n_try += 1
        acc = False
        if status in ("conv", "maxit", "early"):
            if R.e < thr:
                acc = True
                if status == "early":
                    st["early"] += 1
        elif status == "abort":
            st["aborted"] += 1
        elif status == "return":
            st["returned"] += 1
            n_acc += 1  # back in the parent basin: the move was too timid
        else:
            st["bad"] += 1
        if kind == "rs":
            n_try -= 1  # random restarts do not enter the step-size control
            if acc:
                st["rs_accepted"] += 1
        elif acc:
            n_acc += 1
        if acc:
            st["accepted"] += 1
            w.pos, w.cell, w.e = np.array(R.pos), np.array(R.cell), float(R.e)
            w.ea = ctx.atomic_energies()
        if w.e < w.best - p.e_tol:
            w.best, w.since = w.e, 0
        else:
            w.since += 1
        if kind != "rs" and n_try % adapt_every == 0:
            step *= 1.1 if n_acc / n_try > target_accept else 0.9
            step = float(np.clip(step, 0.1, 2.0))
        st.update(step=round(step, 4), accept_rate=round(n_acc / max(n_try, 1), 4))
        if w.since >= stag:
            st["restarts"] += 1
            w = new_walker()
