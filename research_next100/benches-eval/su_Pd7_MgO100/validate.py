"""Reproduce the bounded geometry and NEP smoke checks from the project root."""
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read
from ase.geometry import find_mic

from gopt.context import Context
from gopt.nep import NEP89
from gopt.problem import Problem, _images, min_image_distances
from gopt.run import run_one


here = Path(__file__).resolve().parent
spec = json.loads((here / "problem.json").read_text())
p = Problem(here.name, root=str(here.parent))
host = read(here / "substrate.extxyz")
model_header = Path(NEP89).open().readline().split()
assert set(p.symbols) <= set(model_header[2:])
assert p.name == here.name and np.all(p.pbc) and p.n_sub == len(host)
assert p.free_mask[:p.n_sub].sum() == 0 and p.free_mask[p.n_sub:].all()
assert abs(np.linalg.det(p.cell0)) > 1
print("name", p.name, "problem_hash", p.hash, "model_sha256",
      hashlib.sha256(Path(NEP89).read_bytes()).hexdigest())
print("host_sha256", hashlib.sha256((here / "substrate.extxyz").read_bytes()).hexdigest())
print("formula", host.get_chemical_formula(), "host_atoms", p.n_sub,
      "search_atoms", p.n_search, "cell", np.round(p.cell0, 5).tolist())
dhost = min_image_distances(p.sub_pos, p.sub_pos, p.cell0, p.pbc)
np.fill_diagonal(dhost, np.inf)
print("host_minimum_pair_A", round(float(dhost.min()), 4))
assert dhost.min() > 1.0
syms = np.asarray(p.sub_symbols)
if p.category == "zeolite":
    sio = dhost[np.ix_(syms == "Si", syms == "O")]
    coordination = (sio < 2.1).sum(axis=1)
    print("Si_O_bonds_A", np.round([sio.min(), sio.min(axis=1).max()], 4).tolist(),
          "Si_O_coordination_range", [int(coordination.min()), int(coordination.max())])
    assert np.all(coordination == 4)
else:
    mgo = dhost[np.ix_(syms == "Mg", syms == "O")]
    print("Mg_O_nearest_A", np.round([mgo.min(), mgo.min(axis=1).max()], 4).tolist())
if p.z_wall:
    w = p.z_wall
    thickness = np.ptp(p.sub_pos @ w["normal"])
    vacuum = w["period"] - thickness
    print("slab_thickness_A", round(thickness, 4), "vacuum_A", round(vacuum, 4),
          "top_A", round(w["top"], 4), "wall_start_A", round(w["top"] + w["height"], 4))
    assert vacuum > 15 and spec["region"]["hi"][2] < w["top"] + w["height"]
else:
    center = np.array(p.region["center"])
    clearance = min_image_distances(center, p.sub_pos, p.cell0, p.pbc).min()
    print("cage_center_clearance_A", round(float(clearance), 4),
          "initial_sphere_radius_A", p.region["radius"])
    assert clearance > p.region["radius"]

# Fixed, predeclared seeds, all reported: these are short smoke tests, not a search.
for seed in (0, 1, 2):
    pos, cell = p.random_structure(np.random.default_rng(seed))
    assert p.min_dist_ok(pos) and p.cluster_components(pos) == 1
    ctx = Context(p, budget=100)
    e0, f0 = ctx.evaluate(pos)
    assert np.isfinite(e0) and np.isfinite(f0).all() and ctx.n_bad == 0
    raw_e, raw_f, _, _ = ctx.nep.compute(pos)
    eg = ctx.guard(pos, cell)[0]
    assert np.isfinite(raw_e) and np.isfinite(raw_f).all()
    assert abs(e0 - raw_e - eg) < 1e-6
    if seed == 0 and p.z_wall:
        w = p.z_wall
        probe = pos.copy()
        heights = probe[p.search_idx] @ w["normal"]
        probe[p.search_idx] += (w["top"] + 1 - heights)[:, None] * w["normal"]
        assert ctx.guard._wall(probe) is None
        probe[p.search_idx[0]] += (w["height"] - 0.9) * w["normal"]
        assert ctx.guard._wall(probe) is not None
        print("wall_probe", "inactive_below_and_active_above")
    r = ctx.relax(pos, steps=50)
    assert ctx.n_bad == 0 and np.isfinite(r.e)
    assert np.allclose(r.pos[:p.n_sub], p.sub_pos)
    dist = min_image_distances(r.pos[p.search_idx], r.pos, cell, p.pbc)
    dist[np.arange(p.n_search), p.search_idx] = np.inf
    sh = float(dist[:, :p.n_sub].min())
    ss = float(dist[:, p.n_sub:].min())
    xyz = r.pos[p.search_idx]
    delta, _ = find_mic(xyz - xyz[0], cell, pbc=p.pbc)
    xyz = xyz[0] + delta
    shifts = _images(p.pbc) @ cell
    shifts = shifts[np.linalg.norm(shifts, axis=1) > 1e-8]
    image_min = np.linalg.norm(xyz[:, None, None, :] -
                               (xyz[None, :, None, :] + shifts[None, None, :, :]),
                               axis=-1).min()
    guard = ctx.guard(r.pos, cell)[0]
    extra = {}
    if p.z_wall:
        h = r.pos[p.search_idx] @ p.z_wall["normal"]
        extra["height_range_A"] = np.round([h.min(), h.max()], 4).tolist()
    else:
        cage_d = min_image_distances(r.pos[p.search_idx], center, cell, p.pbc)
        extra["cage_center_max_A"] = round(float(cage_d.max()), 4)
    print("seed", seed, "initial_e", round(e0, 6), "final_e", round(r.e, 6),
          "evals", ctx.nsteps, "converged", r.converged, "fmax", round(r.fmax, 5),
          "guard_wall_e", round(guard, 8), "host_min_A", round(sh, 4),
          "cluster_min_A", round(ss, 4), "image_min_A", round(float(image_min), 4),
          "components", p.cluster_components(r.pos), extra)

result = run_one(p, "bh", seed=0, budget=40)
assert result["status"] == "ok" and result["bench_hash"] == p.hash
assert result["nsteps"] == 40
print("BH runner smoke", "status", result["status"], "steps", result["nsteps"])
