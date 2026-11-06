#!/usr/bin/env python3
"""Reproduce the bounded, non-search validation for the adjacent problem."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from ase.build import bulk
from ase.cluster import Icosahedron, Octahedron
from ase.neighborlist import neighbor_list
from gopt.context import Context
from gopt.nep import FastNEP, NEP89
from gopt.problem import Problem
from gopt.run import run_one


here = Path(__file__).resolve().parent
p = Problem(here.name, root=str(here.parent))
start_cpu = time.process_time()


def describe(pos, cell):
    a = p.to_atoms(pos, cell)
    distances = neighbor_list("d", a, 8.0)
    nearest = float(np.min(distances))
    guard = ctx.guard(pos, cell)[0]
    image_distance = None
    if not p.variable_cell:
        shifts = np.array([(i, j, k) for i in (-1, 0, 1)
                           for j in (-1, 0, 1) for k in (-1, 0, 1)
                           if (i, j, k) != (0, 0, 0)])
        delta = pos[:, None, None, :] - pos[None, :, None, :] - shifts[None, None, :, :] @ cell
        image_distance = float(np.linalg.norm(delta, axis=-1).min())
    return {
        "minimum_periodic_distance_A": nearest,
        "minimum_cluster_image_distance_A": image_distance,
        "guard_eV": float(guard),
        "components": p.cluster_components(pos, cell) if p.connected else None,
        "volume_per_atom_A3": abs(float(np.linalg.det(cell))) / p.n if p.variable_cell else None,
    }


def relax(label, pos, cell):
    before = describe(pos, cell)
    initial_step = ctx.nsteps
    result = ctx.relax(pos, cell, steps=100)
    energy, forces = ctx.evaluate(result.pos, result.cell)
    raw = ctx.nep.compute(result.pos, result.cell if p.variable_cell else None)[0]
    after = describe(result.pos, result.cell)
    assert np.isfinite(energy) and np.isfinite(forces).all()
    assert abs(energy - raw - after["guard_eV"]) < 1e-6
    assert abs(energy - result.e) < 1e-6
    assert result.converged and after["guard_eV"] < 1e-8
    assert after["minimum_periodic_distance_A"] >= 0.77 * 2 * p.rcov[0]
    if p.variable_cell:
        assert 0.6 <= after["volume_per_atom_A3"] / p.vpa <= 1.6
    else:
        assert after["components"] == 1 and after["minimum_cluster_image_distance_A"] > 6.0
    return {
        "label": label, "initial": before, "final": after,
        "energy_eV": energy, "raw_nep_eV": raw,
        "converged": result.converged, "relax_evaluations": ctx.nsteps - initial_step - 1,
        "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
        "stress_eV_A3_max": float(np.abs(ctx.stress(result.cell)).max()) if p.variable_cell else None,
    }


assert p.name == here.name and len(p.symbols) == p.n
assert p.n_sub == 0 and p.free_mask.all() and p.pbc.all()
assert np.linalg.det(p.cell0) > 0
ctx = Context(p, budget=2000)
results = []
if p.category == "cluster":
    assert p.n == 13 and not p.variable_cell and p.connected
    assert min(np.diag(p.cell0)) - 2 * p.region["radius"] > 12.0
    for label, atoms in (
        ("icosahedron", Icosahedron("Cu", 2, latticeconstant=3.615)),
        ("cuboctahedron", Octahedron("Cu", 3, 1, latticeconstant=3.615)),
    ):
        atoms.center(vacuum=None)
        pos = atoms.positions - atoms.positions.mean(axis=0) + np.array(p.region["center"])
        results.append(relax(label, pos, p.cell0))
    solid = bulk("Cu", "fcc", a=3.615, cubic=True)
    solid_nep = FastNEP(solid.get_chemical_symbols(), solid.cell.array, solid.positions)
    bulk_e_atom = solid_nep.compute(solid.positions)[0] / len(solid)
else:
    assert p.category == "bulk" and p.variable_cell and p.n == 8
    atoms = bulk("Ge", "diamond", a=5.658, cubic=True)
    assert atoms.get_chemical_symbols() == p.symbols
    results.append(relax("diamond conventional cell", atoms.positions, atoms.cell.array))
    bulk_e_atom = None

for seed in (0, 1):
    pos, cell = p.random_structure(np.random.default_rng(seed))
    assert p.min_dist_ok(pos, cell)
    assert all(p.in_region(x, cell) for x in pos)
    if p.connected:
        assert p.cluster_components(pos, cell) == 1
    else:
        assert 0.9 <= abs(np.linalg.det(cell)) / (p.n * p.vpa) <= 1.3
    results.append(relax(f"random seed {seed}", pos, cell))

model_sha = hashlib.sha256(Path(NEP89).read_bytes()).hexdigest()
spec_sha = hashlib.sha256((here / "problem.json").read_bytes()).hexdigest()
smoke = []
for seed in (0, 1):
    run = run_one(p, "rs", seed, budget=80)
    assert run["status"] == "ok" and run["nsteps"] == 80 and run["bench_hash"] == p.hash
    smoke.append({k: run[k] for k in ("seed", "status", "nsteps", "best_e", "bench_hash", "alg_hash")})
print(json.dumps({
    "problem": p.name, "problem_hash": p.hash, "problem_json_sha256": spec_sha,
    "model": NEP89, "model_sha256": model_sha,
    "budget": p.budget, "fmax": p.fmax, "e_tol": p.e_tol,
    "reference_cu_fcc_unrelaxed_eV_per_atom": bulk_e_atom,
    "evaluations": ctx.nsteps, "bad_evaluations": ctx.n_bad,
    "cpu_seconds": round(time.process_time() - start_cpu, 3),
    "results": results, "rs_smoke_80_steps_each": smoke,
}, indent=2, allow_nan=False))
