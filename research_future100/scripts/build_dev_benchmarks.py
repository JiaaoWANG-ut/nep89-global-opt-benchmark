"""Build the two dev benchmarks used while algorithms are developed:
su_Pt8_MgO100 (Pt8 on a frozen MgO(001) slab) and ze_Pt4_CHA (Pt4 in a CHA cage)."""
import json, os, sys
import numpy as np
from ase.build import bulk
from ase.io import read, write
from ase import Atoms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gopt.nep import FastNEP
from gopt.context import lbfgs
from gopt.problem import min_image_distances
from calorine.calculators import CPUNEP
from gopt.nep import NEP89
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
import io, contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def calc():
    with contextlib.redirect_stdout(io.StringIO()):
        return CPUNEP(NEP89)

def relax_cell(atoms, fmax=0.01):
    atoms.calc = calc()
    BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=500)
    return atoms

def relax_fixed_cell(atoms, fixed_mask=None, fmax=0.02):
    fn = FastNEP(atoms.get_chemical_symbols(), atoms.cell.array, atoms.positions)
    free = np.ones(len(atoms), bool) if fixed_mask is None else ~fixed_mask
    def fun(x):
        e, f, v, ea = fn.compute(x)
        f[~free] = 0
        return e, f
    x, e, f, conv, n, ab = lbfgs(fun, atoms.positions.copy(), fmax, 3000)
    atoms.positions = x
    print('  relaxed', conv, n, e)
    return atoms

def largest_cavity(atoms, grid=24):
    cell = atoms.cell.array
    g = (np.indices((grid, grid, grid)).reshape(3, -1).T + 0.5) / grid
    pts = g @ cell
    d = min_image_distances(pts, atoms.positions, cell, [True] * 3).min(axis=1)
    k = np.argmax(d)
    return pts[k], d[k]

# ---------------------------------------------------------------- MgO(001)
mgo = relax_cell(bulk('MgO', 'rocksalt', a=4.21, cubic=True))
a = mgo.cell.array[0, 0]
print('MgO a =', a)
slab = mgo.repeat((3, 3, 2))  # 4 atomic layers (2 conventional cells)
slab.center(vacuum=0.0, axis=2)
z = slab.positions[:, 2]
slab.cell[2, 2] = z.max() - z.min() + a / 2 + 18.0
slab.positions[:, 2] -= z.min()
slab.pbc = True
layers = np.round(slab.positions[:, 2] / (a / 2)).astype(int)
fixed = layers <= 1
relax_fixed_cell(slab, fixed)
slab = slab[np.argsort(slab.positions[:, 2], kind='stable')]
d = os.path.join(ROOT, 'benchmarks', 'su_Pt8_MgO100'); os.makedirs(d, exist_ok=True)
write(os.path.join(d, 'substrate.extxyz'), slab)
top = slab.positions[:, 2].max()
c = slab.cell.array
json.dump(dict(name='su_Pt8_MgO100', title='Pt8 / MgO(001)', category='surface',
               description='Pt8 cluster on a frozen 4-layer MgO(001) slab (3x3 conventional cells, 144 atoms).',
               search_symbols=['Pt'] * 8, substrate_file='substrate.extxyz', pbc=[True] * 3,
               region=dict(type='box', lo=[0.0, 0.0, top + 1.6], hi=[c[0, 0], c[1, 1], top + 7.5]),
               budget=30000, fmax=0.05, e_tol=0.02),
          open(os.path.join(d, 'problem.json'), 'w'), indent=1)
print('MgO slab atoms', len(slab), 'top z', top, 'cell', c.diagonal())

# ---------------------------------------------------------------- CHA
cha = read('/tmp/CHA.cif')
print('CHA atoms', len(cha), cha.get_chemical_formula())
cha = relax_cell(cha, fmax=0.02)
center, rad = largest_cavity(cha)
print('cavity center', center, 'radius to nearest atom', rad)
d = os.path.join(ROOT, 'benchmarks', 'ze_Pt4_CHA'); os.makedirs(d, exist_ok=True)
cha.calc = None
write(os.path.join(d, 'substrate.extxyz'), cha)
json.dump(dict(name='ze_Pt4_CHA', title='Pt4 @ CHA', category='zeolite',
               description='Pt4 cluster inside the cha cage of a frozen all-silica chabazite framework (hexagonal cell, 108 atoms).',
               search_symbols=['Pt'] * 4, substrate_file='substrate.extxyz', pbc=[True] * 3,
               region=dict(type='sphere', center=[float(x) for x in center], radius=round(float(rad) - 2.2, 2)),
               budget=20000, fmax=0.05, e_tol=0.02),
          open(os.path.join(d, 'problem.json'), 'w'), indent=1)
