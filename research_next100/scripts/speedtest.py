import time, sys, json, os, numpy as np
from ase.build import bulk
from ase.cluster import Icosahedron
from calorine.calculators import CPUNEP
NEP = "/global/cfs/cdirs/m4517/jiaao/software/gpumd/GPUMD/potentials/nep/nep89_20250409/nep89_20250409.txt"
out = {}
t = time.time(); calc = CPUNEP(NEP); out["load_s"] = time.time() - t
for n in [1, 2, 4]:
    b = bulk('Si', 'diamond', 5.43, cubic=True) * (n, n, 2 * n); b.rattle(0.05); b.calc = calc
    b.get_forces(); t = time.time(); k = 0
    while time.time() - t < 5:
        b.positions += np.random.normal(0, 0.005, b.positions.shape); b.get_forces(); k += 1
    dt = (time.time() - t) / k
    out[len(b)] = dict(ms=dt * 1e3, atom_steps_per_s=len(b) / dt)
print(json.dumps(dict(host=os.uname()[1], cpus=os.environ.get("FARM_CPUS"), **{str(k): v for k, v in out.items()})))
