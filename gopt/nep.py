"""Thin, low-overhead wrapper around calorine's NEP CPU engine (``_nepy``).

The ASE calculator in calorine copies and compares Atoms objects on every call,
which costs ~1-2 ms per call and dominates for small clusters. ``FastNEP`` talks
to the C++ object directly. One instance is bound to a fixed number of atoms and
a fixed species list (positions and cell can change).
"""
import contextlib
import os
import sys

import numpy as np
from ase.data import atomic_masses, atomic_numbers

import _nepy

NEP89 = os.environ.get(
    "GOPT_NEP_MODEL",
    "/global/cfs/cdirs/m4517/jiaao/software/gpumd/GPUMD/potentials/nep/"
    "nep89_20250409/nep89_20250409.txt",
)


@contextlib.contextmanager
def _quiet_stdout():
    """Silence C/C++ writes to fd 1 (NEPY prints the model header)."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def _cell9(cell):
    c = np.asarray(cell, float).reshape(3, 3).flatten()
    return [c[0], c[3], c[6], c[1], c[4], c[7], c[2], c[5], c[8]]


class FastNEP:
    """Energy, forces and virial from NEP for a fixed atom count/species list."""

    def __init__(self, symbols, cell, positions, model=NEP89):
        self.symbols = list(symbols)
        self.n = len(self.symbols)
        self.model = model
        masses = [float(atomic_masses[atomic_numbers[s]]) for s in self.symbols]
        self.cell = np.array(cell, float).reshape(3, 3)
        pos = np.asarray(positions, float).reshape(self.n, 3)
        with _quiet_stdout():
            self.nepy = _nepy.NEPY(model, self.n, _cell9(self.cell), self.symbols,
                                   pos.T.ravel().tolist(), masses)

    def compute(self, positions, cell=None):
        """Return (energy, forces[N,3], virial[3,3], per-atom energies[N])."""
        if cell is not None:
            cell = np.asarray(cell, float).reshape(3, 3)
            if not np.array_equal(cell, self.cell):
                self.cell = cell.copy()
                self.nepy.set_cell(_cell9(cell))
        self.nepy.set_positions(np.asarray(positions, float).T.ravel().tolist())
        e, f, v = self.nepy.get_potential_forces_and_virials()
        e = np.asarray(e)
        forces = np.asarray(f).reshape(3, self.n).T.copy()
        virial = np.asarray(v).reshape(9, self.n).sum(axis=1).reshape(3, 3)
        return float(e.sum()), forces, virial, e
