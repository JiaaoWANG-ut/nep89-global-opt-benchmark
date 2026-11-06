# Third 100-seed extension: isolated source

This directory is a separate snapshot of the 26-method research campaign. It
does not alter the original 22-method registry at the repository root. See
`benches-eval/README.md` and `benches-eval/protocol.json` for the campaign
design, limitations and frozen inputs.

The v1.3 results archive contains 10,400 formal records and 326 development
and pilot records. To reproduce analysis, extract it here so that `results/`
contains the campaign directories; obtain the NEP89 model and dependencies as
described in the repository's main README. Set `GOPT_NEP_MODEL` to the actual
model path, `GOPT_BENCH_DIR` to this directory's `benches-eval`, and
`PYTHONPATH` to this directory. The model must match the sealed hashes.
Run `python scripts/research_freeze.py` before analysis. The report can be
rebuilt using `scripts/extended100_report.py` followed by
`scripts/audit_final_refs.py`, `dev_history/compare_ablations.py` and
`dev_history/publish_next100.py`; use `--help` on the scripts with command-line
arguments. `dev_history/audit_candidate_structures.py` separately rechecks
saved best structures in the reference window. Intermediate trace geometries
were not saved and cannot be audited from the archive.

For the experimental-family unit tests, set `GOPT_BENCH_DIR` to `benches-dev`
instead and run the two `test_*.py` files under
`dev_history/next_algorithms/` with Python.

The supplied Slurm scripts document the original NERSC execution and include
site-specific paths and account settings; adapt them for other environments.
The SHA-256 manifest seals full benchmark inputs and the protocol in addition
to the original landscape hashes. Scores are specific to these four selected
systems and are not directly comparable to other benchmark sets.
