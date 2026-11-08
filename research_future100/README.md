# Fourth 100-seed extension: isolated research source

This directory is an isolated snapshot for a five-system, 31-method campaign.
It does not change the original method registry at the repository root.
The experimental additions are two algorithm families and three mechanism
controls, not five independent inventions. The methods combine existing
ideas; neither literature novelty nor general superiority is established.
See `benches-eval/protocol.json` and `benches-eval/pilot_audit.json` for the
design, selection, and limitations.

The results archive contains 15,500 formal records and 50 selected
pilot records. Extract it here so `results/` contains the campaign
directories. The separate system-pilot archive retains all 60 pilot records,
including the held-out MFI system. Obtain the NEP89 model and dependencies as
described in the repository's main README. Set `GOPT_NEP_MODEL` to the model
path, `GOPT_BENCH_DIR` to this directory's `benches-eval`, and `PYTHONPATH` to
this directory. The model must match the sealed hashes.

From this directory, verify the frozen inputs with:

```sh
python scripts/research_freeze.py
```

Rebuild and independently check the report:

```sh
python scripts/extended100_report.py --campaign future100 --seed-first 100 \
  --reference-campaign future-systempilot --out site_future100
python dev_history/verify_future_report.py --report site_future100 \
  --out site_future100/independent_verification.json
python scripts/audit_final_refs.py --report site_future100 \
  --out site_future100/reference_forces.json
python dev_history/audit_candidate_structures.py --report site_future100 \
  --campaign future100 --reference-campaign future-systempilot --jobs 1 \
  --out site_future100/candidate_geometry_audit.json
python dev_history/compare_ablations.py
python dev_history/publish_future.py
```

The candidate audit checks saved best structures within the reference energy
window. Intermediate trace geometries were not saved, so those checks cannot
certify every recorded hit. NEP89 is a model, not DFT, and no search result
proves a global minimum.

For the algorithm-family unit tests, set `GOPT_BENCH_DIR` to `benches-dev`
and run `python -m unittest discover -s tests`. The supplied Slurm scripts
document the original site-specific execution and require adaptation elsewhere.
Scores are specific to these manually selected systems and cannot be directly
compared with scores from different benchmark sets.
