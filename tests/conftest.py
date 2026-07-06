"""Pytest configuration — keep the suite serial and deterministic.

tspaint's library default is ``n_jobs=None`` → **all CPUs** (:func:`tspaint.parallel.resolve_cores`),
which would spawn a process pool for every default ``paint`` / ``fit`` / QC call — slow, and not
byte-identical (the parallel E-step reduces in a different order). Force serial for the test process
here; tests that exercise parallelism pass an explicit ``n_jobs`` (which wins over this). The
``resolve_cores`` unit test manages its own environment via ``monkeypatch``.
"""
import os

for _v in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
    os.environ.pop(_v, None)
os.environ["TSPAINT_CORES"] = "1"     # resolve_cores(None) -> 1 (serial) unless a test overrides
