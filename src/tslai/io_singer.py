"""SINGER posterior-ARG front end (CLAUDE.md §7.4).

SINGER (Deng et al., 2024; bioRxiv 10.1101/2024.03.16.585351) is a Bayesian MCMC method
that **samples ARGs from the posterior** under the SMC, as tskit tree sequences. Those
thinned posterior samples are the ideal input to the ensemble merge layer
(:func:`tslai.ensemble.merge_posterior_tables`): unlike a point estimate (tsinfer/Relate)
they represent ARG *uncertainty*, so averaging the per-member paintings genuinely
marginalises it (the §9 binding constraint).

SINGER is an external binary (built per platform). The path defaults to the env var
``TSLAI_SINGER`` or a known location; override via ``singer_bin``. This module imports
nothing heavy at load time, so the core package never requires SINGER.

Pipeline (validated): haploid VCF (one column per sample node) -> ``singer -ploidy 1``
(with an auto-retry loop à la SINGER's ``singer_master``) -> raw node/branch/mut text
tables -> tskit tree sequence. **Sample order is preserved** (VCF column i -> sample i),
so per-sample labels/truth transfer by id. Node times are in generations.

Note: SINGER's own ``convert_to_tskit`` omits ``compute_mutation_parents()`` and crashes
on recurrent mutations; the conversion here includes it.
"""
from __future__ import annotations

import glob
import os
import random
import subprocess
import tempfile

import numpy as np

__all__ = ["write_haploid_vcf", "singer_tree_sequences"]

DEFAULT_SINGER = os.environ.get("TSLAI_SINGER", "/Users/kmt/SINGER/SINGER/SINGER/singer")


def write_haploid_vcf(ts, path):
    """Write one haploid VCF column per sample node (drops diploid individuals so SINGER,
    run with ``-ploidy 1``, sees each haplotype as its own sample in node order)."""
    tables = ts.dump_tables()
    tables.nodes.set_columns(flags=tables.nodes.flags, time=tables.nodes.time,
                             population=tables.nodes.population)   # individual -> -1
    tables.individuals.clear()
    ts_hap = tables.tree_sequence()
    with open(path, "w") as f:
        ts_hap.write_vcf(f, ploidy=1,
                         position_transform=lambda x: 1 + np.floor(x).astype(int))


def _read_singer_arg(node_file, branch_file, mut_file=None):
    """Port of SINGER's convert_to_tskit read_ARG, with the missing
    ``compute_mutation_parents()`` so recurrent mutations don't crash the build."""
    import tskit
    node_time = np.loadtxt(node_file)
    edge = np.loadtxt(branch_file)
    edge = edge[edge[:, 2] >= 0, :]
    tables = tskit.TableCollection(sequence_length=float(max(edge[:, 1])))
    prev = -1.0
    for tm in node_time:
        if tm == 0:
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE)
        else:
            tm = max(prev + 1e-4, float(tm))
            tables.nodes.add_row(time=tm)
            prev = tm
    tables.edges.set_columns(left=edge[:, 0], right=edge[:, 1],
                             parent=np.array(edge[:, 2], dtype=np.int32),
                             child=np.array(edge[:, 3], dtype=np.int32))
    if mut_file:
        muts = np.loadtxt(mut_file)
        if muts.ndim == 1:
            muts = muts.reshape(1, -1)
        mp = -1.0
        for i in range(muts.shape[0]):
            if muts[i, 0] != mp and muts[i, 0] < 1e6:
                tables.sites.add_row(position=muts[i, 0], ancestral_state='0')
                mp = muts[i, 0]
            tables.mutations.add_row(site=tables.sites.num_rows - 1, node=int(muts[i, 1]),
                                     derived_state=str(int(muts[i, 3])))
    tables.sort()
    if mut_file and tables.mutations.num_rows:
        tables.build_index()
        tables.compute_mutation_parents()
    return tables.tree_sequence()


def singer_tree_sequences(ts, *, Ne, mutation_rate, recombination_rate, n_samples=20,
                          thin=10, burn_in=5, seed=42, ploidy=1, workdir=None,
                          singer_bin=None, with_mutations=True, max_retries=50):
    """Sample posterior ARGs for ``ts`` (which must carry mutations) via SINGER.

    Returns a list of post-burn-in tskit tree sequences (one per thinned posterior
    sample), with sample order preserved. Mirrors SINGER's ``singer_master`` retry loop:
    on a nonzero exit it re-invokes ``-debug`` with fresh seeds.
    """
    singer_bin = singer_bin or DEFAULT_SINGER
    if not os.path.exists(singer_bin):
        raise FileNotFoundError(
            f"SINGER binary not found at {singer_bin}; set TSLAI_SINGER or pass singer_bin")

    tmp = workdir or tempfile.mkdtemp(prefix="tslai_singer_")
    os.makedirs(tmp, exist_ok=True)
    prefix = os.path.join(tmp, "data")
    write_haploid_vcf(ts, prefix + ".vcf")
    out = os.path.join(tmp, "arg")
    L = int(ts.sequence_length)

    base = [singer_bin, "-Ne", str(Ne), "-m", str(mutation_rate), "-r", str(recombination_rate),
            "-ploidy", str(ploidy), "-input", prefix, "-output", out,
            "-start", "0", "-end", str(L), "-polar", "0.5",
            "-n", str(n_samples), "-thin", str(thin)]

    rng = random.Random(seed)
    seeds = [seed] + [rng.randint(0, 2 ** 30 - 1) for _ in range(max_retries)]
    last = None
    for k, sd in enumerate(seeds):
        cmd = base + (["-debug"] if k > 0 else []) + ["-seed", str(sd)]
        last = subprocess.run(cmd, capture_output=True, text=True)
        if last.returncode == 0:
            break
    else:
        raise RuntimeError(f"SINGER failed after {max_retries} retries:\n{last.stderr[-1000:]}")

    idxs = sorted(int(f.split("_nodes_")[1].split(".txt")[0])
                  for f in glob.glob(out + "_nodes_*.txt"))
    samples = []
    for i in idxs:
        if i < burn_in:
            continue
        mf = f"{out}_muts_{i}.txt" if with_mutations else None
        samples.append(_read_singer_arg(f"{out}_nodes_{i}.txt", f"{out}_branches_{i}.txt", mf))
    return samples
