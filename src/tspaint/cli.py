"""``tspaint`` command-line interface (the GWF / cluster entry points).

Each subcommand is a thin wrapper around the library, reading/writing files so a GWF (or any)
workflow can express a job as ``tspaint <sub> <inputs> -o <out>`` with no glue Python. The
ensemble pipeline decomposes as::

    tspaint trees singer-window ... (× windows)   # Stage 3
    tspaint trees merge-arg     ... (× members)   # Stage 3
    tspaint fit   member_*.trees --labels L -o params.npz        # one pooled fit
    tspaint paint member_i.trees --params params.npz -o m_i.npz  # × members (independent)
    tspaint merge m_*.npz -o merged.npz                          # marginalise the ARG
    tspaint date|qc|introgress|ghost|archaic member_i.trees ...  # analyses

Computed results are ``.npz`` (:mod:`tspaint.serialize`); hand-authored inputs are text: labels
as JSON ``{"<node>": <state>}``, id-lists (``--queries`` / ``--soft-refs`` / ``--samples`` /
``--anchors``) as an inline ``3,4,5`` or ``@file``. ``--cores/-j`` defaults to the SLURM
allocation (``$SLURM_JOB_CPUS_PER_NODE``) else serial.

This module is the only one importing ``click``; the core package never imports it, so
``import tspaint`` works without click installed.
"""
from __future__ import annotations

import json
import os
import re

import click


# --- input helpers --------------------------------------------------------------------------

def read_labels(path):
    """Read a labels JSON file ``{"<node-id>": <state-int>}`` into ``{int: int}``."""
    with open(path) as f:
        return {int(k): int(v) for k, v in json.load(f).items()}


def read_id_list(spec):
    """Parse an id list: ``None`` → ``None``; ``"@file"`` → ids from the file; else inline ``"3,4,5"``."""
    if spec is None:
        return None
    if spec.startswith("@"):
        with open(spec[1:]) as f:
            spec = f.read()
    return [int(tok) for tok in re.split(r"[,\s]+", spec.strip()) if tok]


def _cores(cores):
    from .parallel import resolve_cores
    return resolve_cores(cores)


def _load_ts(path):
    import tskit
    return tskit.load(path)


def _echo(msg):
    click.echo(msg, err=True)


# --- top-level group ------------------------------------------------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="tspaint", message="%(version)s")
def cli():
    """tspaint — calibrated soft local-ancestry painting on tree sequences."""


# --- fit / paint / merge (the GWF spine) ----------------------------------------------------

@cli.command()
@click.argument("trees", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--labels", "labels_path", required=True, type=click.Path(exists=True),
              help="labels JSON {node: state}.")
@click.option("--soft-refs", default=None, help="ids whose credibility is learned (inline or @file).")
@click.option("--estimate-pi", is_flag=True, help="re-estimate pi (default: hold uniform).")
@click.option("--deadband", type=float, default=0.0, help="default segmentation dead-band stored in params.")
@click.option("--max-iter", type=int, default=12, show_default=True)
@click.option("--tol", type=float, default=1e-7, show_default=True)
@click.option("-j", "--cores", type=int, default=None, help="worker processes (default: SLURM / 1).")
@click.option("-o", "--out", required=True, type=click.Path(), help="output params .npz.")
def fit(trees, labels_path, soft_refs, estimate_pi, deadband, max_iter, tol, cores, out):
    """Fit the ancestry model (Q, pi, w) pooled across one or more TREES → params.npz."""
    from .em import fit as _fit
    from .model import make_generator_2state
    from .serialize import save_params

    labels = read_labels(labels_path)
    soft = read_id_list(soft_refs)
    ts_list = [_load_ts(t) for t in trees]
    n_jobs = _cores(cores)
    res = _fit(ts_list if len(ts_list) > 1 else ts_list[0], labels,
               Q0=make_generator_2state(1e-3, 1e-3), max_iter=max_iter, tol=tol,
               soft_refs=set(soft) if soft else None, estimate_pi=estimate_pi, n_jobs=n_jobs)
    save_params(out, Q=res.Q, pi=res.pi, w=res.w, K=res.Q.shape[0], labels=labels,
                deadband=deadband, estimate_pi=estimate_pi, loglik_history=res.loglik_history)
    _echo(f"fit: {len(ts_list)} member(s), {len(labels)} refs, n_jobs={n_jobs} -> {out}")


@cli.command()
@click.argument("trees", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--params", "params_path", default=None, type=click.Path(exists=True),
              help="reuse a fitted model (paint only, no fit) — the GWF per-member unit.")
@click.option("--labels", "labels_path", default=None, type=click.Path(exists=True),
              help="fit+paint from labels JSON (one ts, or an ensemble if several TREES).")
@click.option("--queries", default=None, help="ids to paint (inline or @file; default: non-refs).")
@click.option("--soft-refs", default=None, help="refs whose credibility is learned (with --labels).")
@click.option("--estimate-pi", is_flag=True)
@click.option("--smooth", is_flag=True, help="horizontal BP smoother (recommended on inferred ARGs).")
@click.option("--deadband", type=float, default=0.0, show_default=True)
@click.option("--max-iter", type=int, default=12, show_default=True)
@click.option("-j", "--cores", type=int, default=None, help="worker processes (default: SLURM / 1).")
@click.option("-o", "--out", required=True, type=click.Path(), help="output painting .npz.")
def paint(trees, params_path, labels_path, queries, soft_refs, estimate_pi, smooth, deadband,
          max_iter, cores, out):
    """Paint query haplotypes — either --params (paint only) or --labels (fit+paint / ensemble)."""
    from .serialize import save_painting
    n_jobs = _cores(cores)
    qids = read_id_list(queries)

    if params_path is not None:
        if len(trees) != 1:
            raise click.UsageError("paint --params takes exactly one TREES member.")
        from .serialize import load_params
        from .em import build_emissions
        from .output import posterior_table
        p = load_params(params_path)
        ts = _load_ts(trees[0])
        labels, w, pi, Q = p["labels"], p["w"], p["pi"], p["Q"]
        focal = qids if qids is not None else [int(s) for s in ts.samples() if int(s) not in labels]
        if n_jobs > 1:
            from .parallel import posterior_table_parallel
            table = posterior_table_parallel(ts, Q, pi, w=w, labels=labels, focal=focal, n_jobs=n_jobs)
        else:
            table = posterior_table(ts, Q, pi, build_emissions(ts, labels, w, pi), focal=focal)
        save_painting(out, table, Q=Q, pi=pi, w=w, queries=focal, labels=labels,
                      seqlen=ts.sequence_length, deadband=p["deadband"])
        _echo(f"paint(--params): {len(focal)} queries, n_jobs={n_jobs} -> {out}")
        return

    if labels_path is None:
        raise click.UsageError("paint needs --params or --labels.")
    from . import paint as _paint            # api.paint (lazy: pulls api/matplotlib only when used)
    labels = read_labels(labels_path)
    soft = read_id_list(soft_refs)
    members = [_load_ts(t) for t in trees]
    painting = _paint(members if len(members) > 1 else members[0], labels, queries=qids,
                      soft_refs=set(soft) if soft else None, estimate_pi=estimate_pi,
                      deadband=deadband, smooth=smooth, max_iter=max_iter, n_jobs=n_jobs)
    painting.save(out)
    _echo(f"paint(--labels): {len(members)} member(s), {len(painting.queries)} queries, "
          f"n_jobs={n_jobs} -> {out}")


@cli.command()
@click.argument("paintings", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", required=True, type=click.Path(), help="merged painting .npz (mean + band).")
def merge(paintings, out):
    """Average per-position paintings across an ARG ensemble → mean + uncertainty band."""
    from .ensemble import merge_posterior_tables
    from .serialize import load_painting, save_painting
    tables = [load_painting(p) for p in paintings]
    merged = merge_posterior_tables(tables)
    save_painting(out, merged)
    _echo(f"merge: {len(tables)} members, {len(merged)} samples -> {out}")


# --- analyses -------------------------------------------------------------------------------

@cli.command()
@click.argument("trees", type=click.Path(exists=True, dir_okay=False))
@click.option("--labels", "labels_path", required=True, type=click.Path(exists=True))
@click.option("--n-cells", type=int, default=40, show_default=True)
@click.option("--n-iter", type=int, default=15, show_default=True)
@click.option("--estimate-pi", is_flag=True)
@click.option("-o", "--out", required=True, type=click.Path(), help="rate-through-time .npz.")
def date(trees, labels_path, n_cells, n_iter, estimate_pi, out):
    """Admixture rate through time q_AB(t), q_BA(t) → rtt.npz."""
    from .dating import fit_rate_through_time
    from .serialize import save_rate_through_time
    rtt = fit_rate_through_time(_load_ts(trees), read_labels(labels_path), n_cells=n_cells,
                                n_iter=n_iter, estimate_pi=estimate_pi)
    save_rate_through_time(out, rtt)
    _echo(f"date: {n_cells} cells -> {out}")


@cli.command()
@click.argument("trees", type=click.Path(exists=True, dir_okay=False))
@click.option("--labels", "labels_path", required=True, type=click.Path(exists=True))
@click.option("--anchors", default=None, help="trusted hard-clamped refs (inline or @file).")
@click.option("--deadband", type=float, default=0.3, show_default=True)
@click.option("-o", "--out", required=True, type=click.Path(), help="QC table .npz.")
def qc(trees, labels_path, anchors, deadband, out):
    """Audit a reference panel for admixture / mislabelling → qc.npz."""
    from .introgression import reference_qc
    from .serialize import save_reference_qc
    anc = read_id_list(anchors)
    result = reference_qc(_load_ts(trees), read_labels(labels_path),
                          anchors=set(anc) if anc else None)
    save_reference_qc(out, result, deadband=deadband)
    _echo(f"qc: {len(result.labels)} refs -> {out}")


@cli.command()
@click.argument("trees", type=click.Path(exists=True, dir_okay=False))
@click.option("--labels", "labels_path", required=True, type=click.Path(exists=True))
@click.option("--samples", required=True, help="samples to scan (inline or @file).")
@click.option("--min-score", type=float, default=0.5, show_default=True)
@click.option("-o", "--out", required=True, type=click.Path(), help="foreign-tracts .npz.")
def introgress(trees, labels_path, samples, min_score, out):
    """Anonymous foreign-tract detection (label dissent / fits-nothing) → foreign.npz."""
    from .introgression import foreign_tracts
    from .serialize import save_foreign_tracts
    tracts = foreign_tracts(_load_ts(trees), read_labels(labels_path), read_id_list(samples),
                            min_score=min_score)
    save_foreign_tracts(out, tracts)
    _echo(f"introgress: {len(tracts)} samples -> {out}")


def _run_ghost(trees, labels_path, samples, max_iter, out):
    from .archaic import detect_ghost
    from .serialize import save_ghost
    result = detect_ghost(_load_ts(trees), read_labels(labels_path), read_id_list(samples),
                          max_iter=max_iter)
    save_ghost(out, result)
    _echo(f"ghost: P(ghost) over {len(result.burden)} samples -> {out}")


@cli.command()
@click.argument("trees", type=click.Path(exists=True, dir_okay=False))
@click.option("--labels", "labels_path", required=True, type=click.Path(exists=True),
              help="modern reference ids (a {ref: state} labels JSON; only the ids are used).")
@click.option("--samples", default=None, help="samples to scan (inline or @file; default: non-refs).")
@click.option("--max-iter", type=int, default=50, show_default=True)
@click.option("-o", "--out", required=True, type=click.Path(), help="ghost P(ghost) .npz.")
def ghost(trees, labels_path, samples, max_iter, out):
    """Reference-free ghost / archaic introgression search (depth-emission HMM) → ghost.npz."""
    _run_ghost(trees, labels_path, samples, max_iter, out)


@cli.command(hidden=True)
@click.argument("trees", type=click.Path(exists=True, dir_okay=False))
@click.option("--labels", "labels_path", required=True, type=click.Path(exists=True))
@click.option("--samples", default=None)
@click.option("--max-iter", type=int, default=50)
@click.option("-o", "--out", required=True, type=click.Path())
def archaic(trees, labels_path, samples, max_iter, out):
    """Deprecated alias for `tspaint ghost`."""
    _echo("note: `tspaint archaic` is deprecated; use `tspaint ghost`")
    _run_ghost(trees, labels_path, samples, max_iter, out)


# --- simulate (validation / examples) -------------------------------------------------------

@cli.command()
@click.option("--n-admix", type=int, default=8, show_default=True)
@click.option("--n-ref", type=int, default=8, show_default=True)
@click.option("--length", type=float, default=1e6, show_default=True)
@click.option("--recomb-rate", type=float, default=1e-8, show_default=True)
@click.option("--ploidy", type=int, default=1, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--ne", "Ne", type=float, default=1000.0, show_default=True)
@click.option("--t-admix", "T_admix", type=float, default=30.0, show_default=True)
@click.option("--t-split", "T_split", type=float, default=5000.0, show_default=True)
@click.option("--f-a", "f_A", type=float, default=0.5, show_default=True)
@click.option("--mutate/--no-mutate", default=True, show_default=True)
@click.option("--mu", type=float, default=5e-8, show_default=True, help="mutation rate if --mutate.")
@click.option("-o", "--out", required=True, type=click.Path(), help="output .trees.")
@click.option("--labels-out", default=None, type=click.Path(), help="write reference labels JSON.")
@click.option("--truth", "truth_out", default=None, type=click.Path(), help="write true ancestry .npz.")
def simulate(n_admix, n_ref, length, recomb_rate, ploidy, seed, Ne, T_admix, T_split, f_A,
             mutate, mu, out, labels_out, truth_out):
    """Simulate a 2-source admixture with known truth (the validation workhorse)."""
    import numpy as np
    from .sim import simulate_admixture, local_ancestry_truth, SOURCE_A, SOURCE_B
    from .io_tsinfer import add_mutations

    ts = simulate_admixture(n_admix=n_admix, n_ref=n_ref, sequence_length=length,
                            recombination_rate=recomb_rate, ploidy=ploidy, random_seed=seed,
                            Ne=Ne, T_admix=T_admix, T_split=T_split, f_A=f_A)
    if mutate:
        ts = add_mutations(ts, rate=mu, random_seed=seed)
    ts.dump(out)

    name = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    state_of_pop = {p: (0 if n == SOURCE_A else 1) for p, n in name.items() if n in (SOURCE_A, SOURCE_B)}
    npop = ts.tables.nodes.population
    labels = {int(s): state_of_pop[npop[s]] for s in ts.samples() if npop[s] in state_of_pop}
    _echo(f"simulate: {ts.num_samples} haplotypes, {ts.num_sites} sites, {len(labels)} refs -> {out}")

    if labels_out:
        with open(labels_out, "w") as f:
            json.dump({str(k): v for k, v in labels.items()}, f)
        _echo(f"  labels -> {labels_out}")
    if truth_out:
        tracts, _ = local_ancestry_truth(ts)
        samp, left, right, state = [], [], [], []
        for s, segs in tracts.items():
            for (lo, hi, pid) in segs:
                if pid in state_of_pop:
                    samp.append(s); left.append(lo); right.append(hi); state.append(state_of_pop[pid])
        with open(truth_out, "wb") as f:
            np.savez_compressed(f, _format="tspaint-truth", _version=1,
                                sample=np.array(samp, np.int64), left=np.array(left, float),
                                right=np.array(right, float), state=np.array(state, np.int8))
        _echo(f"  truth -> {truth_out}")


# --- trees: input front ends ----------------------------------------------------------------

@cli.group()
def trees():
    """Obtain a tree sequence from genotypes (tsinfer / relate / singer) or overlay mutations."""


@trees.command("tsinfer")
@click.argument("source", type=click.Path(exists=True))
@click.option("-o", "--out", required=True, type=click.Path(), help="output .trees.")
def trees_tsinfer(source, out):
    """tsinfer point-estimate ARG from a ts / VCF / VCF-Zarr SOURCE."""
    from .io import tsinfer
    src = _load_ts(source) if str(source).endswith(".trees") else source
    ts = tsinfer(src)
    ts.dump(out)
    _echo(f"tsinfer: {ts.num_trees} trees -> {out}")


@trees.command("relate")
@click.argument("anc", type=click.Path(exists=True))
@click.argument("mut", type=click.Path(exists=True))
@click.option("-o", "--out", required=True, type=click.Path(), help="output .trees.")
def trees_relate(anc, mut, out):
    """Convert Relate .anc/.mut to a tree sequence (--compress)."""
    from .io import relate
    relate(anc, mut).dump(out)
    _echo(f"relate -> {out}")


@trees.command("singer")
@click.argument("source", type=click.Path(exists=True))
@click.option("--ne", "Ne", type=float, required=True)
@click.option("--mut-rate", "mutation_rate", type=float, required=True)
@click.option("--recomb-rate", "recombination_rate", type=float, required=True)
@click.option("--n", "n_samples", type=int, default=20, show_default=True)
@click.option("--thin", type=int, default=10, show_default=True)
@click.option("--burn-in", type=int, default=5, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--length", "sequence_length", type=float, default=None)
@click.option("-d", "--out-dir", required=True, type=click.Path(), help="dir for member_*.trees.")
def trees_singer(source, Ne, mutation_rate, recombination_rate, n_samples, thin, burn_in, seed,
                 sequence_length, out_dir):
    """Sample a posterior ARG ensemble with SINGER (whole genome) → member_*.trees."""
    from .io import singer
    src = _load_ts(source) if str(source).endswith(".trees") else source
    ensemble = singer(src, Ne=Ne, mutation_rate=mutation_rate, recombination_rate=recombination_rate,
                      n_samples=n_samples, thin=thin, burn_in=burn_in, seed=seed,
                      sequence_length=sequence_length)
    os.makedirs(out_dir, exist_ok=True)
    for i, ts in enumerate(ensemble):
        ts.dump(os.path.join(out_dir, f"member_{i:03d}.trees"))
    _echo(f"singer: {len(ensemble)} posterior members -> {out_dir}/member_*.trees")


@trees.command("add-mutations")
@click.argument("trees_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--rate", type=float, default=1e-8, show_default=True)
@click.option("--seed", type=int, default=None)
@click.option("-o", "--out", required=True, type=click.Path(), help="output .trees.")
def trees_add_mutations(trees_path, rate, seed, out):
    """Overlay biallelic mutations on a bare ARG."""
    from .io import add_mutations
    add_mutations(_load_ts(trees_path), rate=rate, random_seed=seed).dump(out)
    _echo(f"add-mutations rate={rate} -> {out}")


@trees.command("singer-window")
@click.argument("source", type=click.Path(exists=True))
@click.option("--start", type=float, required=True)
@click.option("--end", type=float, required=True)
@click.option("--out-prefix", required=True, help="SINGER writes {out_prefix}_{nodes,branches,muts}_<i>.txt.")
@click.option("--ne", "Ne", type=float, required=True)
@click.option("--mut-rate", "mutation_rate", type=float, required=True)
@click.option("--recomb-rate", "recombination_rate", type=float, required=True)
@click.option("--n", "n_samples", type=int, default=20, show_default=True)
@click.option("--thin", type=int, default=10, show_default=True)
@click.option("--ploidy", type=int, default=1, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def trees_singer_window(source, start, end, out_prefix, Ne, mutation_rate, recombination_rate,
                        n_samples, thin, ploidy, seed):
    """Run SINGER on ONE genomic window (the GWF per-window unit; bare-singer engine)."""
    from .io_singer import singer_window
    src = _load_ts(source) if str(source).endswith(".trees") else source
    idxs = singer_window(src, start=start, end=end, out_prefix=out_prefix, Ne=Ne,
                         mutation_rate=mutation_rate, recombination_rate=recombination_rate,
                         n_samples=n_samples, thin=thin, ploidy=ploidy, seed=seed)
    _echo(f"singer-window [{start:g},{end:g}): {len(idxs)} samples -> {out_prefix}_*_<i>.txt")


@trees.command("merge-arg")
@click.option("--manifest", required=True, type=click.Path(exists=True),
              help="TSV rows: window_index start end out_prefix.")
@click.option("--member", type=int, required=True, help="MCMC sample index to stitch.")
@click.option("--skip-gaps", default=None, help="regions to skip, e.g. '5e6-8e6,12e6-13e6'.")
@click.option("--coords", type=click.Choice(["local", "absolute"]), default="local", show_default=True)
@click.option("--merge-arg-script", default=None, type=click.Path(), help="path to merge_ARG.py.")
@click.option("--python", "python_bin", default=None, help="interpreter for merge_ARG.py (needs tszip).")
@click.option("--dry-run", is_flag=True, help="print the file_table; do not run merge_ARG.py.")
@click.option("-o", "--out", default=None, type=click.Path(), help="merged member .trees.")
def trees_merge_arg(manifest, member, skip_gaps, coords, merge_arg_script, python_bin, dry_run, out):
    """Stitch per-window SINGER tables into one chromosome-length ARG (wraps merge_ARG.py)."""
    from .io_singer import build_merge_table, run_merge_arg
    windows = _read_manifest(manifest)
    rows = build_merge_table(windows, member, skip_gaps=_parse_gaps(skip_gaps), coords=coords)
    if dry_run:
        for (n, b, m, blk) in rows:
            click.echo(f"{n} {b} {m} {int(blk)}")
        return
    if not out:
        raise click.UsageError("merge-arg needs -o/--out (or --dry-run).")
    run_merge_arg(rows, out, script=merge_arg_script, python=python_bin)
    _echo(f"merge-arg member {member}: {len(rows)} windows -> {out}")


def _read_manifest(path):
    """Read a windows manifest TSV: rows ``window_index start end out_prefix`` (# comments ok)."""
    windows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            windows.append((int(parts[0]), float(parts[1]), float(parts[2]), parts[3]))
    return windows


def _parse_gaps(spec):
    """Parse ``--skip-gaps`` ``"lo-hi,lo-hi"`` into ``[(lo, hi), ...]``."""
    if not spec:
        return []
    gaps = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            lo, hi = part.split("-")
            gaps.append((float(lo), float(hi)))
    return gaps


def main():
    cli()


if __name__ == "__main__":
    main()
