"""``tspaint benchmark`` — run the external LAI comparators from VCF and score them (CLAUDE.md §9).

Sub-commands ``rfmix`` / ``gnomix`` / ``salai`` / ``recombmix`` each take a query VCF (``--vcf``)
plus either a separate reference VCF (``--ref-vcf``) or a combined VCF split by ``--sample-map``,
run the tool in its own environment, and write the calls as a tspaint ``.npz`` painting
(posteriors where the tool provides them, one-hot 0/1 otherwise). ``export-vcf`` writes the VCF
inputs + a matching truth table from a tree sequence, and ``score`` builds the leaderboard.

This module imports ``click`` and is attached to the main CLI by :mod:`tspaint.cli`; the
``tspaint.benchmark`` library itself never imports click.
"""
from __future__ import annotations

import os

import click


def _echo(msg):
    click.echo(msg, err=True)


# --- shared option groups -------------------------------------------------------------------

def _base_opts(f):
    """Query VCF, optional reference VCF, sample map, chromosome, output (shared by all tools)."""
    f = click.option("-o", "--out", required=True, type=click.Path(),
                     help="output painting .npz (tspaint-painting format).")(f)
    f = click.option("--chromosome", default=None, help="contig label (default: from the VCF).")(f)
    f = click.option("-m", "--sample-map", required=True, type=click.Path(exists=True),
                     help="reference sample->ancestry TSV (<sample>\\t<state>).")(f)
    f = click.option("-r", "--ref-vcf", default=None, type=click.Path(exists=True),
                     help="separate reference VCF; omit for a combined query+reference VCF.")(f)
    f = click.option("-q", "--vcf", "query_vcf", required=True, type=click.Path(exists=True),
                     help="query (or combined) phased diploid VCF.")(f)
    return f


def _map_opts(f):
    """Genetic-map options (RFMix / gnomix / Recomb-Mix; SALAI-Net needs none)."""
    f = click.option("--recomb-rate", type=float, default=1e-8, show_default=True,
                     help="per-base recombination rate for the auto-generated genetic map.")(f)
    f = click.option("-g", "--genetic-map", default=None, type=click.Path(exists=True),
                     help="genetic map (auto-generated from --recomb-rate if omitted).")(f)
    return f


# --- group ----------------------------------------------------------------------------------

@click.group()
def benchmark():
    """Run external LAI tools (RFMix / gnomix / SALAI-Net / Recomb-Mix) from VCF → tspaint .npz.

    Provision the git-only tools first with `tspaint benchmark setup` (see `status` for what is
    installed); then run a tool over VCFs, and `score` the outputs against a truth table.
    """


@benchmark.command()
@click.option("--tools", default=None, help="comma-separated subset (default: all in manifest).")
@click.option("--tools-dir", default=None, type=click.Path(), help="clone root (default: external/).")
@click.option("--manifest", default=None, type=click.Path(exists=True),
              help="manifest INI (default: external/tools.ini).")
@click.option("--force", is_flag=True, help="re-provision even if already present.")
@click.option("--dry-run", is_flag=True, help="print the plan and make no changes.")
def setup(tools, tools_dir, manifest, force, dry_run):
    """Clone + build the git-only comparators (gnomix / SALAI-Net / Recomb-Mix) into external/."""
    from .. import benchmark as bm
    bm.setup(tools=tools.split(",") if tools else None, tools_dir=tools_dir, manifest=manifest,
             force=force, dry_run=dry_run, log=_echo)


@benchmark.command()
def status():
    """Show which comparator tools are installed and where."""
    from .. import benchmark as bm
    for r in bm.tool_status():
        click.echo(f"  {'ok ' if r['available'] else '-- '} {r['tool']:10} {r['path']}")


@benchmark.command()
@_base_opts
@_map_opts
@click.option("-G", "--generations", type=float, default=8, show_default=True,
              help="generations since admixture (RFMix -G).")
def rfmix(query_vcf, ref_vcf, sample_map, chromosome, out, genetic_map, recomb_rate, generations):
    """RFMix v2 (posteriors)."""
    from .. import benchmark as bm
    bm.rfmix(query_vcf, ref_vcf, sample_map=sample_map, genetic_map=genetic_map,
             chromosome=chromosome, recomb_rate=recomb_rate, generations=generations,
             out=out, log=_echo)


@benchmark.command()
@_base_opts
@_map_opts
@click.option("--model", default=None, type=click.Path(exists=True),
              help="pretrained gnomix .pkl (default: train from the reference panel).")
@click.option("--phase/--no-phase", default=False, show_default=True,
              help="gnomix Gnofix phasing-error correction.")
def gnomix(query_vcf, ref_vcf, sample_map, chromosome, out, genetic_map, recomb_rate, model, phase):
    """Gnomix (posteriors; trains from the reference panel by default)."""
    from .. import benchmark as bm
    bm.gnomix(query_vcf, ref_vcf, sample_map=sample_map, genetic_map=genetic_map,
              chromosome=chromosome, recomb_rate=recomb_rate, model=model, phase=phase,
              out=out, log=_echo)


@benchmark.command()
@_base_opts
@click.option("--model", default=None,
              help="checkpoint .pth, or 'main'/'hapmap' for a shipped model (default: main).")
def salai(query_vcf, ref_vcf, sample_map, chromosome, out, model):
    """SALAI-Net (hard calls → 0/1; needs no genetic map)."""
    from .. import benchmark as bm
    bm.salai(query_vcf, ref_vcf, sample_map=sample_map, chromosome=chromosome,
             model=model, out=out, log=_echo)


@benchmark.command()
@_base_opts
@_map_opts
@click.option("-e", "--weight", type=float, default=1.5, show_default=True,
              help="recombination-rate weight in the cost function.")
@click.option("-t", "--threads", type=int, default=1, show_default=True)
def recombmix(query_vcf, ref_vcf, sample_map, chromosome, out, genetic_map, recomb_rate,
              weight, threads):
    """Recomb-Mix (hard calls → 0/1)."""
    from .. import benchmark as bm
    bm.recombmix(query_vcf, ref_vcf, sample_map=sample_map, genetic_map=genetic_map,
                 chromosome=chromosome, recomb_rate=recomb_rate, weight=weight, threads=threads,
                 out=out, log=_echo)


@benchmark.command()
@_base_opts
@click.option("--arg", type=click.Choice(["tsinfer", "singer"]), default="tsinfer",
              show_default=True,
              help="ARG front end: a single tsinfer tree sequence, or a SINGER posterior ensemble.")
@click.option("--smooth/--no-smooth", default=True, show_default=True,
              help="horizontal BP smoother (recommended on inferred ARGs).")
@click.option("--estimate-pi", is_flag=True, help="re-estimate pi (default: hold uniform).")
@click.option("--max-iter", type=int, default=12, show_default=True, help="EM iterations.")
@click.option("-j", "--cores", "n_jobs", type=int, default=None,
              help="worker processes (default: SLURM allocation else 1).")
@click.option("--n-singer", type=int, default=100, show_default=True,
              help="SINGER ensemble size = post-burn-in samples to paint (--arg singer).")
@click.option("--thin", type=int, default=20, show_default=True,
              help="SINGER MCMC thinning interval (--arg singer).")
@click.option("--burn-in", type=int, default=20, show_default=True,
              help="SINGER burn-in samples to discard (--arg singer).")
@click.option("--ne", type=float, default=1e4, show_default=True,
              help="effective population size passed to SINGER (--arg singer).")
@click.option("--mu", type=float, default=1.25e-8, show_default=True,
              help="per-base mutation rate passed to SINGER (--arg singer).")
@click.option("--recomb-rate", type=float, default=1e-8, show_default=True,
              help="per-base recombination rate passed to SINGER (--arg singer).")
@click.option("--singer-seed", type=int, default=42, show_default=True,
              help="SINGER base random seed (--arg singer).")
def tspaint(query_vcf, ref_vcf, sample_map, chromosome, out, arg, smooth, estimate_pi, max_iter,
            n_jobs, n_singer, thin, burn_in, ne, mu, recomb_rate, singer_seed):
    """tspaint itself, VCF-native: infer an ARG (tsinfer or a SINGER ensemble) and paint.

    ``--arg singer`` paints a SINGER posterior ARG ensemble (``--n-singer`` post-burn-in samples)
    and averages the per-position posteriors; it needs the SINGER binary (``TSPAINT_SINGER``).
    """
    from .. import benchmark as bm
    bm.tspaint(query_vcf, ref_vcf, sample_map=sample_map, arg=arg, smooth=smooth,
               estimate_pi=estimate_pi, max_iter=max_iter, n_jobs=n_jobs, n_singer=n_singer,
               thin=thin, burn_in=burn_in, Ne=ne, mutation_rate=mu, recombination_rate=recomb_rate,
               singer_seed=singer_seed, out=out, log=_echo)


@benchmark.command("export-vcf")
@click.argument("trees", type=click.Path(exists=True, dir_okay=False))
@click.option("--labels", "labels_path", required=True, type=click.Path(exists=True),
              help="reference labels JSON {node: state}.")
@click.option("--queries", default=None, help="query ids to export (inline or @file; default: non-refs).")
@click.option("--chromosome", default="1", show_default=True)
@click.option("--mutation-rate", type=float, default=4e-7, show_default=True,
              help="rate for overlaid mutations if TREES has no sites.")
@click.option("--seed", type=int, default=1, show_default=True)
@click.option("-o", "--outdir", required=True, type=click.Path(), help="output directory.")
def export_vcf_cmd(trees, labels_path, queries, chromosome, mutation_rate, seed, outdir):
    """Write query/reference VCFs + sample map + truth.npz from a TREES file (for the score loop)."""
    from .. import benchmark as bm
    from ..cli import read_labels, read_id_list, _load_ts
    paths = bm.export_vcf(_load_ts(trees), read_labels(labels_path), queries=read_id_list(queries),
                          outdir=outdir, mutation_rate=mutation_rate, seed=seed,
                          chromosome=str(chromosome))
    for k, v in paths.items():
        _echo(f"  {k}: {v}")


@benchmark.command()
@click.option("--truth", required=True, type=click.Path(exists=True),
              help="tspaint-truth .npz (from export-vcf or `tspaint simulate --truth`).")
@click.option("--deadband", type=float, default=0.4, show_default=True,
              help="confidence dead-band for the switch-density ratio.")
@click.argument("paintings", nargs=-1, required=True, type=str)
def score(truth, deadband, paintings):
    """Score tool paintings against TRUTH. PAINTINGS are `name=path.npz` or `path.npz`."""
    from .. import benchmark as bm
    named = {}
    for p in paintings:
        if "=" in p and not os.path.exists(p):
            name, path = p.split("=", 1)
        else:
            name, path = os.path.splitext(os.path.basename(p))[0], p
        named[name] = path
    rows = bm.score(truth, named, deadband=deadband)
    click.echo(bm.format_table(rows))


@benchmark.command()
@click.option("--truth", required=True, type=click.Path(exists=True),
              help="tspaint-truth .npz (from export-vcf or `tspaint simulate --truth`).")
@click.option("--name", default="", help="painter label stored in the JSON.")
@click.option("--meta", multiple=True, help="scenario metadata key=value (repeatable).")
@click.option("--deadband", type=float, default=0.4, show_default=True)
@click.option("-o", "--out", required=True, type=click.Path(), help="metrics .json output.")
@click.argument("painting", type=click.Path(exists=True, dir_okay=False))
def metrics(truth, name, meta, deadband, out, painting):
    """Full metrics for ONE painting vs TRUTH → JSON (proportions / fragmentation / size-accuracy)."""
    from .. import benchmark as bm
    md = {}
    for kv in meta:
        k, _eq, v = kv.partition("=")
        md[k] = v
    res = bm.score_full(truth, painting, name=name, meta=md, deadband=deadband)
    bm.write_metrics(out, res)
    _echo(f"metrics[{name}]: bal-acc={res['balanced_accuracy']}, "
          f"prop-err={res['proportion_error']}, sw-ratio={res['switch_ratio']} -> {out}")


@benchmark.command()
@click.option("-o", "--outdir", required=True, type=click.Path(),
              help="dir for summary_scalar.csv + summary_by_size.csv.")
@click.argument("jsons", nargs=-1, required=True, type=click.Path(exists=True))
def aggregate(outdir, jsons):
    """Collect `benchmark metrics` JSONs into tidy CSVs (scalar + accuracy-by-size)."""
    from .. import benchmark as bm
    scalar, size = bm.aggregate(list(jsons), outdir)
    _echo(f"aggregate: {len(jsons)} results -> {scalar}, {size}")
