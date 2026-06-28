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
