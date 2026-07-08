"""Sample-ID stamping and label/query resolution (CLAUDE.md §5; :mod:`tspaint.ids`).

The inference front ends stamp the source's sample ids onto the (otherwise anonymous) output tree
sequence, so ``paint`` / ``fit`` / the introgression tools accept ``labels`` / ``queries`` keyed by
sample-ID string as well as by integer node index.
"""
import os

import numpy as np
import pytest

import tspaint
from tspaint import io
from tspaint.ids import attach_sample_ids, resolve_labels, resolve_ids, sample_id_index
from tspaint.io_singer import DEFAULT_SINGER
from tspaint.sim import simulate_admixture, admixture_demography


def _bare(ts):
    """Strip individuals + node metadata to emulate SINGER's anonymous (null-schema) output."""
    t = ts.dump_tables()
    t.individuals.clear()
    t.nodes.set_columns(flags=t.nodes.flags, time=t.nodes.time, population=t.nodes.population)
    return t.tree_sequence()


def _diploid_vcf(ts, path, names):
    """Write ``ts`` as a real diploid VCF (``0|1`` genotypes) with the given individual names."""
    with open(path, "w") as f:
        ts.write_vcf(f, individual_names=names,
                     position_transform=lambda x: 1 + np.floor(x).astype(int))


def _sim(seed=1, n_admix=2, n_ref=2):
    return simulate_admixture(admixture_demography(Ne=1000, T_admix=30, T_split=5000, f_A=0.5),
                              n_query=n_admix, n_reference=n_ref, sequence_length=6e4,
                              recombination_rate=1e-8, random_seed=seed).ts


# --- attach_sample_ids -----------------------------------------------------------------------

def test_stamp_diploid_singer_like():
    """A null-schema, individual-less ts is stamped with individuals + 1-based per-haplotype ids."""
    ts = _bare(_sim())
    n = ts.num_samples
    names = [f"S{i // 2}_{i % 2}" for i in range(n)]     # flattened 0-based, as the readers produce
    st = attach_sample_ids(ts, names, ploidy=2)

    assert st.num_individuals == n // 2
    assert st.individual(0).metadata == {"id": "S0"}
    assert list(st.individual(0).nodes) == [0, 1]
    assert st.node(0).metadata == {"id": "S0_1"}         # 1-based per-haplotype
    assert st.node(1).metadata == {"id": "S0_2"}


def test_stamp_preserves_other_node_metadata_tsinfer_like():
    """Stamping merges the id into sample nodes without dropping other nodes' metadata."""
    ts = io.tsinfer(io.add_mutations(_sim(seed=2), rate=4e-7, random_seed=2))
    internal = next((n for n in range(ts.num_nodes)
                     if not ts.node(n).is_sample() and ts.node(n).metadata), None)
    before = ts.node(internal).metadata if internal is not None else None

    names = [f"S{i // 2}_{i % 2}" for i in range(ts.num_samples)]
    st = attach_sample_ids(ts, names, ploidy=2)
    if internal is not None:
        assert st.node(internal).metadata == before      # tsinfer's ancestor_data_id etc. survive
    assert st.individual(0).metadata == {"id": "S0"}


def test_stamp_haploid_has_no_suffix():
    ts = _bare(_sim(n_admix=1, n_ref=1))
    names = [f"hap{i}" for i in range(ts.num_samples)]
    st = attach_sample_ids(ts, names, ploidy=1)
    assert st.node(0).metadata == {"id": "hap0"}          # base id, no _1
    assert st.individual(0).metadata == {"id": "hap0"}
    assert list(st.individual(0).nodes) == [0]


def test_stamp_guards_return_unchanged():
    ts = _bare(_sim())
    assert attach_sample_ids(ts, None, 1) is ts
    assert attach_sample_ids(ts, ["only_one"], 1) is ts   # length mismatch -> no-op


# --- resolution ------------------------------------------------------------------------------

def _stamped():
    ts = _bare(_sim())
    names = [f"S{i // 2}_{i % 2}" for i in range(ts.num_samples)]
    return attach_sample_ids(ts, names, ploidy=2)


def test_resolve_labels_string_and_int_keys():
    st = _stamped()
    assert resolve_labels(st, {"S0": 0}) == {0: 0, 1: 0}     # base id -> both haplotypes
    assert resolve_labels(st, {"S0_2": 1}) == {1: 1}         # per-haplotype -> one node
    assert resolve_labels(st, {3: 1}) == {3: 1}              # int index passes through
    assert resolve_labels(st, {"S0": 0, "S1": 1}) == {0: 0, 1: 0, 2: 1, 3: 1}


def test_resolve_ids_expands_dedupes_and_orders():
    st = _stamped()
    assert resolve_ids(st, ["S1", 5]) == [2, 3, 5]
    assert resolve_ids(st, None) is None


def test_resolve_digit_string_falls_back_to_index():
    st = _stamped()
    # a JSON labels file keyed {"3": 0}: no id "3" -> treated as node index 3
    assert resolve_labels(st, {"3": 0}) == {3: 0}


def test_resolve_unknown_string_raises():
    st = _stamped()
    with pytest.raises(KeyError):
        resolve_labels(st, {"not_a_sample": 0})


def test_sample_id_index_empty_when_unstamped():
    ts = _bare(_sim())
    assert sample_id_index(ts) == {}
    # int keys still resolve on an unstamped ts (backward compatible)
    assert resolve_labels(ts, {0: 0, 1: 1}) == {0: 0, 1: 1}


# --- Variants.ploidy -------------------------------------------------------------------------

def test_variants_ploidy_from_diploid_vcf(tmp_path):
    from tspaint.io_genotypes import variants_from_vcf
    ts = io.add_mutations(_sim(), rate=6e-7, random_seed=1)
    vcf = str(tmp_path / "dip.vcf")
    _diploid_vcf(ts, vcf, [f"samp{i}" for i in range(ts.num_individuals)])
    v = variants_from_vcf(vcf)
    assert v.ploidy == 2
    assert v.num_haplotypes == ts.num_samples
    assert v.sample_names[:2] == ["samp0_0", "samp0_1"]


# --- end-to-end via a front end (no external binary) -----------------------------------------

def test_tsinfer_diploid_vcf_paints_by_name(tmp_path):
    """io.tsinfer(diploid VCF) stamps ids; paint accepts sample-ID-string labels/queries."""
    ts = io.add_mutations(_sim(seed=5, n_admix=3, n_ref=3), rate=6e-7, random_seed=5)
    npop = ts.tables.nodes.population
    pname = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    names = [f"{pname[npop[ts.individual(i).nodes[0]]]}_{i}" for i in range(ts.num_individuals)]
    vcf = str(tmp_path / "cohort.vcf")
    _diploid_vcf(ts, vcf, names)

    inferred = io.tsinfer(vcf)
    assert inferred.num_individuals == ts.num_individuals
    assert inferred.individual(0).metadata["id"] == names[0]

    labels = {n: 0 for n in names if n.startswith("A_")}
    labels.update({n: 1 for n in names if n.startswith("B_")})
    queries = [n for n in names if n.startswith("ADMIX")]
    painting = tspaint.paint(inferred, labels, queries=queries)

    # each diploid reference contributed both haplotype nodes; queries expanded likewise
    assert len(painting.labels) == 2 * len(labels)
    assert set(painting.labels.values()) == {0, 1}
    assert len(painting.queries) == 2 * len(queries)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(DEFAULT_SINGER), reason="SINGER binary not available")
def test_singer_diploid_vcf_ploidy_and_stamping(tmp_path):
    """io.singer on a diploid VCF: -ploidy stays 1 (no sample doubling) and ids are stamped."""
    ts = io.add_mutations(_sim(seed=11, n_admix=3, n_ref=3), rate=6e-7, random_seed=11)
    names = [f"samp{i}" for i in range(ts.num_individuals)]
    vcf = str(tmp_path / "cohort.vcf")
    _diploid_vcf(ts, vcf, names)

    ens = io.singer(vcf, _Ne=1000, _m=6e-7, _r=1e-8,
                    ts=4, mcmc_step=2, mcmc_burnin=4, _seed=42, sequence_length=ts.sequence_length)
    g = ens[0] if isinstance(ens, list) else ens
    assert g.num_samples == ts.num_samples          # NOT doubled (the -ploidy clobber regression)
    assert g.num_individuals == ts.num_individuals
    assert g.individual(0).metadata == {"id": "samp0"}

    painting = tspaint.paint(ens, {"samp0": 0, "samp1": 1}, queries=["samp2"])
    assert painting.labels == {0: 0, 1: 0, 2: 1, 3: 1}
    assert painting.queries == [4, 5]
