# PLAN — the road to publication

> **Single merged plan.** Supersedes the former `PLAN_PUBLICATION.md` + `PLAN_DEFERRED_PUBLICATION.md`.
> Evidence base: the **ten** comparator reads in `papers/*.md` — RFMix, HAPMIX, **MOSAIC**, Recomb-Mix,
> SALAI-Net, ARGweaver-D, GhostBuster, hapla, ARGMix, ARGformer.
>
> Part I (§0–§5) is the **near-term**: what is true, what must be measured, what to stop claiming.
> Part II (§6–§10) is the **deferred/ambitious**: what to build, and the poster-child question that
> turns a method into a *Nature Genetics* paper.
>
> **Status: the original framing is dead, a better one is available, and three cheap measurements
> decide which paper we write. Do not choose the framing before they land.**

---

# PART I — THE NEAR TERM

## 0. The situation, without varnish

### 0.1 Every clause of our pitch is claimed

**What we thought `tspaint` was:** the first generative, calibrated, genealogy-native local ancestry
method, with EM-learned parameters and reference-free ghost detection, reaching older admixture than
window/copying methods.

| Claim | Claimed by | When |
|---|---|---|
| Generative + calibrated LAI | **HAPMIX** — calibrated, self-estimates its own `r²` | 2009 |
| **Panels need not match the admixing sources** | **MOSAIC** (Salter-Townshend & Myers) | **2019** |
| Reference-panel QC / label noise | **MOSAIC** — an admixed panel shows as a multi-entry copying-matrix row | **2019** |
| Accuracy without ground truth; *why* a fit is weak | **MOSAIC** (`E[r²]`, `Rst`) | **2019** |
| ARG-native generative ghost detection, calibrated | **ARGweaver-D** (Siepel) | 2020 |
| Tree-native LAI | **AncestralPaths**, **ARGMix** (Ioannidis) | 2023 / 2026 |
| Genealogy-native + EM + local ancestry + ghost + dating | **GhostBuster** (Myers & Speidel) | **Apr 2026** |
| Reaching older admixture | **HAPMIX** (λ=400), **ARGMix** | 2009 / 2026 |
| No genetic map / non-model organisms | **SALAI-Net**, **Loter** | 2018 / 2022 |
| Haplotype-sharing beats SNPs | **hapla** (Meisner) — peer-reviewed ablation | 2026 |

### 0.2 🔴 The lineage — the single most important thing in this document

Reading MOSAIC exposes a **17-year research programme by one lab**, which `tspaint` has been walking
into the middle of:

> **HAPMIX** (2009 — 2-way; the panels *are* the sources)
> → **GLOBETROTTER** (2014 — learn the panel↔source relationship, but with an *ancestry-unaware* HMM)
> → **MOSAIC** (2019 — K-way; learn the panel↔source relationship **inside** the HMM, so panels need
>   not match the sources at all; ghosts fall out for free)
> → **GhostBuster** (2026 — *the same idea moved onto the genealogy*: coalescence rates to reference
>   populations replace copying probabilities onto donor panels).

**Simon Myers is a co-author of all four.** GhostBuster is not a surprise attack; it is the fourth
instalment of a programme with obvious momentum. **Assume a K-way, uncertainty-aware GhostBuster v2
within a year.** Any framing that competes head-on with this lineage on *its own axis* will lose.

**And against the LAI incumbents we do not win on accuracy.** Measured (CLAUDE.md §9): we *tie* RFMix
at strong structure (0.99/0.99) and at weak structure (0.78–0.85 / 0.77–0.80). Recomb-Mix beats us
intra-continentally — the regime the field is moving toward. **MOSAIC beats RFMix, ELAI and LAMP-LD
even when they are given the panel↔ancestry correspondence and MOSAIC is not.** We have never run
MOSAIC's, GhostBuster's or ARGMix's benchmark.

---

## 1. What actually survives

Revised after MOSAIC. Three solid, two contingent on a measurement, one correction.

### Solid

**1. Nobody propagates genealogical uncertainty. Nobody.** GhostBuster reads a 300 kya–1 Mya human
history off a **single Relate point estimate**, whose branch lengths are fitted under a **panmictic
prior** — misspecified precisely in the structured regime it studies — and *says so*:

> *"Admixture dates exceeding 2,500 generations … may reflect **biases in genealogical inference**
> rather than true admixture."*

ARGMix's headline f4 statistic **flips sign** (`Z = −5.6 → +6.14`) under a *hard* ancestry mask, with
no error bar. ARGweaver-D and GhostBuster **publicly disagree** on whether super-archaic introgression
into Denisovans exists at all. MOSAIC has no genealogy, so it cannot even pose the question.
**We have SINGER ensembles and `posterior_std`, built and measured (§7.4). This is the biggest
unclaimed opening in the field, and MOSAIC *strengthens* it.**

**2. No training, on any demography.** ARGMix, ARGformer, SALAI-Net, gnomix and AncestralPaths are all
supervised classifiers trained on simulated demographies. ARGMix names this as *the* fundamental
problem: *"the lack of ground truth and the reliance on simulated data … simulations that may never
fully capture covariate shifts and unknown biases."* An EM-fitted generative model does not have it.

**3. Reference QC — but at a finer granularity than MOSAIC's, and with feedback.** ⚠️ **Softened.**

| | MOSAIC (2019) | `tspaint` |
|---|---|---|
| Granularity | **per panel** (copying-matrix row) | **per individual** (`w_i`) *and* **per locus** (LOO map) |
| Output | "this panel is admixed" | "*this haplotype* is admixed, and *here are the tracts*" |
| Feedback | none — a fitted nuisance | `soft_refs` / `mask` → **re-paint** |

**Retract the claim that "nobody else has a label-noise model."** MOSAIC has had one since 2019.
What is ours is the *resolution* and the *feedback loop*. The cited, honest version is far more
credible than the overclaim.

### Contingent on Phase 0

**4. The whole tree, and every edge.** MOSAIC makes the ladder a **law of the field**: *six published
methods, six different things thrown away, all for cost.*

| Method | What it discards |
|---|---|
| RFMix / SALAI-Net / Recomb-Mix / LAMP-LD | the genome → **fixed windows** |
| **MOSAIC** | the donors → **top 100 per gridpoint**, ranked by an *ancestry-unaware* pass |
| AncestralPaths | the topology → **GNN proportions** |
| ARGMix | the tree → **29-node subgraph**; the genome → **every 10th tree** |
| GhostBuster | the tree → **coalescence times only**; the genome → **1 tree per 10 kb** |
| **`tspaint`** | **nothing — every edge, span-weighted, counted once** |

This is CLAUDE.md §3.3's lead-novelty claim, and it **survives contact with all ten papers.** But it
is an *assertion*. **The ladder (§2.3) makes it a number. If thinning costs little, we have no methods
engine.** ⚠️ *Honesty: we discard nothing **given the ARG** — but the ARG is itself a lossy inference.
Say so before a reviewer does.*

**5. Ghost detection — the differentiator is the *time axis*, not the ghost.** ⚠️ **Reframed.**

|  | Ghost defined as | Can it say "archaic"? |
|---|---|---|
| **MOSAIC** | a component whose **copying profile** matches no panel | ❌ **no time axis at all** |
| **GhostBuster** | a component whose **coalescence-rate profile** matches no panel | ✅ (rates over 20 epochs) |
| **`tspaint`** | a tract whose **nearest coalescence with any panel member is unusually deep** | ✅ absolute depth, with an archaic floor |

Against **MOSAIC**, the time axis is a clean structural win — a copying model literally cannot express
"this diverged 1 Mya." Against **GhostBuster** it is *not* a differentiator; their rate profiles carry
time. **Our edge over GhostBuster on ghosts is uncertainty, topology and per-locus resolution — not
the ghost mechanism itself.** Be precise about this or be corrected in review.

---

## 2. Phase 0 — the three measurements that decide the paper (~2–3 weeks)

**No framing decision before these land.** All three are cheap.

### 2.1 🔴 Read GhostBuster's Supplementary Note — *and* MOSAIC's model section properly
The exact GhostBuster likelihood is in its Supplement and nobody here has seen it. And the
HAPMIX→GLOBETROTTER→MOSAIC→GhostBuster lineage (§0.2) is *the* intellectual history of this problem;
getting it wrong in print would be embarrassing and would cost us Myers as a friendly reviewer.
**Highest-value reading left in the project.**

### 2.2 🔴 Head-to-head vs. GhostBuster on their own simulations
Public msprime designs (ghost population A at 20%, reference B, 72+ settings). Score on **their** axis:
`R²` vs. true local ancestry. **They report `R² = 0.59` on the headline ghost scenario.**
*Hypothesis:* full-topology pruning over every edge beats a coalescence-time summary over thinned trees.
**This single number decides whether a methods paper exists.**

### 2.3 🔴 The topology / thinning ladder
Run `tspaint` at each rung of §1.4 — full, then restricted to a 29-node subgraph, then thinned to one
tree per 10 kb. **The direct, isolated measurement of what §3.3 buys, in the exact settings where five
competitors chose otherwise.**

**Decision gate:** favourable → Framings A *and* B (§3). Flat → Framing B only, and §3.3 becomes
machinery rather than headline.

---

## 3. The framing decision

### Framing A — Methods: *"The whole genealogy, and every edge"*
Engine: §2.2 + §2.3. **Risk:** if thinning is cheap, there is no paper. **Ceiling:** solid, not headline.

### Framing B — Science: *"How much of deep human admixture history survives genealogical uncertainty?"*
The field is producing extraordinary deep-time claims — three back-to-Africa waves, a 300 kya
structured split, two PRDM9 lineages forming both humans *and* Neanderthals, contradictory verdicts on
super-archaic Denisovan admixture — **all read off single point-estimate ARGs under a panmictic prior,
with no uncertainty propagated.** We move the ancestry model onto the *posterior* over genealogies and
ask which conclusions survive.
- Not scooped; GhostBuster's own limitations invite it. Adjudicates a live public disagreement.
- Makes Myers and Speidel **allies rather than rivals**.
- **Risk:** it is fundamentally a **critique**. Critique papers cap out lower and make enemies.

### Framing C — Tool: *one fitted object, four deliverables*
High floor, low ceiling. **Best as the companion software note, not the flagship.**

### Framing D — ⭐ **Recommended: the poster-child paper (§9)**
*Validated* ghost inference, with **baboons as the ground truth the field has never had** — then the
human application it licenses. **Makes Framing B's points constructively**: uncertainty arrives as
*validation*, not accusation. **See Part II.**

---

## 4. Phase 1 — close the gaps that block *any* framing

### 4.1 Metrics we lack and cannot be compared without
- **`validate.ancestry_r2`** — LAMP-LD/K-way `r²`. **Required by HAPMIX, MOSAIC, Recomb-Mix,
  GhostBuster, FLARE — i.e. by everyone.** We do not compute it.
- **Plain accuracy** (ARGMix, SALAI-Net, gnomix) alongside balanced accuracy.
- **`E[r²]`** — accuracy with **no ground truth**. **MOSAIC's** (GhostBuster inherited it). Free for a
  calibrated method. **Steal it.**
- **`Rst`** — **MOSAIC's panel-adequacy statistic**; separates "bad panels" from "old admixture."
  Better than anything we have. **Steal it.**
- The **MAF ≤ 0.005 / MAC ≤ 50** marker filter (FLARE's; adopted by Recomb-Mix).

### 4.2 Benchmarks not yet run
- **SALAI-Net live** — pre-trained, wired, costs *seconds*. No excuse.
- **gnomix live** — wired.
- **🔴 MOSAIC** — the strongest *genotype-based* comparator; beats RFMix/ELAI/LAMP-LD *even when they
  are given information it is denied*; best-in-class on small panels. **No runner exists. Write one.**
- **FLARE, Loter** — no runners. **These, not Recomb-Mix, are the deep-admixture opponents.**
- **K-way.** `K=3` implemented and tested; **never benchmarked.** Run ARGMix's four-way West Eurasian
  design (Anatolian/WHG/EHG/CHG, ancient references) at `K=4`.

### 4.3 Known defects
- **`detect_ghost`'s emission anchoring breaks for archaic queries.** The modern floor is anchored on
  *reference-internal* depths, so a Denisovan query sits above it everywhere and the whole genome is
  flagged. Scope the claim to modern queries, or generalise to a **per-query baseline**.
- **No phase-error model.** RFMix corrects strand flips (≤1/window); **MOSAIC corrects an unbounded
  number**; HAPMIX integrates over phase. We have nothing.
- **Tract-length dating is contested.** ARGweaver-D *abandoned* segment-length dating — *"strong
  ascertainment bias towards finding longer regions"* — and used the frequency spectrum instead. Our
  switch-density result (0.99 ± 0.01) claims the opposite. **Sweep the deadband and measure
  length-distribution bias against truth across migration ages.** If our claim survives Siepel's
  critique it is a *stronger* result than we have been treating it as.

---

## 5. Phase 2 — the experiments, by value-per-day

1. **🔴 Uncertainty propagated into ancestry-specific downstream analyses.** ARGMix's masked f4 flips
   sign; their selection result rests on an allele called at *exactly* 0%; Ötzi's reassignment from
   Sardinians to Bergamo Italians rests entirely on which segments are masked. **All are hard-masking
   operations on uncalibrated calls with no error bars.** Propagate our posterior (+ a SINGER band) into
   masked-f4 / masked-PCA / CLUES2 and report what survives. *Converts a competitor's best result into
   our best argument, and needs no new modelling.*
2. **🎯 Withheld-reference ghost experiments — the same figure, three times.** Denisovan withheld
   (Wohns et al. 2022 ARG — ARGformer *cannot* function without the archaic reference); wolf withheld
   (canid data, SALAI-Net's own flagship, public); Recomb-Mix's published Yamnaya-as-nearest-panel
   artefact (HGDP chr18).
3. **The two-simulator design.** HAPMIX's λ=400 claim rests on a **mosaic-of-extant-haplotypes**
   simulator in which the admixed population has *no demography of its own*, so the query↔reference
   link cannot decay. Our collapse at `T_admix = 1000` under msprime is the same phenomenon seen
   honestly. **Run both.** Recomb-Mix's SLiM forward sims independently corroborate: with a real
   admixed deme, accuracy drops with age.
4. **Admixture-age curve** vs. RFMix / MOSAIC / Recomb-Mix / SALAI-Net / FLARE, `T ∈ {30…1000}`,
   ≥6 seeds, both simulators. Give RFMix its best window size. Design the demography so the
   query↔reference link survives to `T ≈ 700`, or everything goes to chance.
5. **The intra-continental stress test — our biggest exposure.** Recomb-Mix gets `r²` 0.93–0.98 on
   TSI/FIN/GBR; our weak-structure number is 0.77–0.80. **Does SINGER rescue it?** Find out before a
   reviewer does.
6. **`w_i` on ARGMix's admixed ancient references** (Yamnaya = CHG+EHG; Neolithic farmer =
   WHG+Anatolian). If learned credibility recovers the known mixture fractions, it validates itself.
7. **Runtime, honestly.** Report "given the ARG" *and* "end-to-end from VCF." We lose to SALAI-Net
   (0.08 s), Recomb-Mix (`O(n·p)`) and hapla (128k samples in 5 min). **Own it.** Answer with the
   deliverable and with amortisation (one ARG serves painting + dating + QC + ghost).

---

# PART II — THE DEFERRED / AMBITIOUS PLAN

## 6. What a high-impact paper actually needs

A *Bioinformatics* paper needs a better method. A ***Nature Genetics*** paper needs a **finding**,
delivered by a method that could not have delivered it otherwise. Three tests:

1. **Generality** — a non-specialist cares. *"Did an unsampled hominin contribute to living Africans?"*
   passes. *"Our switch-density ratio is 0.99"* does not.
2. **Only-us** — unreachable by the incumbents, not merely harder. If GhostBuster could have found it,
   the method is decoration.
3. **Falsifiability** — the claim must be checkable. **This is where the entire ghost literature is
   weakest, and where we have an advantage nobody has exploited.**

> **The strategic problem: archaic ghost work is unfalsifiable by construction.** ARGweaver-D says ~1%
> of the Denisovan genome is super-archaic. GhostBuster says there is *no support* for it. Two of the
> strongest labs, opposite conclusions, on the field's highest-profile ghost claim — **and there is no
> way to check, because the ghost is gone.** The field has no ground truth and therefore no way to
> calibrate its own methods.
>
> **We can fix that (§9.1).**

---

## 7. Deepening the edges — by impact per unit of work

### 7.1 🔴 Uncertainty propagation into downstream analyses — *the feature that changes what `tspaint` is*
Today we emit a calibrated posterior and every downstream user immediately **throws it away** with an
argmax. Masked PCA, masked f-statistics, ancestry-stratified selection scans, Tractor-style GWAS — all
are **hard-masking operations on uncalibrated calls, reported without error bars** (see §5.1 for the
evidence that this bites).

**Build:** `Painting.weights()` → per-site posterior weights, not a 0/1 mask; posterior-weighted
**f2/f3/f4/D** with an ARG-ensemble bootstrap as the error bar; posterior-weighted
**ancestry-specific PCA**; posterior-weighted **allele-frequency trajectories** → CLUES2. All of it
consuming a SINGER ensemble, so the band includes genealogical uncertainty.

**Unlocks:** `tspaint` stops being "another painter" and becomes **the substrate that makes
ancestry-specific inference honest** — a much larger, unoccupied claim, which makes the ancient-DNA
selection-scan literature our *user* rather than our competitor.

### 7.2 🔴 The per-locus power / detectability track
`MISSING_INFO` currently tags isolated spans. **Generalise it into a genome-wide map of how much
ancestry information the local genealogy carries at all**, independent of the data: at each locus,
take the inferred local tree and the fitted `Q`, simulate ancestry down it, and ask how well the
down-pass could recover it. That is a per-locus **expected maximum achievable posterior**.

⚠️ **MOSAIC is the precedent — cite it and differentiate precisely.** `E[r²]` and `Rst` already answer
*"how good can this analysis be, and why not better?"* — but they are **global, one number per run.**
**The power track is per-locus.** *"Are introgression deserts real, or power deserts?"* needs a
genome-wide map; `E[r²]` cannot answer it. **That distinction is the whole contribution. Do not pretend
the idea is unprecedented.**

A discriminative method still cannot do this at all — RFMix, Recomb-Mix, SALAI-Net and ARGMix have no
generative ancestry model on the tree, so they cannot ask *"what is the best I could possibly do here?"*

### 7.3 Ghost detection — fix, generalise, make it falsifiable
- **Fix the archaic-query bug** (§4.3): per-query baseline depth.
- **Multiple ghosts.** GhostBuster and MOSAIC both fit `K` free components; we fit 2 states. Generalise.
- **Ship the validation protocol as a feature:** `validate_ghost_by_holdout(ts, labels, held_out)` —
  withhold a source population, run reference-free detection, score against the answer you get when it
  *is* in the panel. **This is the single most valuable thing we could add, because it is the thing the
  field has no way to do (§9.1).**

### 7.4 Admixture dating: close HAPMIX's 2009 open problem
HAPMIX's Discussion, still standing after seventeen years: *"our results motivate additional work to
enable detection of **multiple admixture events at different points in time**."* Under a double pulse
(λ = 6 + 100) it blurs to a single `T̂ = 45`. `fit_rate_through_time` is exactly this. **Validate**
against HAPMIX's own double-pulse sim, against **MOSAIC/GhostBuster coancestry curves** (the field
standard), and against ARGweaver-D's ascertainment-bias warning.

### 7.5 Orthogonal validation: the mutational-signature module
GhostBuster corroborates every call with signatures *not used in fitting* — TCC→TTC, GC-biased gene
conversion, PRDM9-A/C hotspot activity. **It is why their deep claims are believable, and it has
permanently raised the bar.** Simulation-only validation now looks thin.

### 7.6 Own the structured-coalescent concession
CLAUDE.md §6 concedes the mugration approximation. **ARGweaver-D does not have this bias.** Do two
cheap things: **quantify** it on simulations, and **front-end around it** (paint on ARGweaver-D or
SINGER ARGs — coalescent-calibrated times, no panmictic prior). Concede in print, cite ARGweaver-D and
SCAR, make the modularity-and-scale argument. **A reviewer who finds it first is a much worse outcome.**

### 7.7 The gaps MOSAIC exposes that we should simply close
- **Unbounded phase-error correction** (MOSAIC has it; RFMix has a weaker version; we have none).
- **Ancestral-population reconstruction** + `F̂st` to every panel (MOSAIC and HAPMIX both do it; we could,
  by posterior-sampling tips, and it was a headline contribution in both papers).
- **A richer panel↔source object.** Our `w_i` is a scalar per reference; MOSAIC's copying matrix is a
  full `K × n_panels` relationship. Theirs is richer. Consider a per-reference, per-ancestry credibility.

---

## 8. New features that create genuinely new capability

| Feature | Why nobody else can | Unlocks |
|---|---|---|
| **Posterior-weighted f-stats / PCA / selection scans** (§7.1) | Needs a calibrated posterior **and** an ARG-uncertainty band. RFMix saturates; Recomb-Mix/SALAI-Net/ARGMix are hard; MOSAIC/GhostBuster have no ARG posterior | §9.4 — recalibrating the ancestry-specific literature |
| **Per-locus power track** (§7.2) | Needs a *generative* model on the tree. MOSAIC's `E[r²]`/`Rst` are per-run, not per-locus | §9.2 — introgression deserts vs. power deserts |
| **Hold-out ghost validation** (§7.3) | Needs a system where the sources are **extant and sequenced**. **We have baboons** | §9.1 — the first real-data ground truth for ghost detection |
| **Rate-through-time dating** (§7.4) | HAPMIX's stated open problem; MOSAIC and GhostBuster fit pulses/grids, not a rate | §9.5 — one Neanderthal pulse or many? |
| **Cross-species portability** (have it) | Species-agnostic, no training. The whole ML lineage must retrain | §9.3 — the baboon/human comparison |

---

## 9. Poster-child candidates, ranked

### ⭐ 9.1 — *"Ghost introgression, validated: what baboons can tell us that humans cannot"*

**The problem.** Every claim about admixture from an unsampled population is **unfalsifiable** — the
ghost is gone. The consequence is exactly what you would predict: **the field's two strongest methods
disagree, publicly, on its highest-profile claim** (ARGweaver-D: ~1% super-archaic in Denisovans;
GhostBuster: no support). Nobody can adjudicate.

**The idea.** *Papio* is the natural experiment the field has been missing: six extant species, deep
divergences, pervasive well-documented hybridisation — and **every source population is alive and
sequenced.** So we can do the one thing impossible in hominins:

> **Withhold a source species from the panel. Run reference-free ghost detection. Score the recovered
> "ghost" tracts against the answer you get when the species *is* in the panel.**

**A real-data, ground-truthed calibration of ghost detection — the first.** Then, having *earned* the
method's credibility on a system where the answer is known, apply it to hominins where it is not.

**Why it works as a flagship:** general ✅; only-us ✅ (needs reference-free ghost detection **and** a
system with sampled sources — **and the baboons**); falsifiable ✅ — *it is the falsifiability
contribution*. **It converts our weakest position into the finding:** we are *not* the first
genealogy-native ghost detector (GhostBuster is; MOSAIC had the idea in 2019). But we can be the first
to **validate** one — and validation is a bigger contribution than another detector.

**Also: it is the piece Myers's group would find hardest to copy, because it needs *data*, not code.**

**Needs:** the hold-out protocol (§7.3); the archaic-query fix (§4.3); SINGER ensembles; baboon ARGs.
**🔴 Open question that gates everything: which *Papio* data does the group have — which species, what
coverage, and is there a characterised hybrid zone with known parental sources?**

---

### ⭐ 9.2 — *"Introgression deserts, or power deserts?"*

**The problem.** The Neanderthal-ancestry deserts — megabase regions, and the whole X chromosome —
anchor a large literature on hybrid incompatibility and selection against archaic alleles (**BDMI**,
already a keyword in Kasper's own project list). **Every published desert map is built from hard calls
by methods that cannot say "I don't know."**

**The question nobody has asked:** *is a desert a region with no archaic ancestry, or a region where no
method has power to detect it?* Deserts are enriched exactly where genealogical information is thin —
low recombination, low diversity, high background selection. **The confound is systematic and has never
been controlled, because no method could compute its own power per locus.**

**We can (§7.2).** Overlay the power track on the desert map: deserts that survive the correction are
**real** (and the selection story is strengthened); deserts that vanish are **power artefacts**, and a
chunk of the literature needs revisiting. **Either outcome is a major result.** ARGweaver-D already
noticed something adjacent — Hum→Nea introgression **is** present inside Nea→Hum deserts, i.e. the
depletion is *unidirectional*, which is hard to explain by simple selection.

---

### ⭐ 9.3 — *"Is human–Neanderthal admixture unusual? A baboon control"*

Human-archaic admixture is called exceptional — the ~2%, the deserts, the X depletion, the male bias.
**Exceptional compared to what? There is no control.** Run the *identical* pipeline on *Papio*, where
hybridisation is pervasive, the sources are sampled, and the phylogeny is known: are there deserts? Is
the X depleted? Does the ancestry-vs-recombination-rate signature replicate?

If the baboon pattern mirrors the human one → **hybrid incompatibility is generic to primate
admixture**, and the human deserts are not special. If it does not → **something is genuinely unusual
about *Homo* × Neanderthal**, and *that* is the finding.

Only tractable with a method that is **species-agnostic and needs no training** — which rules out the
entire ML lineage and anything needing a human genetic map. **Combines with 9.1 and 9.2: one baboon
dataset, three findings.**

---

### 9.4 — *"How much of ancestry-specific inference survives uncertainty?"*
Immediate consequences for a hot field (Irving-Pease et al. 2024 and the ancient-DNA selection-scan
literature); needs **no new modelling**, only §7.1. *If Ötzi's Bergamo affinity survives, we strengthen
a beautiful result. If it does not, that is a major result.* **Risk:** a critique paper. Frame as
*"here is how to do this correctly,"* never *"they were wrong."*

### 9.5 — *"One Neanderthal pulse or many?"*
Contested; HAPMIX named the methodological gap in 2009 and nobody closed it. **Best as a *section*, not
the flagship.**

### 9.6 — *"Archaic ancestry in Africans: the question ARGweaver-D could not answer"*
Live and contested. ARGweaver-D has low power for Sup→Afr because African `Ne` is large and it can only
use **two** African genomes; a true ghost tract is deep **and shared in contiguous blocks by a specific
subset of haplotypes** — a signature that needs *many* samples. **We can use hundreds.**
**High risk:** they state that *their* power did not improve with more African samples. **Simulate
before promising anything**, and pursue only *after* 9.1 has established a calibrated false-positive
rate on real data — otherwise this is exactly the unfalsifiable claim the field is drowning in.

---

## 10. Recommended combination

> **One instrument, one dataset, three findings — with falsifiability as the spine.**

**The paper:** *Calibrated, validated inference of ghost and archaic ancestry from genome-wide
genealogies — and what it changes about the human introgression landscape.*

1. **Method** — a generative ancestry CTMC on the *whole* genealogy (full-topology pruning, every edge
   span-weighted), calibrated posteriors, learned per-individual reference credibility, no training,
   running on an **ARG posterior** rather than a point estimate.
2. **Validation (9.1)** — the baboon hold-out: **the first real-data ground truth for ghost detection.**
   The paper's moral centre; we *earn* the right to make hominin claims.
3. **Finding A (9.2)** — introgression deserts vs. power deserts.
4. **Finding B (9.3)** — the baboon control: is the human pattern unusual?
5. **Coda** — with a calibrated FPR and an uncertainty band, revisit the ARGweaver-D vs. GhostBuster
   disagreement, and report what survives.

**Why this over Framing B alone?** They are compatible, but B is fundamentally a *critique*, and
critique caps out lower and makes enemies. **This makes the same points constructively** — uncertainty
arrives as *validation*, the baboons give us something nobody else has, and the deserts give a
non-specialist something to remember.

---

## 11. Steal list

| Take | From | Why |
|---|---|---|
| **`E[r²]`** (accuracy with no ground truth) | **MOSAIC** → GhostBuster, HAPMIX | Free for a calibrated method; the field expects it |
| **`Rst`** (panel-adequacy diagnostic) | **MOSAIC** | Separates "bad panels" from "old admixture." Better than anything we have |
| **Orthogonal mutational signatures** | GhostBuster | The bar for deep-time claims. Simulation-only validation now looks thin |
| **Held-out-chromosome log-likelihood** | GhostBuster | Principled `K` selection |
| **Coancestry-curve dating** | MOSAIC / GhostBuster / GLOBETROTTER | Field-standard cross-check on `fit_rate_through_time` |
| **False-positive control bands** | ARGweaver-D | Calibrates the FPR; familiar to this audience |
| **UCSC track hub / public browser** | ARGweaver-D, MOSAIC (95 populations!) | Why their results are usable and citable |
| **The mosaic simulator** | HAPMIX | Isolates *resolution* from *signal decay* |
| **`tspop`** for ancestry truth | hapla | Community standard |
| **Ancient + admixed samples as references** | ARGMix | Enlarges the panel — and admixed refs are what `w_i` is for |
| **Their public sims and benchmarks** | GhostBuster, Recomb-Mix, ARGMix, MOSAIC | Cheapest credible head-to-heads available |

---

## 12. Stop-claiming list

- ❌ *"First tree-native LAI"* — AncestralPaths, ARGMix.
- ❌ *"First generative genealogy-native LAI with reference-free ghost detection"* — **GhostBuster.**
- ❌ *"Generative and calibrated"* as differentiators — **HAPMIX is both**, and self-estimates its accuracy.
- ❌ *"Nobody else has a reference-panel label-noise model"* — **MOSAIC, 2019.** Ours is *finer-grained
  and feeds back*; say that instead.
- ❌ *"Nobody can estimate accuracy without ground truth"* — **MOSAIC's `E[r²]`, 2019.**
- ❌ *"Works on non-model organisms without a genetic map"* — SALAI-Net and Loter got there first.
- ❌ *"More accurate"* — we tie RFMix, lose intra-continentally, and have never run MOSAIC's,
  GhostBuster's or ARGMix's benchmark.

**What to say instead:** *we model the ancestry process on the whole genealogy, generatively, with no
training. Every competitor throws something away for cost — windows, donors, nodes, trees, topology —
and every one of them reads a single point-estimate ARG. We keep the whole tree, we keep every edge,
and we carry the uncertainty in the genealogy through to the answer.*

---

## 13. Risks

- **🔴 The baboon plan is load-bearing and unverified.** §9.1 and §9.3 assume *Papio* whole genomes
  across species with a characterised hybrid zone. **Check this before planning anything else.**
- **The power track may not work.** It assumes the inferred local tree is good enough that "power given
  the tree" is meaningful. On a bad ARG, power and accuracy are confounded. Test on simulations with
  known truth *and* known ARG error before betting a paper on it.
- **Myers's lab will not stand still.** Four papers in seventeen years, accelerating. **Assume
  GhostBuster v2** (K-way, uncertainty-aware) within a year. The baboon validation is the piece hardest
  for them to copy — **lean on it**.
- **The deserts question is obvious *once you have a power track*** — and not obvious without one. Move.
- **Do not let Part II delay Phase 0.** Nothing here is worth building if full-topology pruning turns
  out not to beat a thinned coalescence-time summary.

---

## 14. Housekeeping (CLAUDE.md corrections, verified against the PDFs)

- ✅ **Clear the `[verify-DOI]` flag (§10, §13).** `10.64898` is genuinely bioRxiv's current DOI prefix.
- ✅ **ARGMix author order correct**; **but the title has no "ARGMix:" prefix** — it is *"Graph
  transformer for ancient ancestry inference."*
- ✅ **ARGformer attributed to the Ioannidis "AI-sandbox" group** (not Lewanski) — confirmed; and the
  §10 characterisation (*embedding + retrieval, not calibrated painting*) is **correct**.
- ✅ **AncestralPaths resolved:** it *does* consume tree sequences, and it is **not calibrated**.
- ➕ **Add GhostBuster as the primary prior-art threat**, ahead of ARGMix.
- ➕ **🔴 Rewrite §10's MOSAIC entry.** It is currently one line ("nearest ARG-native but different
  objects"). MOSAIC is **the intellectual ancestor of GhostBuster and the origin of the panel↔source
  decoupling, panel-level reference QC, and accuracy-without-truth diagnostics.** The
  HAPMIX→GLOBETROTTER→MOSAIC→GhostBuster lineage belongs in the introduction of any paper we write.
- ➕ **Add:** FLARE, Loter, and two more 2026 bioRxiv LAI preprints surfaced in search — *Point cloud
  local ancestry inference (PCLAI)* and *Improving Local Ancestry Inference through Neural Networks*.
  The field is crowded and moving fast.

---

## 15. Immediate next steps, in order

1. **🔴 Answer the baboon question.** *Which* Papio data, what coverage, which hybrid zone? Everything in
   Part II hinges on it, and nothing else does.
2. **🔴 Read GhostBuster's Supplementary Note, and MOSAIC's model section properly** (§2.1).
3. **🔴 Run the GhostBuster head-to-head (§2.2) and the thinning ladder (§2.3).** These decide whether a
   methods paper exists.
4. **Add `r²`, `E[r²]`, `Rst`, plain accuracy** to `validate.py` (§4.1). Cheap; required by everyone.
5. **Write a MOSAIC runner** (§4.2). The strongest genotype-based comparator, and absent.

**The one-paragraph version.** GhostBuster took the paper we were writing — and MOSAIC shows it was the
fourth instalment of a seventeen-year programme by Simon Myers, so it was never going to be a surprise.
What that programme has *never* done is **propagate genealogical uncertainty**: every deep-time claim in
this field, theirs included, is read off a single point-estimate ARG fitted under a panmictic prior, and
GhostBuster's own limitations section names this as its binding constraint. We have posterior ARG
ensembles, calibrated posteriors and an uncertainty band, already built and measured. **But the paper
should not be a critique.** It should be the one thing this field has never had: **a ghost detector
validated against ground truth** — which baboons can supply and hominins cannot — and the human findings
that such a validated instrument licenses. **Three cheap measurements and one question about the baboon
data decide everything. Do those first.**
