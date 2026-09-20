<h1 align="center">BinML</h1>

<p align="center">
  <b>Multi-band, 6-class triage of progressively revealed microlensing and variable-star light curves.</b><br>
  A synthetic Roman-like benchmark for the <i>Nancy Grace Roman Space Telescope</i> Galactic
  Bulge Time-Domain Survey.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-orange.svg" alt="PyTorch">
  <a href="https://github.com/kunalb541/BinML/actions"><img src="https://github.com/kunalb541/BinML/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT"></a>
</p>

---

BinML classifies simulated Roman-like light curves into **six operational classes** from three
photometric bands (F146 at 15-min cadence, F087/F213 at 6-h). It is designed for triage as a
season is revealed. The present evidence is simulation-only and does not establish autonomous
real-time planet triggering.

The released checkpoint uses a **legacy Cycle-7-inspired schedule**, not the current GBTDS
definition: one 72-day season, 15-min F146 sampling, and separate 6-h colour visits. The current
design uses approximately 12-min F146 sampling, 66-s exposures, staggered F087/F213 visits, and a
multi-season programme. Results below describe the released legacy-schedule benchmark.

| class | meaning |
|---|---|
| **Flat** | no detectable event (baseline / noise) |
| **PSPL** | single-lens microlensing |
| **NonPSPL** | detectable non-PSPL structure from stellar- or planetary-mass-ratio binary lenses |
| **PeriodicVar** | short-period variables (RR Lyrae, eclipsing binaries, δ Scuti) |
| **LongPeriodVar** | Miras, semiregulars, OSARGs — the dangerous microlensing impostor |
| **Eruptive** | dwarf novae, Be outbursts |

Two design choices set it apart from a standard classifier:

1. **Detectability-conditioned labelling.** An event is labelled by what is *observable*, not by
   what we simulated. A microlensing event whose peak falls outside the season, or whose
   amplitude is buried by noise, is **Flat**. A binary whose caustic anomaly is below the noise
   floor is assigned **PSPL** under the adopted synthetic detectability rule. This rule prevents
   the classifier from being scored against injected structure that the simulation declares
   undetectable; the floor is a modelling choice, not a theorem about real data.

2. **A partial-season cascade objective.** During truncation training, a simulated binary is
   relabelled **Flat → PSPL → NonPSPL** as more of the season is exposed. The transition uses
   a truth-informed, noise-free anomaly-onset proxy. It therefore tests streaming behaviour in a
   controlled simulation; it is not an observable alert timestamp available to a live broker.

## The model

The network is a **convolutional stem feeding a small transformer encoder** — **505,479
parameters** — deliberately small, because the task is not data-starved (millions of simulated
events) but inference-heavy at survey scale.

- **Input: 3 bands → 156 tokens × 5 channels.** In the released legacy schedule, F146 is the workhorse (15-min cadence, 6912
  epochs / 72-day season); F087 and F213 are the colour bands (6-h, 288 epochs). Each band is
  binned into fixed token slots (F146→864, colour→96) and every bin carries `mean, min, max,
  observed-fraction, observed-mask`.
- **Conv stem with non-learned min/max carry lanes.** The stem downsamples each band to its
  token count (F146 864→108, colour 96→24, total **156 tokens**). A learned averaging filter
  would smear a ~2.9-mag caustic spike down to ~0.36 mag — the size of ordinary PSPL curvature,
  erasing the PSPL-vs-NonPSPL distinction. So two stem channels are reserved as **non-learned
  max/min pooling lanes** carried through every downsampling step: the caustic extremum survives
  *by construction*.
- **Transformer:** 4 pre-norm blocks, 4 heads, `d_model=96`, fused SDPA. Absent colour bands are
  masked out of attention (F087 drops out in ~38% of events, F213 in ~1%, under extinction).
- **Head:** masked attention-pooling → a flat **6-way** classifier. (A hierarchical head was
  tested and shipped worse.)

Details: [`docs/architecture.md`](docs/architecture.md).

## Results (held-out final-test set: 360,472 events)

Per-class F1 (population-weighted, selection-corrected):

| Flat | PSPL | NonPSPL | PeriodicVar | LongPeriodVar | Eruptive |
|---|---|---|---|---|---|
| 0.97 | 0.96 | 0.82 | 0.97 | 0.91 | 0.88 |

> The NonPSPL F1 (0.82) is precision-limited; its recall is 0.95. Of its false positives, 94.4%
> (final test, argmax, weighted) were generated as binaries but demoted to PSPL by the adopted detectability floor. They remain
> false positives for the stated observational task. This diagnostic may reflect sensitivity to
> sub-threshold structure or correlated simulation properties and is not a second precision
> estimate.

- **Completeness at fixed purity: 0.879** (the headline triage metric, rather than accuracy or
  F1). Average precision 0.952. NonPSPL-versus-rest calibration has weighted ECE **0.0533** and
  weighted Brier score **0.0209**.
- **The cascade, measured event by event (n=1000, F146 only, 0.5 d steps):** the model alerts on
  **89.0%** of eligible binaries within the season. Among detections that were not premature,
  the median first crossing is **+5.0 days** after a truth-informed, noise-free anomaly-onset
  proxy. The first crossing is *premature* relative to that proxy for
  **1.6%** of eligible binaries (95% CI 1.0–2.6%), rising to **4.5%** under a stricter definition
  of onset that requires the anomaly to stay detectable. The premature rate depends on the alert
  policy and on the onset definition far more than on the model — see
  `validation/cascade_reproduce_result.json`, which reports it under coarser grids, a
  two-crossing persistence rule, multi-band revealing, and an argmax rule.
- **Cascade ablation:** the matched 400-event comparison is an exploratory risk–coverage curve.
  Its thresholds and paired outcomes were selected on the same events, so its conditional
  McNemar values are descriptive and do not support confirmatory population-level inference.
- **Streaming scope:** the 1,000-event prefix scan contains eligible binaries only and measures conditional
  detection timing. An every-class scan of two held-out-pool shards (15,016 events; `validation/referee_round.json`, `mixed_class_stream{,_f146}`; shipped model, frozen complete-season threshold) measures the burden on our simulator: with all three bands revealed no flat or variable-star event alerts, and the scan raises 0.77 alerts per 1,000 events per day, 89% of them detectable anomalies at the simulated 5.5% prevalence (58% at 1% and 12% at 0.1% by prior shift); with F146 alone it raises 0.98 per day, 67% of them detectable anomalies (26% at 1%), and 4 eruptive variables alert. With three bands nearly all false alerts come from binaries whose anomaly falls below the detectability policy (29 of 32; 3 of the 1,759 generated single lenses with a PSPL label alert); with F146 alone 51 of those single lenses alert as well. The threshold is still the complete-season one; a streaming threshold calibrated on disjoint prefixes is untested.
- **Stress testing:** the full suite (14.9 million events) was scored with the **stage-5 checkpoint**, the
  released model's predecessor. Its first shards per quoted regime were regenerated with the same seeds, class mix
  and peak-time draw (a new realisation, not the same events: the cloud fleet's software was not pinned) and
  scored with both (`validation/stress_rescore_local.json`, paper Table `tab:stress`): on the same-prior part the
  released model reproduces its held-out macro-F1 (0.919 vs 0.919). The out-of-range sweeps expose failures; the
  low-q regime and faint sources at m 23.5-25 (below the sweep's 25-27.5) were training pools from stage 4, and stage 6 added pools overlapping the
  wide-separation and sub-day sweeps, so they are diagnostics of how far targeted coverage carries, not evidence of
  broad population validity.

Full methodology and the honest reading of each number: [`docs/evaluation.md`](docs/evaluation.md).

## Install

BinML is not yet on PyPI; install the inference package (torch ≥ 2.0 + numpy, weights bundled)
from source:
```bash
pip install git+https://github.com/kunalb541/BinML.git
```
For the full simulation, validation, and paper pipeline, install all optional dependencies
(including `VBBinaryLensing`):
```bash
git clone https://github.com/kunalb541/BinML.git && cd BinML
pip install -e ".[all]"
```
The conda environment is an alternative:
```bash
conda env create -f environment.yml && conda activate binml
```

## Classify a light curve (Python)

```python
import binml
clf = binml.Classifier()                      # BinML 1.0 (6-class), CPU, weights bundled

# multi-band: {band: (time_days, magnitude)}; F146 required, colour bands optional
r = clf.predict({"F146": (t146, m146), "F087": (t087, m087), "F213": (t213, m213)},
                m_base_ref=22.1)              # F146 baseline magnitude (recommended)

print(r)                    # <BinML NonPSPL 0.98 | microlensing 0.99 anomalous 0.98>
r.probabilities            # {'Flat':.., 'PSPL':.., 'NonPSPL':.., 'PeriodicVar':.., ...}
r.is_microlensing          # P(PSPL)+P(NonPSPL)
r.is_anomalous             # P(NonPSPL) -- anomalous binary-lens score

# single band (F146 only) is fine:
r = clf.predict(t146, m146, m_base_ref=22.1)

# streaming probabilities as the season is revealed
days, probs = clf.predict_evolution({"F146": (t146, m146)}, m_base_ref=22.1)
```
Command line: `binml classify lc.csv --m-base 22.1` (`--m-base` is required; see `--estimate-baseline` for the caveated fallback). More: [`docs/usage.md`](docs/usage.md).

## Train / evaluate from scratch

```bash
# one shard of simulated data (all 6 classes) -> cache -> train -> evaluate
python -m pipeline.run_shard --shard 0 --n-shards 1 --out data/raw
python -c "from pipeline.cache import build_cache; import glob; \
build_cache(sorted(glob.glob('data/raw/*.h5')), 'data/cache/shard_00000.h5')"
python -m pipeline.to_memmap --in-dir data/cache --out data/mm
python -m pipeline.train    --cache data/mm --out runs/binml.pt --epochs 6 --device mps
python -m pipeline.evaluate  --ckpt runs/binml.pt --cache data/mm_test --out eval/
```
File-by-file walkthrough: [`docs/pipeline.md`](docs/pipeline.md).

## Simulating data at scale (AWS)

The simulator runs at two scales from one codebase — a single shard on a laptop, or thousands
across a transient cloud fleet — because the modules take `--bucket`/`--prefix` and write locally
if you omit the bucket. Generation is embarrassingly parallel:

- **Index-addressed shards.** One process builds one shard (an HDF5 file of a few thousand light
  curves), with `seed = seed_base + shard * 7919`. Workers HEAD the key
  `prefix/shard_NNNNN.h5` and skip it if present, so a Spot interruption costs at most one shard
  within a frozen run configuration. This is **not content addressing**: the skip check does not
  compare source, simulator configuration, regime, or seed base. Use a new prefix for every such
  change and preserve a run manifest. Work is split by a modulo partition
  (`--worker W --workers N`).
- **`--seed-base` makes evaluation honest.** Train/val/test used base `20260720`; a far-off base
  (e.g. `900000000`) gives a different PCG64 stream — parameter tuples the model did not see in
  training. The full stress suite has a 4.5M same-prior subset, 8.7M events in 12 targeted regimes (all of them
  enriched pools also used in training) and 1.7M in 17 out-of-range sweeps (scored with stage 5; the released model's
  numbers come from a regenerated subset).
- **Binning & inference run in-region.** Raw shards are ~312 MB each (~125 GB for a full run);
  binning them to compact caches (~46 MB) and running the model *in the S3 region* means the
  light curves never leave — only the compact predictions (~30 floats/event) come back.
- **Self-terminating fleets.** Free-tier Spot instances install deps from a code tarball, refuse
  to start without `VBBinaryLensing` (else binary lenses silently degrade to single-lens and
  ruin the dataset), generate → upload → shut down; a watchdog force-terminates any stragglers.

The `aws/` launch scripts are account-specific infra (hard-coded bucket/region/IAM) and are kept
out of the public tree; the `pipeline` modules underneath them are what run. See
[`docs/pipeline.md`](docs/pipeline.md).

## Repository layout

```
binml/             pip package — the 6-class inference API (Classifier, predict, CLI) + weights
binml/legacy/      the earlier 3-class (Flat/PSPL/Binary) model, preserved for provenance
pipeline/          simulation, training & evaluation modules (run via `python -m pipeline.<mod>`)
docs/              architecture, pipeline, evaluation, data format, usage, model card, leakage audit
paper/             software paper + references
examples/, tests/  usage examples and smoke tests
aws/               (local, gitignored) account-specific fleet-launch scripts
```

## Limitations & intended use

- **Legacy Roman-like cadence.** Trained on dense 15-min F146 sampling for one 72-day season.
  The current survey design differs in cadence, exposure, colour interleaving, and multi-season
  structure. Binary *characterization*
  needs that density — the short caustic anomaly must be observed. On sparse ground-survey
  cadence (LSST, multi-day gaps) detection degrades and characterization is not recoverable.
- **The shipped checkpoint requires a continuous F146 season; the survey as simulated in RMDC26
  is not one.** Found 2026-08-23, after submission. The RGES-PIT RMDC26 release (a GULLS simulation
  of the survey) pauses F146 for 43-44 h per season, in six or seven pauses whose phases differ from
  season to season (`validation/gulls/rmdc26_schedule.json`). The training grid has no such gaps, and
  the model reads an empty mid-season bin as evidence against a single lens: imposing the pauses on
  our own held-out seasons collapses single-lens and Flat recall and also degrades the eruptive and
  long-period variables and NonPSPL (`validation/gulls/schedule_finetune.json`). A single gap of
  0.5 h costs nothing; one gap of 1-2 h at mid-season (day 43) already costs about 7 points of single-lens recall
  (the cost depends on where the gap falls).
  **For a gapped schedule use the recommended gap-aware checkpoint**, shipped as `binml/weights/binml-gapaware.pt`
  (`binml.Classifier(weights="gapaware")`, CLI `binml classify --weights gapaware`; identical to
  `validation/gulls/weights/ft_fspl5s_seasons_g08.pt`: finite-source single lenses and the measured RMDC26
  season pauses in training) at its own threshold, 0.956 (`binml.GAPAWARE_THRESHOLD`), chosen at 90% purity on our own simulated
  seasons with the measured pauses imposed (68% of validation slices: 0.947-0.965). On 56,975 RMDC26 events
  it flags 2.4% of single lenses at that threshold (1.8-3.1% across the slice spread; 3.5% weighted by
  RMDC26's event rates) with planetary recall 0.29 / 0.33 (1S2L / 2S2L); at the shipped threshold it
  flags 6.3% against the shipped weights' 35.6%. It replaced the earlier finite-source checkpoint
  `ft_fspl5s_g08.pt` (threshold 0.957, 1.6%, recall 0.24 / 0.27) on 2026-09-12: the two tie at matched
  false-alarm budgets per simulated event, and the new one is ahead when events are weighted by rate and
  on our own held-out seasons with the pauses (clean 0.929 vs 0.931). Two further training seeds of the same recipe, trained afterwards, trail
  `ft_fspl5s_g08.pt` per event (recall 0.349 / 0.358 against 0.390 at a 5.2% false-alarm budget) and tie it
  weighted, so RMDC26 does not separate the two recipes; the released run has the highest planetary recall of the three
  seeds at every table budget and at its own threshold, although another seed flags fewer single lenses at the
  shipped threshold.
  RMDC26 also guided the diagnosis and the choice of this checkpoint, so these numbers are optimistic for it. Under our own label policy 44-46% of the selected RMDC26 planetary
  events carry no anomaly the policy would claim within one season; on those with one, recall at 0.956
  is 0.41 / 0.48. The shipped weights are unchanged so that the submitted numbers stay exact. Full
  account, including what did not work and what eight verification rounds corrected, in
  [`paper/REVISION.md`](paper/REVISION.md) §1½, [`docs/VERIFICATION_2026-09-11.md`](docs/VERIFICATION_2026-09-11.md)
  [`docs/VERIFICATION_2026-09-12.md`](docs/VERIFICATION_2026-09-12.md) and
  [`docs/VERIFICATION_2026-09-12b.md`](docs/VERIFICATION_2026-09-12b.md) (with its addenda); the post-submission
  reference check and code self-check are in [`docs/VERIFICATION_2026-09-20.md`](docs/VERIFICATION_2026-09-20.md). The RMDC26 tables regenerate
  from a clone via `validation/gulls/rmdc26_scores.csv.gz` (the `command_from_clone` field of
  `validation/gulls/transfer_tradeoff_all.json` is the exact `gulls_summary_tables.py --scores` invocation); the scan,
  sub-day, relabelling, floor, between-season, occupancy, noise-model, schedule and dataset-fact results need a rerun on
  the pinned release (paper, Data availability).
- **Known weak spots** (documented in [`docs/model_card.md`](docs/model_card.md)): within the training prior, wide
  binaries at s > 2.9 and q >= 0.1 (argmax recall 0.70; over half of all argmax misses); in targeted out-of-range tests:
  faint sources m>25 (noise-dominated: 12% of faint microlensing events without a detectable anomaly exceed the
  operating threshold, against 0.9% in the natural population), detectable sub-day single lenses tE 0.2-1 d (69%
  called anomalies, 35% above the operating threshold), wide caustics s>5 (anomaly recall 0.33 on 64 events, in a
  regenerated sweep that labels 27% fewer detectable anomalies than the suite).
- **Synthetic support, not a population forecast.** The simulator uses broad analytic training
  supports, including an authored truncated-lognormal timescale distribution anchored to a
  literature mean. Variable-star curves are analytic or phenomenological shapes, not sampled
  OGLE templates. The 0.02-mag detectability floor has not been validated on Roman data.
- **Known colour-photometry mismatch.** Relative to the current Roman calibration, the released
  simulator's F087 and F213 zeropoints are optimistic by about 0.10 and 0.14 mag, respectively;
  its F087 saturation limit was carried from a longer exposure and is too faint (at equal exposure
  F087 saturates ~1.3 mag brighter than F146, not fainter), and its colour-band
  background ratios do not reproduce the published thermal backgrounds. Measured on our simulator (`validation/referee_round.json`, 15,016 events of two held-out-pool shards): with the audited colour photometry the shipped model's anomaly ranking is unchanged (AP 0.958 vs 0.957), but the variable classes lose precision (periodic 0.949 to 0.908, eruptive 0.806 to 0.776; F1 0.970 to 0.949 and 0.891 to 0.871) because more flat sources are called periodic and more single lenses eruptive; recall is unchanged. Short fine-tunes on each calibration (one seed each) keep AP; the audited-calibration fine-tune falls to periodic-variable F1 0.750 on the old photometry (0.872 at its last epoch), against 0.961 (0.972) for the old-calibration fine-tune; on the audited photometry the two differ at the kept epoch (0.957 vs 0.917) but not at the last (0.967 vs 0.967). The released model has not been retrained with corrected values.
- **Provide `m_base_ref`.** The model input is baseline-relative; give the F146 quiescent
  magnitude when you have it (a catalogue value). The faint-tail estimate is only reliable for
  short, well-sampled events.
- **Intended use:** research triage/vetting of partial Roman-like seasons. The +5-day conditional
  latency does not demonstrate response during a short planetary perturbation. Use the output to
  prioritise modelling and review, not as an autonomous discovery or follow-up trigger.

## Documentation

- [Usage](docs/usage.md) · [Pipeline](docs/pipeline.md) · [Architecture](docs/architecture.md)
- [Evaluation](docs/evaluation.md) · [Data format](docs/data_format.md) · [Model card](docs/model_card.md)
- [Glossary](docs/glossary.md) · [Leakage audit](docs/leakage_audit.md) · [Legacy 3-class model](docs/legacy_3class.md)
- Runnable example: [`examples/quickstart.py`](examples/quickstart.py)
- Notebooks (executed, with plots; run by the test suite):
  - [`examples/00_quickstart_synthetic.ipynb`](examples/00_quickstart_synthetic.ipynb): offline, under a minute:
    simulate events, classify them, and plot the class probabilities as a season is revealed.
  - [`examples/01_classify_roman_event.ipynb`](examples/01_classify_roman_event.ipynb): fetch an event from the
    RGES-PIT RMDC26 release (GULLS simulation of the Roman bulge survey, pinned Hugging Face revision; needs network
    access), classify it with the gap-aware checkpoint at its own threshold, plot its light curve and P(NonPSPL) as
    the season is revealed in half-day steps, score an event that peaks between seasons, and read the population
    numbers from the committed artifacts.

## Citing

The methods paper is under review at *Astronomy and Computing*. A preprint is available:

> Bhatia, K. *BinML: Detectability-Conditioned Anomaly Triage for Synthetic Roman-like Partial
> Light Curves.* SSRN preprint, 2026. https://doi.org/10.2139/ssrn.7295158

```bibtex
@article{Bhatia2026BinML,
  author  = {Bhatia, Kunal},
  title   = {{BinML: Detectability-Conditioned Anomaly Triage for Synthetic
             Roman-like Partial Light Curves}},
  journal = {SSRN preprint},
  year    = {2026},
  doi     = {10.2139/ssrn.7295158},
  url     = {https://ssrn.com/abstract=7295158}
}
```

For the software itself, see [`CITATION.cff`](CITATION.cff). The state submitted to the journal
is tagged [`submitted-ac-2026-08-15`](https://github.com/kunalb541/BinML/releases/tag/submitted-ac-2026-08-15).
