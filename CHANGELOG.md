# Changelog

## Unreleased — reference check and code self-check (2026-09-20)

Manual, no agents; record: `docs/VERIFICATION_2026-09-20.md`. All 48 references were read at their published
record and checked against the sentence each supports: every claim is supported and every bibliographic detail
matches. Six entries are incomplete or stale (missing article numbers for Wyrzykowski 2015, Kim 2018 and
Zhang 2021; no DOI for Gaudi 1998; PLAsTiCC now published as ApJS 267, 25; the Roman zeropoints file path moved),
left as pending fixes. `pipeline/evaluate.py` was read end to end and its headline numbers, split, label
consistency and intervals reproduced from the frozen artifact. Four documentation statements disagree with the
code without moving a number (the leakage audit and glossary say recall is never reweighted, but Table 3's recall
is; a stale docstring in `mix_finetune.py`; "effective events" means a sum of weights in Fig. 9 but Kish n_eff in
Sec. 8; no length assertion between memmap and label columns). Also records how the defect classes of an external
review of the author's CAMELS paper map onto this codebase.

## Unreleased — presentation revision (2026-09-15)

Answers the referee's presentation comment (item 8): the paper was hard to read because nearly every sentence carried
several numbers, qualifiers and cross-references, and the hedging hid which findings are robust.
- **Principal results:** the Introduction now ends with a ranked list of the five findings that survive our checks,
  followed by the two that are conditional and the four that are not established.
- **Abstract:** six numerical values instead of about twenty (233 words); the hedges the eighth check restored are kept
  (F146 alone, truth-informed onset, "mostly" the pauses, per simulated event, optimistic, not yield forecasts).
- **Sections 7-8 shortened by a third** (2,990 to 2,034 words of prose). The RMDC26 results are a four-step ladder of
  checkpoints; the stratum comparison of the RMDC26 cascade and the input, label, between-season and sub-day checks
  moved to tables with their hedges in the notes.
- **New Appendix C** holds the second-order numbers: training-label fidelity, the every-class stream, the alert-policy
  table (moved from Sec. 4) and the stress table (moved from Sec. 7), the operating point under prevalence, floor, colour and cadence changes, the matched-density
  cadence test, the RMDC26 cascade by stratum, and the other RMDC26 checks. The two ablations are a table in Sec. 9.
  Sections 4, 6 and 9 lost 10-14% of their prose; the Conclusion carries no numbers.
- **Hedge audit:** every hedged sentence of the previous text was checked against the new text and table notes;
  "largely intact" (floor) and "mostly" (Conclusion) were restored during the audit.
- **Layout:** `twocolappendix` and narrow-first table order keep the appendix tables from sitting one per page;
  24 pages (the main text ends two pages earlier than before).
- **Final check (same day):** clean rebuild from the committed tree, every table entry and label read against its
  macro's definition, the efficiency-plane wording tested cell by cell, hedges re-audited; two wordings sharpened
  ("regimes tested below", "the finite-source recipe without the measured pauses"); the alert-policy table moved to
  Appendix C after the class stranded it alone on page 9 behind the full-page Figure 4.
- **Code:** a new fail-closed guard in `paper/make_gulls_macros.py` (long-period variables have the largest
  truncation-label disagreement, which the text now states) with a perturbation case in
  `tests/test_macros_fail_closed.py`.

## Unreleased — eighth check of the cut and the new abstract (2026-09-15)

A narrow check of the unreviewed commits 52e2d8c..e65a510 (the seventh-round fixes, the length cut, the new abstract
and Appendix B; record: the fifth addendum of `docs/VERIFICATION_2026-09-12b.md`) found that the cut had removed
hedges the data still require and that the new abstract overclaimed. Restored or corrected:
- **Abstract:** no survey cadence claim (15 min is our legacy benchmark; the survey samples F146 every ~12 min);
  labels follow an *adopted* detectability policy (not "what no survey could claim"); the half-day numbers are F146
  alone and the stream numbers three-band; the RMDC26 failure is "mostly" the pauses; RMDC26 guided the checkpoint
  choice, so its numbers are optimistic; "artifacts are public" without "every number". 247 words.
- **Hedges the cut had dropped:** detection lower at high mass ratios only *nominally* resolved (one of six
  comparisons) and the intermediate-ratio lag *at the edge of resolution*; weighted by event rate the finite-source
  physics resolves *nothing*; the measured pauses lead random gaps in *recall* in both weightings (false alarms per
  event only); the stress failures are not *established* limits of the method; *essentially* none of the training
  events has an empty mid-season bin; the NonPSPL false-positive diagnostic *suggests* (or correlated source
  properties); the floor arms are partly overlapping draws; the colour boundary depends on where training stops; the
  RMDC26 burden is slightly optimistic; the finite-source cause is the *likely* one and explains only part.
- **Wrong populations or pointers:** the gap-augmentation numbers are now the relabel-off arm the text describes;
  the baselines are named with their scores; q < 1e-3 for the 0.892 slice; the 43% sub-day demoted share is the
  regenerated tier's; the conclusion's stream numbers are three-band; the 19.9% is of the final-test split; Data
  availability again says the referee-round results reduce from the runner's work directory; captions of Figs. 3, 4
  and 9 and Tables 6, 8, 9 and 10 restored their conditions (relabel rows, full-precision comparison, blanking rules).
- **Code and records:** the probability-evolution figure's self-checks now match the sentences as they stand; the
  Appendix B table's bin edges are guarded (with a case); `paper/README.md` lists `make_gulls_macros.py`,
  `render_draft.py` and the committed outputs; `REVISION.md` notes where the cut moved statements to the draft.

## Unreleased — length cut (2026-09-14)

The revision had doubled the manuscript (7,900 to 15,200 words of prose; 25 to 32 pages): every verification round
closed findings by adding hedges, disclosures and sensitivity prose and never removed any. Eight sections were
rewritten to 8,300 words and 21 pages with every table and figure kept, but secondary numbers and many hedges
cut from the text (the eighth check restored the hedges the data require; the dropped numbers remain in
`paper/draft_gulls_section.tex` and the artifacts): one statement of each result in
the section that owns it, one hedge per result, provenance only in Data availability, and no restating of tables in
prose. The abstract was rewritten around the paper's two ideas and its cross-simulator finding (249 words, a third
of the numbers). The RMDC26 transfer table lost its four source-size columns, which moved with the seed-spread
table to a new Appendix B (`outputs/gulls_rho_table.tex`, a new generated output), so no page is a lone rotated
table. The RMDC26 section went from 3,600 to 1,600 words, Sec. limits from 3,150 to 1,180, Data availability from
870 to 280. `paper/draft_gulls_section.tex` is no longer a block-by-block mirror of the paper; it stays as the
long-form record behind `paper/REVISION.md`. Guards in the macro generators were left in place (they check the
artifacts and the docs, not only the paper).

## Unreleased — seventh check and per-class probability figure (2026-09-14)

- **New Figure (Sec. cascade): class probability at every half-day cut for a clear and a marginal event of each class,**
  with the 16-84% band over 30 re-observations of the same event with fresh photometric noise
  (`paper/make_data_figures.py`, which checks the caption's statements on the events it draws). The clear events
  settle once their evidence appears; the marginal ones wander, and the noise draw alone moves their calls.
- **Seventh check** of the sixth check's fixes (record: the fourth addendum of `docs/VERIFICATION_2026-09-12b.md`; 22
  findings, none refuted). The ablation paragraph now says only what the 7.2 d onset grid records; the matched-density
  test gained a full-cadence arm (0.94: thinning costs recall even without empty bins) and counts empty bins with the
  model's own binning; the calibration paragraph attributes the error to the over-confident mid-range; one stratum
  difference is "nominally" resolved; more RMDC26 results are listed as needing the extracted caches; guards for "at
  the edge of resolution", "later alerts", the plane minimum's location and a stale in-house copy; unit tests for the
  new reduction helpers; docs and records corrected.

## Unreleased — sixth check (2026-09-14)

Three reviewers of the fifth round's fixes, their adversarial checkers and the re-run completeness critic (record: the
third addendum of `docs/VERIFICATION_2026-09-12b.md`; 36 findings, 1 refuted, and 8 critic gaps). What changed:
- **In-house cascade, alert grids.** The daily and two-day grid rows of Table tab:policy had started at 0.5 d and so
  never evaluated the 72 d cut. Anchored at the season's end they give 1.2% premature at 88.7% detection (daily) and
  1.0% at 88.4% (two-day), not 1.6% at 86.4%: a coarser grid lowers the premature rate at almost unchanged detection.
- **Truncation-augmentation ablation.** Its premature rates use the generator's 7.2 d onset grid; almost all of the
  augmented arm's premature alerts fall between the grid onset and the cut before it, and the alerts at or before that
  cut (premature on the grid and under the persistent onset) favour the augmented arm under both rules (0 vs 32, 9 vs
  45), so the argmax reversal is no longer claimed.
- **Throughput.** 1,015 light curves per second on an otherwise idle M5 (the committed 359/s had been measured under
  load); the benchmark now records the load average, code and checkpoint.
- **Matched-density (KMTNet-style) test.** A gap-free regular arm shows that the collapse to zero at low density comes
  from empty two-hour bins, not the visit count (regular grid 0.62/0.59/0.57 where random thinning falls to 0; the
  seventh check added full cadence, 0.94, so thinning itself also costs recall); read from its artifact directly.
- **Sec. results.** Wide binaries are missed at a similar rate only up to dchi2 = 1e6 (14% above); the low-q corner
  is described as a corner; the weighted calibration's mid-range over-confidence is stated; the false-negative
  definition, the labelling-ablation wording and the baselines' own class mix are made exact.
- **RMDC26 cascade strata.** Exact counts, Fisher tests and bootstrap lag-difference intervals are recorded and
  guard the text (premature about as frequent, p >= 0.36; detection lower, resolved only at high ratios; lags +2 d at
  the edge of resolution and +0.5 d unresolved).
- **Data availability.** Every RMDC26 result that needs the extracted caches is listed; the full stress set's column
  is an exception; the generator-identity claim is scoped to the commits compared, with the legacy switches named;
  training measurements join the typed-constants exemption.
- **Code and tests.** Reused stress evaluations must match their checkpoint; `--from-archive` no longer overwrites the
  artifact; the July-shard test is exact on macOS arm64 and within 3 labels elsewhere; the legacy attributes are dated;
  every new guard has a perturbation case, and seven older ones gained theirs. The archive re-derivation test compares
  at floating-point precision (CI's Linux Python 3.11 differs from macOS in the last bits of the unrounded sums).

## Unreleased — fifth verification (2026-09-14)

Six verifiers and their adversarial checkers checked the fourth round's fixes; the round's completeness critic stalled
and returned nothing, so a sixth check (three reviewers of these fixes, their checkers and the critic) followed (record:
the second and third addenda of `docs/VERIFICATION_2026-09-12b.md`). What changed:
- **Sec. results, where the misses sit, corrected again.** The fourth round's "not concentrated at the weakest
  anomalies" was wrong: there are two sources. Wide binaries (s > 2.9, 11% of detectable anomalies, 56% of the
  NonPSPL-to-PSPL confusions) are missed 26-30% of the time at every evidence strength up to dchi2 = 1e6 (14% above it;
  the sixth check found the open top bin had hidden that decline), almost all at q >= 0.1; at
  smaller separations the misses concentrate at weak anomalies (dchi2 < 2000: 19% of those anomalies, 56% of their
  misses). The labelling-surgery numbers are now derived on the final test (stored and weighted: 37.4% of stored
  generated binaries keep an anomaly, 8.2% of all generated ones); the false-positive diagnostic is 94.4% (was 94.7%,
  the whole pool's any-label share). New guards, each with a perturbation test.
- **Faint-sweep precision re-attributed.** Faint photometry, not the sweep's class mix, drives the fall (0.15 at the
  natural class mix against 0.73; the natural population at the sweep's mix 0.64); the 0.44 prior shift is no longer
  quoted. The regenerated wide-separation sweep labels 27% fewer detectable anomalies than the suite (disclosed; the
  other tiers are checked to agree). The stress numbers now re-derive from an archive of their per-event inputs
  (`validation/stress_rescore_archive/`, `--from-archive`, tested); the legacy switches are pinned against the July
  shard; tier stamps can no longer be invented.
- **RMDC26 cascade:** the burden's optimism compared like for like (F146 alone: 4.8% vs 5.8%, not vs the three-band
  6.3%); the stratum comparison states counts and what is resolved (detection lower in both strata, the intermediate-ratio
  difference unresolved; not pooled, because the compositions differ; premature counts
  2 of 149 vs 2 of 138 and 21 of 826 vs 2 of 48; RMDC26 higher with three bands); strata named by mass ratio; 1% of
  RMDC26's eligible anomalies have q > 1e-2; the artifact records its code.
- **Appendix and Data availability:** the cause of non-identical regeneration is the unpinned cloud environment, not
  code (the generator gives the same shards for a held-out shard at the commits compared; the sixth check found the
  out-of-range sweeps did change after July); the v5 fleet's region and instance size
  sourced from its launch scripts; class-balanced loss stated; throughput and hours on one basis (22 h including a
  stretched epoch, about 17 without); training-configuration values exempted from the "every number" statement; the
  three RMDC26 and the referee-overlap exceptions named [the sixth and seventh checks found more RMDC26 results that
  need the extracted caches; the paper now lists nine].
- **Abstract** cut to 247 words (Astronomy and Computing's limit is 250). **F087/F213 constants** checked against
  Roman's published tables (2024-03-01 zeropoints, 2024-06-03 thermal backgrounds): the audited values match; the
  F087 saturation offset is 1.3 mag, not 1.2.

## Unreleased — fourth verification of the final state (2026-09-13)

Six verifiers, six adversarial checkers and a completeness critic checked the state after the third round's fixes
(record: the addendum of `docs/VERIFICATION_2026-09-12b.md`). What changed:
- **Stress subset made faithful and described honestly.** The July generator kept the peak time drawn for the
  unperturbed timescale in out-of-range sweeps; `run_shard --legacy-t0-pad` reproduces it, and the wide-separation,
  long-period and sub-day tiers were regenerated with it (the corrected sub-day tier is kept as a second row). The
  subset is a new realisation of the suite's populations, not its events (the fleet's software was not pinned):
  checkpoints are compared within it. Each tier records the commit that generated it. The faint sweep's anomaly
  precision was first attributed to its class mix (withdrawn by the fifth verification: faint photometry drives it), and the
  paper quotes the false-anomaly rates among microlensing events instead; the
  abstract quotes the operating-threshold rates (35% of sub-day single lenses, 12% of faint microlensing events
  without an anomaly, 0.9% in the natural population). Stage-4/6 training coverage of the regimes stated correctly.
- **RMDC26 cascade compared stratum by stratum** with the in-house scan (giant planets and Neptunes, F146 and three
  bands; 40% of RMDC26's eligible anomalies have q ≤ 1e-4, which the in-house scan samples with 9 events), the
  single-lens burden given with its interval and its slight optimism, and every directional sentence guarded.
- **The referee round's per-event archive was never committed** (gitignored); it is now, with a test that every
  hashed file is tracked and matches.
- **Appendix:** the simulation paragraph had described an earlier 10M-event run; the training recipe now gives the
  one-cycle peak learning rates from the checkpoints' optimizer states (base 4e-4, not 3e-4), the base run's early
  stop, the fine-tunes' batch and class factor, and the logged epoch throughput instead of an unsourced benchmark.
- **Smaller text fixes:** held-out regeneration reproduces 75% of the pool's events (not "identical"); "almost always
  delays" quantified (212 of 241); a single mid-season gap's lost single lenses go to NonPSPL (not PeriodicVar);
  RMDC26 sub-day single lenses compared at matched timescales; several ledger rows, README/model-card/evaluation
  statements and the notebook pointer corrected.
- **Tooling:** `make_macros.py --list-inputs` and a manifest test for its inputs; the fail-closed tests assert the
  guard each case is named for and cover the new stress and cascade guards; `paper/build.sh` reruns LaTeX until
  references settle.

## Unreleased — third verification of the revision work (2026-09-13)

Seven verifiers, seven adversarial checkers and a completeness critic re-checked everything added since the second
pass (record: `docs/VERIFICATION_2026-09-12b.md`). What changed:
- **Seed rule restated.** The three seeds of the recommended recipe are a rough scale, not a test (two runs of one recipe
  exceed a three-run range about a third of the time), applied per budget and weighting in a new table (Table tab:seeds).
  The first version's verdicts were partly wrong: the released run's weighted lead over `ft_fspl5s_g08.pt` (0.024) does
  exceed the weighted range (0.022), and the extra training does not lead at every budget. Now: the measured pauses lead
  at every budget in both weightings; the extra training carries false alarms and the low budgets, the finite-source
  physics the 11.7% budget, mean recall and the largest rho/|u0| bins, per event; weighted the physics resolves nothing;
  the recommended recipe's lead is not established. "Best of three seeds" narrowed to planetary recall.
- **Truth-label measurements redone against the generator's own rule** (a single-lens refit on the surviving epochs of
  each binary's rebuilt noise-free curve). The first measurement used the first-detectable onset as reference, which the
  fixed truncation rule matches by construction; now: released labels wrong for 10.8% of truncated binaries, the fixed
  rule 2.4% with the half-day onset; pause relabel 9.4% (was 10.5%, both directions), relabel off 2.5% (was 1.4%); LPV
  16.3% (was 16.4%, double rounding). Truth binning fixed (4% of epochs one bin early); mixing truth and no-truth
  caches now refused; truth relabelling refuses a cache made at another floor and warns about the 7.2-d onset.
- **Referee round:** the floor arms are partly overlapping draws (a fifth / a twentieth of events shared), not
  independent; shards 90-91 are held-out-pool shards with ~15% threshold-selection rows; AP and F1 now on every event;
  the colour effect is lower variable-class precision (more flat sources called periodic), not worse rejection of
  variables; the colour fine-tunes are single runs and support only the collapse on mismatched photometry; the
  every-class stream also run with F146 only; single-lens stream alerts split into generated single lenses and
  demoted binaries; per-event inputs archived in `validation/referee_round_archive/`.
- **Paper:** stale Table 5 note (the Paczynski fit is at full cadence), a fourth contribution in the Introduction,
  within-season pauses in the list of schedule differences, the non-monotone onset partly numerical, the abstract
  and Conclusion no longer call the three-band scan "the same way", weightings and units labelled in the RMDC26 alert
  burden, "no deployed operating threshold", detection comparison flagged as mixing checkpoint and simulator; the
  in-house pauses-vs-random-gaps AP differences stated to be within the seed spread; the Conclusion's list of open
  tests rewritten; the RMDC26 fine-tunes' uncommitted working trees disclosed.
- **Two errors older than the revision, found by the completeness critic.** (1) The 14.9M-event stress suite was scored
  with the stage-5 checkpoint, the released model's predecessor; the paper had quoted its numbers as the released
  model's. `validation/stress_rescore_local.py` regenerates the first shards of each quoted tier with the suite's
  seeds, class mix (`run_shard --legacy-oor-mix`) and, for the sweeps, its generator's peak-time draw
  (`--legacy-t0-pad`, added after the fourth verification), and scores the same events with both checkpoints; the
  paper now quotes the released checkpoint on that subset (Table `tab:stress`), compares checkpoints within it, and
  says which model scored the suite. The released model reproduces its held-out macro-F1 there (0.919). The suite's
  sub-day "PSPL recall 0.524" mixed in demoted binaries: the sub-day single lenses are classified PSPL 0.285 and
  called anomalies 69% of the time (35% above the frozen threshold). The faint sweep's anomaly precision (0.026) is
  driven mostly by faint photometry (0.15 at the natural class mix); among faint microlensing events without an anomaly the false-anomaly rate is 26%
  (3.3% natural). The low-q regime and faint sources at m 23.5-25 were stage-4 training pools and stage 6 added pools overlapping two
  sweeps; the text says so. The "10.4M out-of-distribution" figure of earlier entries counts 8.7M events in enriched
  in-prior regimes and 1.7M in out-of-range sweeps. (2) The appendix training recipe described the base run's
  values not from the stage logs (batch 256 and 3e-4 are train.py defaults; 5 epochs matched no stage) and an earlier
  10M-event AWS run; it now
  describes the six-stage chain from the stage logs and checkpoint optimizer states (`paper/canonical_numbers.json`).
- CI and tooling: the paper-build workflow checks both macro files and compares the regenerated macros, tables and
  rendered draft with the committed ones; generator inputs all go through `load()` and are checked against the manifest
  by a test; fail-closed tests now perturb values, not only delete keys.

## Unreleased — referee-round items, seed replicates, RMDC26 section merged (2026-09-12, afternoon)

- **Referee round on our own simulator** (`validation/referee_round_local.py` -> `validation/referee_round.json`;
  paper §limits and §cascade via `\bmlRef*`):
  - Detectability floor 0.01 / 0.02 / 0.05 mag, label side (shipped model, frozen threshold): completeness
    0.808 / 0.891 / 0.948, purity 0.961 / 0.913 / 0.766, prevalence 6.5 / 5.5 / 4.2%, AP 0.942 / 0.957 / 0.934.
    The three arms are independent draws (the keep draw desynchronises the random stream), not the same events.
  - Colour calibration: the shipped model on the audited F087/F213 photometry of the same events keeps its anomaly
    channel (AP 0.956 vs 0.957) and loses 2 points of F1 on periodic and eruptive variables; short fine-tunes on
    each calibration keep AP at 0.957-0.958 and tie periodic-variable F1 to the training calibration (0.955 vs 0.918).
  - Mixed-class sequential scan (all six classes, all bands): no Flat or variable-star alert; 0.77 alerts per 1,000
    events per day, 89% anomalies at the simulated 5.5% prevalence (58% at 1%); 29 of 32 single-lens alerts are
    binaries below the detectability policy; timing matches the in-house three-band scan.
  - Seed sweep of the shipped model: not possible (training set on S3; AWS unavailable).
- **Seed replicates of the recommended recipe** (`fspl5s_seasons_g08_s2`, `_s3`): 1S2L recall at the 5.2% budget
  0.349-0.391 per event, 0.385-0.407 weighted; the paired event-bootstrap intervals exclude zero between seeds, so
  the paper now interprets no difference between single runs smaller than the seed range. The combined arm's lead
  over `ft_fspl5s_g08.pt` does not survive (seeds 2-3 trail it per event, tie it weighted); the extra training, the
  measured pauses and the physics' false-alarm drop at large rho/|u0| do. `binml-gapaware.pt` stays seed 1, stated
  as the best of three.
- **Truth relabelling corrected for truncation** (audit findings 8-10): full-season-fit residuals taught 13.7% of
  truncated binary presentations NonPSPL before the onset (15.2% against the refit reference of the third verification); truncation now takes only the floors from the truth and
  keeps the onset (use `--onset-resolution-days 0.5`). Measured impact of the released labels in
  `validation/truth_relabel_impact.json` and the paper (§cascade, §limits). No released checkpoint used truth relabelling.
- Abstract block [A] merged into `paper.tex`; the paper now carries every RMDC26 block.
- `paper/make_gulls_macros.py`: referee and seed blocks, fail-closed direction checks for the sentences that
  describe them, `--list-inputs` for manifest hashing; seed replicates are not counted as candidates.

## Unreleased — second verification pass and the recommended checkpoint (2026-09-12)

A second pass (five verifiers, five adversarial checkers, a completeness critic) on the corrected state found
74 issues, none refuted; record in `docs/VERIFICATION_2026-09-12.md`. Changes that alter conclusions:
- **Rate weighting.** RMDC26 numbers had been per simulated event without saying so; the sample over-represents
  short single lenses (38% with t_E < 3 d, 16% weighted). `gulls_summary_tables.py` now writes rate-weighted
  blocks, each checkpoint at its own threshold, and paired bootstrap differences; the draft gives both weightings.
- **Recommended gap-aware checkpoint: `ft_fspl5s_seasons_g08.pt` at 0.956** (finite-source single lenses +
  measured-season pauses). It matches or beats `ft_fspl5s_g08.pt` at every matched budget per event and leads it
  rate-weighted (+0.017 to +0.024 in 1S2L recall), and is better on our own held-out. Operating point, labels,
  floor, sub-day, occupancy, between-season and cascade analyses rerun for it (earlier results reproduced exactly).
- Withdrawn: "the combined arm's extra false alarms are in the largest-ratio bins" (they are in every bin) and
  "the finite-source physics accounted for a large part of the residual" (the extra training did most of it).
- `paper/make_gulls_macros.py` fail-closed for real (tested), single rounding from counts, every scored
  checkpoint counted; draft typed numbers limited to method constants.
- Code: cascade/between-season caches bound to weight hashes, between-season curves stored for local rescoring,
  onset scan tests the window end, generation settings carried into cache and memmap metadata, legacy
  ESPLMag2 regimes (`fspl5s_legacy`), fine-tune runner no longer reuses outputs of a replaced checkpoint,
  `gulls_summary_tables.py --scores` covers weights and t_E at full precision.

## Unreleased — verification of the revision work (2026-09-11)

A workflow of eight independent verifiers and two adversarial checkers re-derived every number,
claim and code path going into the revision: 182 findings in 61 issues, none refuted; record and
dispositions in `docs/VERIFICATION_2026-09-11.md`. Changes that alter conclusions:
- RMDC26's pause schedule differs between seasons: `validation/gulls/rmdc26_schedule.py` measures it;
  `pipeline.train --gap-schedule rmdc26_seasons`; the first-season mask is kept as `rmdc26` and labelled
  as such; calibration redone under the measured seasons on one pool, with the full-pool threshold and
  its slice spread (`fspl5s`: 0.957 → RMDC26 FA 1.6%).
- Colour-band gain and "the schedule hides half the planets" readings withdrawn; "0.56 recall ceiling"
  and "no anomaly a survey could claim" corrected.
- Onset: default back to the legacy 7.2-d grid (the 2026-09-11 morning change flipped a released default
  and had two off-by-one errors); opt-in full-grid scan via `run_shard --onset-resolution-days`; shard
  attributes record onset / noise / regime settings. Audit finding 9 re-opened.
- Finite-source magnification: VBBinaryLensing ESPLMag (ESPLMag2 had 4-8 mmag hand-off steps; legacy
  flag `PSPL_ESPL_LEGACY`); truncation amplitude uses it for finite-source single lenses.
- Relabelling of RMDC26: full precision, single lenses per the training rule, the documented sample
  enforced, a multi-start refit when the peak is outside the window, every derived number in the artifact.
- New artifacts: `rmdc26_dataset_facts.json`, `occupancy_sensitivity.json`, per-event
  `rmdc26_scores.csv.gz`; `validation/gulls/PROVENANCE.md` for artifacts made before generation-time
  provenance was recorded. Controls: `pspl5s_ctrl` (point-source continued training: most of the
  gain over g08e12 is the extra training; the physics lowers false alarms further in every rho/|u0| bin, most in
  rate in the largest ones, and its recall gain is not resolved when events are weighted by rate), the combined
  arm `fspl5s_seasons_g08` (smooth magnification + measured-season pauses; the recommended checkpoint since the
  2026-09-12 entry above) and the `fspl5s` recipe with the smooth magnification alone (`fspl5s_espl_g08`:
  indistinguishable from `fspl5s`, so ESPLMag2's steps did not shape the finite-source result).
- `binml.gulls.classify_event(mode=...)`; notebooks and `ft_g08e12.pt` committed; manuscript numbers
  only through `paper/make_gulls_macros.py` (fail-closed).

## Unreleased — follow-up experiments after the GULLS transfer (2026-09-10/11)

Generator / training (all opt-in or version-flagged; the released checkpoints and frozen artifacts are unchanged):
- `SurveyConfig.onset_resolution_days`: the recorded anomaly onset `t_anom` can be resolved on the 0.5 d grid the
  cascade evaluation uses. *Corrected by the 2026-09-11 verification (section above):* this entry first made 0.5 the
  default and its fine scan skipped grid points; the default is the legacy 7.2 d grid again and 0.5 is an opt-in
  full-grid first-detectable scan. Audit finding 9 measured: under a FIXED gap schedule the old grid put 2 of 10 onset
  values inside the blanked bins, making 20.1% of NonPSPL-labelled pool events (7.4% of generated binaries) eligible
  for the caustic-in-gap relabel (`validation/schedule_finetune_local.py`; corrected wording, 2026-09-11 verification). Released checkpoints were trained on the legacy grid (paper says so).
- `pipeline.train --gap-schedule rmdc26` (the FIRST season's seven pauses + 70.7 d season end; the other seasons differ,
  see `--gap-schedule rmdc26_seasons` in the entry above) and
  `--gap-relabel-anomaly off`; `SurveyConfig.noise_mult` / `bkg_mult` (defaults bit-identical); regimes
  `fspl5s_noisy{,_highmag}` (bkg_mult 7, measured on GULLS).
- Finite-source single lenses (`PSPL_FINITE_SOURCE`, VBBinaryLensing ESPL), regimes `fspl*`; round 3 `fspl5s`
  is the sidecar candidate (`validation/gulls/weights/ft_fspl5s_g08.pt`; threshold 0.957 under the measured
  per-season pauses; the 0.949 first-season calibration is superseded).

Validation (new scripts and artifacts under `validation/gulls/`):
- `detectability_relabel.py`: BinML's own label policy applied to GULLS from `true_flux_uJy`/`flux_err_uJy`
  (44-46% of GULLS planetary events carry no anomaly our policy would claim within one season; recall on those with
  one reported).
- `gulls_summary_tables.py` (matched-budget table, colour ablation), `calibrate_gapped_threshold.py`,
  `subday_summary.py` (sub-day t_E out-of-support row), `gulls_noise_vs_ours.py` (noise-model comparison),
  `validation/schedule_finetune_local.py`, `validation/fspl_finetune_local.py`.
- `binml.gulls` (new, extras `[roman]`): pinned-revision RMDC26 access, season finding, empirical baseline,
  `classify_event` (single season or per-season combiner); offline-tested. Example notebooks
  `examples/00_quickstart_synthetic.ipynb` (offline, executed in CI) and `examples/01_classify_roman_event.ipynb`
  (network). `validation/gulls/floor_sensitivity.py` (label-side floor sweep on GULLS) and
  `validation/gulls/cascade_gulls.py` (half-day cascade on GULLS: timing, alert burden, purity at stated prevalence).
- Results and what NOT to claim: `paper/REVISION.md` §1½ (experiment ledger).

## Unreleased — full-codebase audit fixes (2026-09-09)
Nine-agent audit of every Python file (record: `docs/AUDIT_2026-09-09.md`): 0 critical, 13 major,
64 minor; no submitted headline number wrong. Changes that touch reported numbers or their meaning:
- **Fitted-PSPL baseline rescored at full cadence** (`validation/baselines.py`): AP 0.261 → **0.545**.
  It had been scored on an 800-epoch thinning while its residual statistic is a running window in
  POINTS, so thinning integrated a 9× longer timescale and suppressed caustic-length residuals.
  All other baseline numbers are seed-identical. `canonical_numbers.json` / `MANIFEST.json` updated.
- **`binml.preprocess.bin_band` pools one value per Roman epoch before binning.** Bit-identical for
  on-grid input (all training/evaluation data); for denser input it now reproduces the training
  cache's representation instead of inflating min/max and frac with the sampling density. The GULLS
  full-population transfer (56,975 matched events) was re-scored with it: single-lens false alarms
  at threshold 35.6% → 11.7% (shipped → gap-aware; 11.6% after exact rounding, 2026-09-11), planetary recall at threshold 0.50 → 0.46.
  Raw-observation pooling gave 45.6% → 9.7% / 0.36; both are kept (`transfer_full_*_rawpool.json`).
- **Cadence comparison rerun with a held-out evaluation** (`validation/cadence_local.py`): same training
  shards, seeds and recipe as `modal_cadence.py` (which had scored each arm on ~80% training data),
  scored on 30,013 disjoint-seed events. AP 0.827 (15 min) vs 0.830 (12 min); max per-class F1
  delta 0.021. The training-pool scoring inflated AP by +0.019 / +0.005. `cadence_result.json`,
  `canonical_numbers.json`, `MANIFEST.json` and the paper paragraph updated.
- **OOR shard mix duplicate-key bug** (`run_shard.py`): the swept class got 500–1,000 events per
  shard, not 9,000. Recalls unbiased; the paper now quotes swept-class support (`\bmlStress*N`),
  e.g. wide-separation NonPSPL recall rests on n = 438.
- **F087 saturation physics corrected** in `photometry.py`, the paper and docs: a shorter exposure
  saturates *brighter*; equal-well F087 sits ~1.2 mag brighter than F146. `ROMAN_BANDS_AUDITED`
  F087 saturation 16.1 → 13.6 (derived; not used by the released model).
- `evaluate.py`: efficiency plane computed on test rows only (the committed artifact already was;
  regression test pins it); macro-F1 scores a never-predicted class 0 instead of dropping it;
  `false_anomaly_by_true_class` now buckets by true class; `--max-events` subsets every column.
- `plots.py`: PR-panel chance line is the weighted prevalence (0.056, as in the abstract), not 0.119.
- `binml classify` requires `--m-base` (or explicit `--estimate-baseline`); the fallback estimator's
  1.28σ faint bias on quiescent sources is documented. Legacy CLI import and `--flux` fixed.
- `make_macros.py` fails closed on NaN/inf and on a missing `figures_stats.json`; `bmlFtWorstRegress`
  is the largest F1 drop; McNemar discordant counts are macros, not prose. Three tautological tests
  replaced with real bounds; a manifest-pinned regression test added for the efficiency plane.
- Provenance: `modal_ablations.py` uses a `.done` marker and records trained epochs per arm (the
  reported cascade_off arm was verified at 10/10 epochs); `modal_gap_finetune.py` records
  hyper-parameters in its marker; `modal_gulls_transfer.py` pins the dataset revision.
- Still open (design decision, no reported metric affected): augmentation relabelling in
  `train.py` — `t_anom` quantised to 7.2 d, periodic amplitude proxy mis-scaled, dead Flat branch.
  Paper text and REVISION.md now describe what the g08e12 fine-tune actually ran with.

## Unreleased — gap sensitivity and first cross-simulator validation (2026-08-23)
- **Shipped checkpoint fails on Roman's planned F146 schedule.** The training grid is
  continuous; the planned GBTDS schedule pauses F146 ~6 h seven times per season. Inserting
  those gaps into in-distribution events drops PSPL recall 0.93 → 0.11 and Flat 1.00 → 0.08
  (`validation/gulls/gap_sensitivity.py`). Reproduced with no external data.
- **`pipeline/train.py --gap-aug`** blanks contiguous Roman-like runs in every band and
  relabels; `tests/test_gap_augmentation.py` pins the contract. A warm-start fine-tune
  (`validation/modal_gap_finetune.py`, checkpoint `validation/gulls/weights/ft_g08e12.pt`)
  restores held-out macro-F1 under the gapped schedule from 0.384 to 0.879.
- **First valid GULLS (RMDC26) transfer numbers.** `validation/gulls_transfer.py` now reads the
  172 GB obs table locally by contiguous `event_id` block (0.09 s/event), pins HF revision
  `a338d5ba` (RGES-PIT re-uploaded on 2026-08-18), measures the baseline empirically (the
  catalogue value is 0.471 mag off in this release), and accepts `--weights`. On 1,286 events
  the fine-tuned model cuts single-lens false alarms at the frozen threshold from 52% to 7%.
- Shipped weights unchanged; submitted numbers unaffected. Write-up plan: `paper/REVISION.md`.

## Unreleased — audit round 3 (streaming analysis corrected)
- **Cascade numbers rebuilt from one stored scan.** `validation/cascade_trace.py` records
  P(NonPSPL) at every 0.5 d cut for the frozen 1,000-event sample, under F146-only and
  all-three-band revealing, plus the anomaly-detectability mask on the same grid;
  `cascade_reduce.py` derives every reported statistic from it. Alert policy and onset definition
  are now knobs in the reduction rather than reasons to re-scan a different sample.
- **Two premature-alert numbers withdrawn.** The previously reported 31.3% scored alerts against
  an onset quantised to a 7.2 d grid (median inflation 3.7 d), and the 1.3 d "median lag after
  onset" was taken over all alerts including the premature ones. Corrected: 1.6% premature
  (1.0-2.6%) under a first-detectable onset, 4.5% under a strict persistent-detectable onset, with
  a median lag of +5.0 d. The old figure is retained in the artifact for comparison.
- **Labelling ablation de-confounded.** `pipeline/train.py` gained `--weight-labels`, because
  `compute_weights` derives class weights from whichever labels it is handed: the original arms
  differed in objective as well as in target. `validation/modal_labelling_ablation.py` pins the
  weights, scores every arm against both ontologies, and reports a paired bootstrap interval.
- **Equal-detection-rate cascade comparison run.** The experiment the paper named as its own
  clearest outstanding weakness. `validation/modal_cascade_matched.py` stores full probability
  traces for both ablation arms on a shared sample; `cascade_matched_reduce.py` sweeps each arm's
  threshold to a common within-season detection rate and compares premature rates there with a
  paired McNemar calculation. The augmented arm is premature less often at every matched detection
  rate from 0.50 to 0.90, but the thresholds and outcomes use the same 400 events and the arms use
  one training seed each. The conditional values are therefore descriptive rather than valid
  confirmatory p-values; the sign also reverses at 0.95. A causal benefit remains unestablished.
- **Prevalence thresholds moved off the final test set.** `validation/prevalence_thresholds.py`
  selects each threshold on the reserved validation rows and reports on the frozen test rows.
- **Provenance.** Future Modal runs key caches and checkpoints by a hash of the simulator/model source
  plus the full config, with a completion marker so an interrupted run cannot be silently reused
  as a finished one, and write a manifest recording git commit, pinned dependency versions, and
  per-artifact hashes. The published labelling-ablation artifact predates a verified execution of
  that mechanism; its source hash was repaired after the run.
- **Build.** `make_macros.py` reads the cascade, prevalence and ablation artifacts directly and
  fail-closed; a new CI job builds the paper end to end from `git archive HEAD`. Packaging moved
  to an SPDX license expression and explicit package-data configuration.

## 1.0.0 — 6-class multi-band model
- **New model: `pipeline/`.** 6 classes (Flat, PSPL, NonPSPL, PeriodicVar,
  LongPeriodVar, Eruptive) from 3 bands (F146/F087/F213). Conv-stem + transformer,
  505,479 params, replacing the 3-class CNN-GRU.
- **Detectability-conditioned labelling** — events labelled by what is observable, not by
  generator intent.
- **Partial-season cascade** — truncation labelling uses a truth-informed, noise-free per-binary anomaly-onset
  day (`t_anom`). Alert timing is measured event-level on a frozen 1,000-binary sample; see
  `validation/cascade_trace.py` and `validation/cascade_reduce.py`. The legacy 42%->9% premature
  flagging figure was untracked, could not be reproduced, and is withdrawn.
- **Distributed pipeline** — AWS spot generation/binning, in-region inference (`run_bineval`,
  `eval_shard`), stratified fine-tune mix (`mix_finetune`), stress-test aggregation (`agg_stress`).
- Headline (a 90,117-event threshold-calibration split plus a disjoint 360,472-event final-test
  split): completeness@purity 0.879, AP 0.952. The 14.9M-event stress suite comprises a 4.5M
  same-prior reproduction and 10.4M targeted out-of-distribution diagnostics; it is not a single
  population-validation set.
- Docs: new `docs/pipeline.md`; `architecture.md`, `evaluation.md`, `data_format.md`,
  `training.md`, and `README.md` updated to v5.


## 0.2.0
- Added detectability-conditioned evaluation (`binml.evaluate`): 3-class
  `classification_report`, `detectability_curve` (binary recall vs Δχ², indistinguishable
  fraction, detectable-only recall), and `evaluate_dataset` for compact HDF5 test sets.
- Added `Classifier.predict_arrays` for batched, GPU-vectorized prediction from stored grids.
- New CLI subcommand `binml evaluate`; added `binml --version`.
- Docs: "report binary performance the honest way" (detectability caveat).

## 0.1.0
- Initial release: `Classifier` (Flat / PSPL / Binary), real-survey preprocessing,
  OGLE/MOA/generic loaders + `fetch_ogle_ews`, probability-evolution, CLI, bundled weights
  (fine-tuned default + base).
