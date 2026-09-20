# Reference check and code self-check (2026-09-20)

A manual check after resubmission, done without agents: (1) every reference in `paper/refs.bib` read at its
published record and checked against the sentence it supports in `paper/paper.tex`; (2) one load-bearing
module, `pipeline/evaluate.py`, read end to end and tested against the frozen artifact; (3) a comparison of
this codebase against the classes of defect that an external review found in the author's published CAMELS
paper (`~/Desktop/Research/camels/audit/AUDIT.md`: a sample filter described but not applied, a feature built on
the wrong grouping, biased bootstrap machinery, latent bugs that never fired, docstring numbers that were wrong).

Nothing in the paper changes. Six bibliography entries are incomplete or stale (below, "Pending fixes");
four documentation statements in the repository disagree with the code or are misleading; none moves a number.

## 1. References (48 of 48 checked)

Sources: Crossref API for every DOI, arXiv abstract pages, ar5iv full text where the abstract did not carry
the claim, ADS or the publisher for entries without a DOI, and the live pages for the three web references.
"Bib" compares year, volume and pages with the published record; "Claim" is whether the source supports the
sentence that cites it.

| key | published record | bib | claim | what was read |
|---|---|---|---|---|
| Astromer2023 | A&A 670, A54 (2023) | ok | ok: transformer embedding for light curves | arXiv abstract |
| Bozza2010 | MNRAS 408, 2188 | ok | ok: contour-integration algorithm | arXiv abstract |
| Bozza2018 | MNRAS 479, 5157 | ok | ok; `ESPLMag` finite-source single-lens functions confirmed (Sec. 8 use) | ar5iv full text |
| CrispimRomao2025 | MNRAS 543, 351 | ok | ok: iForest on drip-fed synthetic LSST curves (ML, binary ML, subhalo/boson-star ML, RR Lyrae, constant); stage 1 separates signal from constant, stage 2 offline classifier | ar5iv full text, OUP |
| Dominik2007 | MNRAS 380, 792 | ok | ok: automated anomaly detector with immediate feedback (SIGNALMEN, unnamed in abstract) | arXiv abstract |
| Dominik2008 | AN 329, 248 | ok | ok: ARTEMiS expert system, SIGNALMEN alerts | arXiv abstract |
| Gaudi1998 | ApJ 506, 533 | **DOI missing** (10.1086/306256) | ok: binary-source events "masquerade as planetary events" | arXiv abstract, IOP |
| Gaudi2012 | ARA&A 50, 411 | ok | ok: review of microlensing surveys, cited among yield predictions | Crossref, Annual Reviews |
| Godines2019 | A&C 28, 100298 (2019; arXiv posting 2020) | ok | ok: random-forest classifier (LIA) | arXiv abstract |
| GouldLoeb1992 | ApJ 396, 104 | ok | ok: planets detectable as microlenses of bulge stars | ADS |
| Hlozek2020 | **published: ApJS 267, 25 (2023), 10.3847/1538-4365/accd6a**; bib cites the 2020 preprint | update | ok: "non-representative training set for a large photometric test set" | arXiv abstract, IOP |
| Husseiniova2021 | MNRAS 506, 2482 | ok | ok: decision-tree classifier on 2010-2019 VVV archive (completed data) | arXiv abstract |
| Johnson2020 | AJ 160, 123 | ok | ok: FFP abundance and mass function | arXiv abstract |
| Khakpash2021 | AJ 161, 132 | ok | ok: features for simulated Roman curves, "rapid, automated classification" as the stated purpose | arXiv abstract |
| Kim2016 | JKAS 49, 37 | ok | ok: three 1.6 m telescopes at CTIO, SAAO, SSO | Crossref, KoreaScience |
| Kim2018EventFinder | AJ 155, 76 | **pages missing** (renders "AJ, 155,") | ok: the paper's own term is the "completed event" event-finder | arXiv abstract |
| Kluter2025 | AJ 169, 317 | ok | ok: SynthPop population synthesis; the RMDC26 dataset card asks users to cite it | arXiv abstract, HF card |
| Mao1991 | ApJ 374, L37 | ok | ok | ADS |
| Moller2021 | MNRAS 501, 3272 (online Nov 2020; ADS year 2021) | ok | ok: microlensing module "using LIA based on Godines et al. 2019", run on live alerts; no binary or anomaly class anywhere in the text | ar5iv full text |
| Mroz2019 | ApJS 244, 29 | ok | ok: mean t_E 22 d in the central bins (rising to 32 d outward; 34 d at \|b\| <= 1.5 deg); our log-normal (median 16.3 d, sigma 0.35 dex) has mean 22.6 d | ar5iv full text |
| Mroz2020 | Acta Astron. 70, 169; DOI 10.32023/0001-5237/70.3.1 resolves (not registered with Crossref) | ok | ok: neural-network classifiers for single and binary events, OGLE data | arXiv abstract, doi.org |
| Muthukrishna2019 | PASP 131, 118002 | ok | ok: RAPID early classification | arXiv abstract |
| Paczynski1986 | ApJ 304, 1 | ok | ok | ADS |
| Penny2013 | MNRAS 434, 2 | ok | ok: Penny et al. 2019 states gulls is the simulator of Penny et al. 2013 (then MaBmuLS), renamed | arXiv abstract, ar5iv of Penny 2019 |
| Penny2019 | ApJS 241, 3 | ok | ok: ~1400 bound planets predicted | arXiv abstract |
| RGESPIT2026 | HF dataset `RGES-PIT/MachineLearning`, CC0-1.0 | ok | ok: RMDC26; classes 1S1L, 1S2L, 2S2L as described; card asks to cite GULLS and SynthPop | dataset card |
| ROTAC2025 | arXiv:2505.10574 | ok | partly on its own: 12.1 min F146 cadence and six 72-day high-cadence plus four low-cadence seasons are in the report; it gives **67 s** exposures and colour cadence "12 hours or faster". The 66 s and the 6-hour staggered colour cycle rest on STScI2025GBTDS, cited alongside, so the two sentences are covered | ar5iv full text |
| RomanTechInfo | file exists: `roman_technical_information/data/WideFieldInstrument/Imaging/ZeroPoints/Roman_zeropoints_20240301.ecsv` | **path in the note is stale** (repository moved `data/` under `roman_technical_information/`; the old path returns 404) | ok | GitHub tree API, raw file |
| STScI2025GBTDS | RDox page, published 2025-10-31, last updated 2026-08-24 | ok | ok: "six-tile cycle in F146 takes 12.1 min"; F087/F213 "each filter repeats on a 6-hour cycle, staggered"; 66 s; 6 high-cadence + 4 low-cadence seasons of ~72 d | live page |
| Soszynski2013 | Acta Astron. 63, 21 | ok | ok: Miras, semiregulars, OSARGs in the bulge | arXiv abstract |
| Soszynski2014 | Acta Astron. 64, 177 | ok | ok: 38,257 RR Lyrae in the bulge fields | arXiv abstract, ADS |
| Spergel2015 | arXiv:1503.03757 | ok | ok: WFIRST-AFTA report with the microlensing programme | arXiv abstract |
| Sumi2023 | AJ 166, 108 | ok | ok: FFP mass function from MOA-II; also predicts the Roman FFP yield | arXiv abstract |
| Suzuki2016 | ApJ 833, 145 | ok | ok: mass-ratio function with a break near q ~ 1e-4; cold Neptunes beyond the snow line | arXiv abstract |
| Tsapras2009 | AN 330, 4 | ok | ok: RoboNet-II with ARTEMiS anomaly detection | arXiv abstract |
| Tsapras2016 | MNRAS 457, 1320 | ok | ok: detection efficiency on a grid of ten mass ratios (1e-5 to 1e-2) and planet positions, later converted to physical units | ar5iv full text |
| Udalski2003 | Acta Astron. 53, 291 | ok | ok: EWS "designed to detect on-going microlensing events" | arXiv abstract |
| Vaswani2017 | NeurIPS 30 | ok | ok | arXiv abstract |
| Wilson2023 | ApJS 269, 5 | ok | ok: 60,000-200,000 transiting planets | arXiv abstract |
| Wyrzykowski2015 | ApJS 216, 12 | **pages missing** (renders "ApJS, 216,") | ok: "3718 unique events", "150 million objects", 2001-2009 data | arXiv abstract |
| Zang2022 | MNRAS 515, 928 | ok | ok: AnomalyFinder fits anomalies in the residuals to a PSPL model; the method is defined in paper I (Zang et al. 2021, AJ 162, 163), which could be added | arXiv abstract, later papers of the series |
| Zhang2021 | AJ 161, 262 | **pages missing** (renders "AJ, 161,") | ok: real-time amortised inference; Sec. III.1 priors q ~ LogU(1e-6, 1), s ~ LogU(0.2, 5), u0 ~ U(0, 2), rho ~ LogU(1e-4, 1e-2), f_s ~ LogU(0.1, 1), t0 ~ U(0, 72) d, matching Table 1 | ar5iv full text |
| astropy2022 | ApJ 935, 167 | ok | software | arXiv abstract |
| harris2020 | Nature 585, 357 | ok | software | arXiv abstract |
| hunter2007 | CiSE 9, 90 | ok | software | Crossref |
| paszke2019 | NeurIPS 32 | ok | software | arXiv abstract |
| scikit-learn | JMLR 12, 2825 | ok | software | JMLR |
| virtanen2020 | Nat. Methods 17, 261 | ok | software | arXiv abstract |

### Pending fixes (bibliography only; the author decides when)

1. `Wyrzykowski2015`: add `pages = {12}`.
2. `Kim2018EventFinder`: add `pages = {76}`.
3. `Zhang2021`: add `pages = {262}`.
4. `Gaudi1998`: add `doi = {10.1086/306256}`.
5. `Hlozek2020`: cite the published version, ApJS 267, 25 (2023), doi 10.3847/1538-4365/accd6a.
6. `RomanTechInfo`: path `roman_technical_information/data/WideFieldInstrument/Imaging/ZeroPoints/Roman_zeropoints_20240301.ecsv`.
7. Optional: cite Zang et al. 2021 (AJ 162, 163) with Zang2022 for the AnomalyFinder method.

## 2. `pipeline/evaluate.py` read end to end (466 lines) and tested against `paper/results/`

Verified on the frozen artifact:
- `test_idx.npy` equals `rng(7).permutation(450589)[90117:]`, as the manifest says; 360,472 unique rows, disjoint
  from the 90,117 validation rows on which the threshold was chosen.
- The file's own functions on the committed logits reproduce the metrics file: threshold 0.9042, completeness
  0.879, purity 0.9079, average precision 0.9515.
- Every NonPSPL-labelled row was generated as NonPSPL, with dchi2_anomaly >= 160.04 and dchi2_event >= 503.7;
  the lowest dchi2 bin for true anomalies is empty.
- All 89,685 subsampled rows (keep_prob 0.15) are generated binaries relabelled PSPL (78,157) or Flat (11,528);
  3,034 of them have dchi2_anomaly >= 160 (amplitude-floor vetoes; the amplitude is not stored). No other class is
  subsampled, so the weighting only touches Flat and PSPL rows.
- Quoted intervals reproduce from first principles: Wilson 16/1000 -> [1.0, 2.6]%, 45/1000 -> [3.4, 6.0]%;
  binomial 1.96 sigma on 42,805 anomalies -> [0.876, 0.882], the paper's bootstrap interval.
- The `efficiency_plane` blanking gate (`n_eff < 30`) is a sum of weights, not an effective sample size; every
  populated cell has >= 1,016 raw events (Kish n_eff >= 1,008), so the gate never bites.

Found:
- **Naming.** "Effective events" in the Figure 9 caption and the `n_eff` field mean a sum of weights; in Sec. 8
  "effective sample size" means Kish n_eff. Two meanings for one phrase; no numeric effect.
- **Latent.** No assertion that the label columns have the memmap's `n_events` rows; a cache paired with columns
  from another version would misalign silently (the CAMELS F1 pattern). Has not fired: the artifact is
  self-consistent as above.
- Threshold choice takes the deepest validation index still meeting the purity target, which leans on noise in
  the validation ranking; it is frozen before it touches the test rows, where purity came out above target.

## 3. Documentation statements that disagree with the code (found in the CAMELS-class comparison)

| where | statement | what the code and artifact do | status |
|---|---|---|---|
| `docs/leakage_audit.md` l.21, `docs/glossary.md` l.58 | keep_prob reweighting "never" applied to recall | Table 3 recall is weighted (Flat 0.963 -> 0.960, PSPL 0.944 -> 0.936); the paper caption is right, the docs are wrong. The paper's Sec. 5 sentence stays true because NonPSPL rows all have weight 1 | open |
| `pipeline/mix_finetune.py` docstring l.20-22 | base training used 676,191 events; test set 443,030 | 676,191 is the pool before the 80/20 split (0.8 x 676,191 = 540,952, the paper's base-training count); 443,030 is an older pool size (now 450,589) | open |
| `paper/paper.tex` Fig. 9 caption, `pipeline/evaluate.py` | "effective events" | sum of weights (Sec. 2 above) | open |
| `pipeline/evaluate.py` | column/memmap length | no assertion (Sec. 2 above) | open |

## 4. Mapping the CAMELS review's defect classes onto this codebase

| CAMELS finding | BinML equivalent | status |
|---|---|---|
| F8: text describes a sample filter the code never applied | stage-6 training pool asserted as shards 0-89, not enforced (audit 2026-09-09 item 14) | open; only checkable from the S3 listing (AWS unavailable); indirect evidence only |
| F3: feature built on the wrong grouping | `bin_band` pooled raw observations (audit items 29, 30) | found and fixed before resubmission; it moved a number (45.6% -> 35.6% false alarms) |
| F2, F4: bootstrap machinery biased | cascade and completeness intervals | reproduced from textbook formulas (Sec. 2) |
| latent bugs that never fired | 38 audit items left "noted", 35 outside paper-facing code | known, unfixed, listed in `docs/AUDIT_2026-09-09.md` |
| docstring numbers wrong | Sec. 3 above | found here |

Checked today and clean: the RMDC26 score table has one row per event (56,975; 33,353 / 11,388 / 12,234 by class),
so no many-to-one join; the frozen arrays are aligned and the split is in the manifest.
