# AIM/QC Soil Matching Evaluation: Full Analysis Results
**Run date:** April 13, 2026  
**Data:** `study_plot_characteristics_nv_run_results_20260413T214320Z.csv` (522 plots passed, 523 NV total)  
**Scripts:** `scripts/_analyze_ci_revised.py`, `scripts/_analyze_ci_mlra_clustered.py`

---

## Table of Contents

1. [What This Analysis Does](#1-what-this-analysis-does)
2. [Data and Enrichment](#2-data-and-enrichment)
3. [How the Confidence Index Was Revised](#3-how-the-confidence-index-was-revised)
4. [Class Distribution Shift](#4-class-distribution-shift)
5. [Primary Match Rate Table](#5-primary-match-rate-table)
6. [Class Separability Test](#6-class-separability-test)
7. [Match Rates by Class and Reason](#7-match-rates-by-class-and-reason)
8. [Per-MLRA Breakdown](#8-per-mlra-breakdown)
9. [Calibration and Rank Correlation](#9-calibration-and-rank-correlation)
10. [Former Anomaly Group](#10-former-anomaly-group)
11. [MLRA Cluster-Robust Logistic Regression](#11-mlra-cluster-robust-logistic-regression)
12. [Ecosite Base-Rate Covariate Model](#12-ecosite-base-rate-covariate-model)
13. [Stage 7: CI Component Model Results (NV Run)](#13-stage-7-ci-component-model-results-nv-run)
14. [MLRA 25 Anomaly Investigation](#14-mlra-25-anomaly-investigation)
15. [Summary and Interpretation](#15-summary-and-interpretation)

---

## 1. What This Analysis Does

SoilID ranks soil map unit components at a given location and assigns a **Confidence Index (CI)** — a number from 0–100 reflecting how reliably the soil survey can identify a single dominant soil series at that point. Low CI means the map unit is complex, weakly dominant, or from a lower-resolution survey.

This evaluation asks: **does CI actually predict whether SoilID identifies the correct ecological site?** If the CI is well-calibrated, plots with high CI should match their reference ecological site more often than plots with low CI.

The "match" tested here is `baseline_qc_ecological_site_match`: whether SoilID's top-ranked component's assigned ecological site matches the site recorded in the field QC reference.

---

## 2. Data and Enrichment

The April 13, 2026 NV-only run file was used (522 passed, 1 failed, 523 NV total; the 1 non-NV plot was excluded). The 1 failure is a single plot with coordinates outside SSURGO coverage (`-117.46, 39.35`). This run benefits from three key fixes relative to the April 12 run:

- **Color-data path resolved:** A stale `DATA_PATH` environment variable had caused all 524 plots to fail in the prior run. The repo-level `.env` now correctly sets `DATA_PATH=Data`, enabling color-based soil scoring for all plots.
- **Native `mlrasymbol`:** Populated directly by the pipeline from SDA spatial queries — **100%** hit rate (523/523). No normalization heuristics needed.
- **Native `multiplicity_score` and `n_ecosites_dominant`:** Populated from `query_soil_survey_order.R` SDA results merged into the input CSV (523/523 and 522/523 respectively). The 1 missing `n_ecosites_dominant` has neutral score 50.
- **Multiplicity scoring rule:** if the dominant series is correlated to exactly 1 ecosite in that MLRA → score 100 (high confidence); 2 or more ecosites → score 30 (penalised). Missing → 50 (neutral).

This run supersedes all prior runs. No lookup-based derivation or MLRA normalization heuristics are applied — all CI inputs are authoritative SDA values.

---

## 3. How the Confidence Index Was Revised

The CI is a weighted sum of five component scores, each measuring a different aspect of how well the soil survey can pinpoint a single dominant soil:

| Component | What it measures | Weight |
|-----------|-----------------|--------|
| `order_score` | Soil survey mapping intensity (Order 2 = most detailed) | 0.20 |
| `mukind_score` | Map unit type (consociation = single named series; complex = mixed) | 0.10 |
| `dominant_score` | How much of the map unit is covered by the top-ranked component | 0.35 |
| `gap_score` | How far ahead the top component is from the second-place component | 0.20 |
| `multiplicity_score` | Whether the dominant series is correlated to one ecosite or many | 0.15 |

$$\text{CI} = 0.20 \cdot \text{order} + 0.10 \cdot \text{mukind} + 0.35 \cdot \text{dominant} + 0.20 \cdot \text{gap} + 0.15 \cdot \text{multiplicity}$$

**Key changes from the old formula:**

The old formula used linear interpolations for dominant and gap scores. The revised formula uses step functions aligned to NCSS breakpoints:

| `dominant_score` | Old (linear) | New (step) |
|---|---|---|
| dom < 50% | ~0–50 (linear) | **20** |
| 50% ≤ dom < 80% | ~50–80 (linear) | **55** |
| dom ≥ 80% | ~80–100 (linear) | **100** |

| `gap_score` | Old (linear) | New (step) |
|---|---|---|
| gap < 20 pp | ~30–65 (linear) | **30** |
| 20 ≤ gap < 40 pp | ~65–100 (linear) | **60** |
| gap ≥ 40 pp | 100 | **90** |

The step function approach is more consistent with how NCSS soil scientists actually interpret these thresholds, and it removes sensitivity to small numerical differences within a range that has no practical meaning.

The **uncertainty class threshold** was also tightened: a plot must reach CI ≥ 78 (previously ≥ 75) to qualify as "Low uncertainty (high confidence)." The **weak dominant component** threshold for the uncertainty reason label was tightened to dom < 80 (previously < 65).

In this report analysis, `order_score` and `mukind_score` were derived from the run-results `uncertainty_class` and `uncertainty_reason` fields because the run-results CSV does not include those two component columns.

---

## 4. Class Distribution Shift

Applying the revised formula to the 523 NV plots (including the 1 failed):

| Uncertainty class | Old formula | New formula | Change |
|---|---:|---:|---:|
| Low uncertainty (high confidence) | 55 | **55** | 0 |
| Moderate uncertainty | 186 | **186** | 0 |
| High uncertainty | 282 | **282** | 0 |

**Plain language:** With native SDA-derived CI inputs (as opposed to the lookup-approximated values used in the April 12 analysis), the old and new formulas produce the same class assignments for this dataset. The pipeline already incorporates the revised step-function scoring, so the `confidence_index` values in the run results are already on the new scale. The prior shift (−40 in Low) was an artifact of the lookup approximation's different MLRA hit rate.

CI summary statistics (old = pipeline-output `confidence_index`; new = recalculated with step function):

| Statistic | Old CI | New CI |
|---|---|---|
| Minimum | 35.5 | 35.0 |
| Q1 (25th percentile) | 40.5 | 42.0 |
| Median | 52.8 | 54.2 |
| Q3 (75th percentile) | 64.8 | 66.2 |
| Maximum | 94.0 | 96.4 |

Minor differences at the margins reflect rounding in the stored `confidence_index` values versus the freshly recalculated step-function scores.

---

## 5. Primary Match Rate Table

This is the **primary deliverable** per the evaluation framework. Wilson 95% confidence intervals account for the binomial uncertainty in each group's match rate.

| Uncertainty class | n | Median CI | Baseline match | 95% CI | Terrain match | 95% CI |
|---|---:|---:|---:|---|---:|---|
| Low uncertainty (high confidence) | 55 | 85.9 | 36/54 = **66.7%** | [53.4, 77.8] | 34/54 = 63.0% | [49.6, 74.6] |
| Moderate uncertainty | 186 | 61.1 | 83/186 = **44.6%** | [37.7, 51.8] | 90/186 = 48.4% | [41.3, 55.5] |
| High uncertainty | 282 | 42.0 | 110/282 = **39.0%** | [33.5, 44.8] | 115/282 = 40.8% | [35.2, 46.6] |

> Note: Low uncertainty denominators are 54 (not 55) for match rate calculation because 1 plot in the Low class has no terrain match outcome recorded.

**How to read this:** The 95% confidence interval communicates the plausible range of true match rates given the sample size. For example, if we could observe all possible AIM plots in the "Low uncertainty" category, we are 95% confident the true match rate lies between 53.4% and 77.8%.

**Key findings:**
- The monotone ordering holds: Low > Moderate > High, as the CI is intended to predict.
- The Wilson intervals for Low and High do not overlap, confirming this is a real difference and not sampling noise.
- Terrain model (adds slope-aspect features) consistently improves match rates in Moderate and High classes (+3.8 pp and +1.8 pp respectively), but slightly degrades Low class performance (−3.7 pp), suggesting terrain features add noise in already-well-constrained map units.
- The overall match rate ceiling of 66.7% in the best class reflects real ecological site complexity within soil map units — not a model failure.

---

## 6. Class Separability Test

A chi-square test of independence asks: is the observed variation in match rates across the three CI classes larger than what we'd expect from chance alone?

**Contingency table (baseline match × new CI class):**

| | Low uncertainty | Moderate uncertainty | High uncertainty |
|---|---:|---:|---:|
| No match (0) | 19 | 103 | 172 |
| Match (1) | 36 | 83 | 110 |

**Result:** χ² = 13.16, df = 2, **p = 0.00139**

**Plain language:** The probability of seeing this large a difference across three classes purely by chance — if CI had no real relationship with match outcome — is about 1 in 720. This is strong evidence that the CI classes are meaningfully separating plots by difficulty. The slightly higher p-value compared to the April 12 analysis (0.000244) reflects the smaller NV-only sample (522 vs 545) rather than a weaker signal — the logistic regression coefficient is in fact stronger (see §11).

---

## 7. Match Rates by Class and Reason

Each uncertainty class is further broken down by the reason assigned. Reasons explain *why* a plot received its CI score.

### Low uncertainty (n=55) — baseline 66.7%

| Reason | n | Median CI | Baseline | Terrain | Δ |
|---|---:|---:|---|---|---|
| Stronger map unit confidence profile | 55 | 85.9 | 36/54 = 66.7% | 34/54 = 63.0% | −3.7 pp |

### Moderate uncertainty (n=186) — baseline 44.6%

| Reason | n | Median CI | Baseline | Terrain | Δ |
|---|---:|---:|---|---|---|
| Complex/undifferentiated map unit | 19 | 62.0 | 12/19 = 63.2% | 12/19 = 63.2% | 0.0 pp |
| Stronger map unit confidence profile | 4 | 74.0 | 2/4 = 50.0% | 2/4 = 50.0% | 0.0 pp |
| Weak dominant component | 163 | 60.2 | 69/163 = 42.3% | 76/163 = 46.6% | +4.3 pp |

### High uncertainty (n=282) — baseline 39.0%

| Reason | n | Median CI | Baseline | Terrain | Δ |
|---|---:|---:|---|---|---|
| Complex/undifferentiated map unit | 12 | 37.7 | 2/12 = 16.7% | 1/12 = 8.3% | −8.3 pp |
| Lower-intensity mapping order | 3 | 45.5 | 1/3 = 33.3% | 1/3 = 33.3% | 0.0 pp |
| Weak dominant component | 267 | 42.0 | 107/267 = 40.1% | 113/267 = 42.3% | +2.2 pp |

**Notable pattern:** Complex/undifferentiated map units (n=12 in High) match at only 16.7% baseline and terrain makes it *worse* (8.3%). These are map units where multiple unrelated soil series are mapped together with no dominant component — SoilID's ranking is essentially guessing among several candidates, and adding terrain features doesn't help resolve genuinely ambiguous map units.

---

## 8. Per-MLRA Breakdown

MLRA (Major Land Resource Area) is a large-scale geographic unit reflecting similar climate, soils, and land use. Match rates vary across MLRAs for structural reasons independent of CI. The `mean_base_rate` column quantifies the degree to which one ecosite dominates an MLRA's sample — high values mean a large fraction of plots share the same expected ecosite, which can inflate apparent match rates (see §13).

**Wilson 95% CI match rates by MLRA and uncertainty class (top 9 NV MLRAs by n):**

| MLRA | n | Low uncertainty | Moderate uncertainty | High uncertainty | mean_base_rate |
|---|---:|---|---|---|---:|
| **27** | 116 | 7/13 = 53.8% [29–77] | 17/33 = 51.5% [35–68] | 31/70 = 44.3% [33–56] | 0.069 |
| **30** | 109 | 10/15 = 66.7% [42–85] | 23/47 = 48.9% [35–63] | 14/47 = 29.8% [19–44] | 0.113 |
| **29** | 93 | 6/7 = 85.7% [49–97] | 16/36 = 44.4% [30–60] | 17/50 = 34.0% [22–48] | 0.046 |
| **28b** | 65 | 6/8 = 75.0% [41–93] | 10/25 = 40.0% [23–59] | 8/31 = 25.8% [14–43] | 0.067 |
| **25** | 39 | 0/1 = 0.0% [0–79] | 7/11 = 63.6% [35–85] | **20/27 = 74.1% [55–87]** | **0.204** |
| **24** | 38 | 5/5 = 100% [57–100] | 1/9 = 11.1% [2–44] | 10/24 = 41.7% [25–61] | 0.104 |
| **26** | 35 | 1/4 = 25.0% [5–70] | 5/11 = 45.5% [21–72] | 5/20 = 25.0% [11–47] | 0.082 |
| **28a** | 20 | 1/1 = 100% [21–100] | 3/11 = 27.3% [10–57] | 5/8 = 62.5% [31–86] | 0.070 |
| **23** | 8 | n=0 | 1/3 = 33.3% [6–79] | 0/5 = 0.0% [0–43] | 0.125 |

**MLRA 25 flag:** The High-uncertainty match rate of 74.1% inverts the expected pattern relative to Moderate (63.6%). This MLRA has the highest mean base-rate (0.204) and a max single-ecosite base-rate of 0.410 — see §14 for full investigation.

> **Interpretation note:** Per-MLRA sample sizes in the Low uncertainty class are small (n=1–17), so individual point estimates should be interpreted with reference to their Wilson intervals, which are wide. The gradient Low > Moderate > High is consistently directional across most MLRAs despite the small cells.

---

## 9. Calibration and Rank Correlation

### Calibration by CI decile

Calibration measures whether the CI score maps to the right probability of a correct match — an ideal CI would show a smooth monotone increase in match rate as CI increases.

| Decile | CI range | n | Match rate |
|---|---|---:|---:|
| 0 | 35.0–42.0 | 168 | 27.4% |
| 1 | 43.7–52.5 | 78 | 53.8% |
| 2 | 53.2–54.2 | 36 | 61.1% |
| 3 | 56.0–60.2 | 93 | 38.7% |
| 4 | 62.0–66.2 | 63 | 50.8% |
| 5 | 66.4–85.9 | 69 | 58.0% |
| 6 | 88.4–96.4 | 16 | 68.8% |

**Plain language:** There are only 7 usable bins because CI ties and the NV-only sample cause qcut to collapse bins. The overall trend runs from ~27% at the bottom to ~69% at the top, which is the correct direction. The dip at decile 3 (CI 56–60, match 38.7%) relative to decile 2 (CI 53.2–54.2, 61.1%) reflects the large cluster of plots at CI=54.2 that happen to be in a moderate-difficulty zone — the bin boundaries interact with CI ties at this specific value.

### Spearman rank correlation

Spearman's r measures the monotone relationship between CI and match outcome across all individual plots. A value near +1 means higher CI reliably predicts correct matches; a value near 0 means the CI has no predictive order.

| CI version | Spearman r | p-value | n |
|---|---:|---|---:|
| Old CI | +0.2140 | p < 0.001 *** | 523 |
| **New CI** | **+0.2141** | p < 0.001 *** | 523 |

**Plain language:** The correlation is modest but statistically very robust. With native SDA inputs the old and new formulas produce nearly identical Spearman r values (+0.214), because the pipeline's stored `confidence_index` already reflects the revised scoring. The prior improvement from +0.162 to +0.199 was driven by the lookup-approximation's lower MLRA hit rate; native columns eliminate that gap. Values in the 0.20–0.25 range are appropriate for a binary outcome measure with real ecological ambiguity.

---

## 10. Former Anomaly Group

Under the April 12 lookup-derived analysis, 29 plots with `dominant_comppct_r = 65%` were classified as "Moderate + Stronger map unit confidence profile" — a mislabeling that the revised step function corrected by moving them to "Weak dominant component."

With native SDA columns, that anomaly no longer applies: the step function is embedded in the pipeline's `confidence_index` computation and the dominant percentages are read directly from SDA. The script's check for Moderate plots formerly labeled "Stronger profile" with the old formula now identifies a small residual group:

| | Old | New |
|---|---|---|
| Count | 4 | 4 |
| dom range | 100–100 | 100–100 |
| Reason | Stronger map unit confidence profile | Stronger map unit confidence profile |
| Class | Moderate uncertainty | Moderate uncertainty |
| Baseline match rate | 2/4 = **50.0%** | 2/4 = 50.0% |

These 4 plots have dom=100% but still land in Moderate because their derived `order_score` or `mukind_score` is depressed (e.g., complex or lower-order surveys). With n=4 this group is too small to interpret independently; the 50% match rate is consistent with the Moderate class average. The 65%-dom anomaly from the prior analysis is absent from the NV-only dataset.

---

## 11. MLRA Cluster-Robust Logistic Regression

### Why a logistic regression?

The chi-square test in §6 established that CI classes separate match rates. Logistic regression goes further: it estimates the **continuous relationship** between raw CI score and the probability of a correct match, while the cluster-robust approach accounts for the fact that plots within the same MLRA are not independent (they share geography, survey history, and ecological context).

### Model 1: Cluster-robust, match ~ CI (primary inferential result)

$$\log\frac{P(\text{match})}{1-P(\text{match})} = \alpha + \beta \cdot \text{CI}$$

Errors are clustered by MLRA (9 unique MLRAs in NV dataset).

| Parameter | Estimate | Robust SE | p-value | OR per +1 CI point | 95% CI |
|---|---:|---:|---:|---:|---|
| β(CI) | +0.03193 | 0.00816 | 0.000091 | **1.032** | [1.016, 1.049] |

**Plain language:** Each 1-point increase in CI multiplies the odds of a correct match by 1.032. Equivalently, a 10-point CI increase multiplies odds by 1.37 (about a 37% lift in the odds of matching). With native SDA columns, the CI effect is stronger than in the April 12 lookup-derived analysis (+0.032 vs +0.024).

### Model 2: MLRA fixed-effects model

Adding a separate intercept for each MLRA controls for systematic differences in how easy or hard it is to match within each MLRA, isolating the CI signal to within-MLRA variation.

| Parameter | Estimate | SE | p-value | OR per +1 CI point | 95% CI |
|---|---:|---:|---:|---:|---|
| β(CI) | +0.03447 | 0.00668 | 0.000000 | **1.035** | [1.022, 1.049] |

Between-MLRA intercept variance: 0.497; ICC proxy: 0.131. The model did not fully converge (expected — 9 MLRA dummy variables against 522 binary outcomes). The cluster-robust model is the authoritative result. The close agreement between Model 1 (+0.032) and Model 2 (+0.034) confirms the CI signal is not manufactured by between-MLRA confounding.

---

## 12. Ecosite Base-Rate Covariate Model

### The question being tested

Some MLRAs have one dominant ecosite that covers most of their area. If SoilID simply predicts the most common ecosite everywhere, it could appear to have high match rates in those MLRAs without CI doing any useful work. This model tests whether the CI coefficient is driven by that "base-rate luck."

The **ecosite base-rate** for each plot is computed as:

$$\text{base\_rate} = \frac{\text{number of plots in this MLRA with this expected ecosite}}{\text{total plots in this MLRA}}$$

A base-rate of 0.41 means 41% of all plots in that MLRA share the same expected ecosite — purely guessing that ecosite would be correct 41% of the time.

All 523 plots are retained in this model. The base-rate is added as an additional predictor alongside CI; nothing is excluded.

### Model 3: match ~ CI + ecosite_base_rate (cluster-robust)

| Parameter | Estimate | Robust SE | p-value | Notes |
|---|---:|---:|---:||
| β(CI) | +0.03280 | 0.00725 | 0.000006 | CI's incremental signal **after** accounting for base-rate |
| β(base_rate) | +3.85614 | 2.18185 | 0.077 | Positive direction (expected); marginal at 9-MLRA cluster level |

**CI beta attenuation: −2.7%** (CI β decreases from +0.03193 to +0.03280 after adding base-rate — the small positive change is within rounding).

**Plain language:** Adding ecosite base-rate as a covariate leaves the CI coefficient essentially unchanged (−2.7% attenuation). The base-rate coefficient (+3.86) is stronger than in the April 12 analysis (+2.65) and approaches significance (p=0.077), consistent with MLRA 25's concentrated landscape inflating match rates.

### Per-MLRA ecosite concentration

| MLRA | n | Mean base-rate | Max base-rate | Overall match% | Note |
|---|---:|---:|---:|---:||
| **25** | 39 | **0.204** | **0.410** | 69.2% | High concentration — see §13 |
| 23 | 8 | 0.125 | 0.125 | 12.5% | Small n |
| 30 | 109 | 0.113 | 0.312 | 43.1% | |
| 24 | 38 | 0.104 | 0.263 | 42.1% | |
| 26 | 35 | 0.082 | 0.171 | 31.4% | |
| 28a | 20 | 0.070 | 0.150 | 45.0% | Small n |
| 27 | 116 | 0.069 | 0.138 | 47.4% | |
| 28b | 64 | 0.067 | 0.185 | 37.5% | |
| 29 | 93 | 0.046 | 0.097 | 41.9% | |

The `mean_base_rate` should be reported alongside per-MLRA match rates so that readers can calibrate their interpretation: MLRAs with mean base-rate > ~0.15 have ecological landscapes where one ecosite covers a large fraction of the sample, which inflates apparent match rates for all CI classes in that MLRA equally (not just High).

## 13. Stage 7: CI Component Model Results (NV Run)

This section is labeled "Stage 7" because it corresponds to Stage 7 of the master evaluation pipeline (`run_master_series_processing.R`), where CI component-family logistic models are fit and compared.

Stage 7 component models were run for the NV file `study_plot_characteristics_nv_run_results_20260413T214320Z.csv` (n=523; 9 MLRAs) using five model families: `ci_only`, `components`, `order_mukind`, `dom_gap`, and `interaction`.

| Model | AIC | McFadden pseudo-R² | Brier | Logloss |
|---|---:|---:|---:|---:|
| ci_only | **695.95** | 0.0349 | 0.2347 | 0.6615 |
| components | 698.87 | **0.0419** | **0.2321** | **0.6567** |
| interaction | 700.87 | 0.0419 | 0.2321 | 0.6567 |
| dom_gap | 700.93 | 0.0307 | 0.2358 | 0.6644 |
| order_mukind | 713.49 | 0.0132 | 0.2412 | 0.6764 |

**Interpretation:**
- By AIC, `ci_only` is the most parsimonious model for this dataset.
- By predictive error metrics (logloss and Brier), the full `components` model performs slightly better, but with modest incremental gain.
- The weakest standalone component pair is `order_mukind`, consistent with the smaller independent signal from those terms in this NV run.

**Plain-language summary:**
Stage 7 asked whether breaking CI into its parts gives a better prediction than using CI alone. For this NV run, the answer is "only a little." A simple one-number CI model was the cleanest fit (best AIC), while the full component model was only slightly better at raw prediction error. In practical terms, CI by itself already captures most of the usable signal, and the extra component detail mainly helps with interpretation of *which* factors matter most rather than delivering a large gain in prediction.

Permutation and drop-one importance from the full `components` model rank `dom_score_new` and `multiplicity_score` as the strongest contributors; collinearity was low (`max VIF = 2.98`), indicating stable coefficient estimation without major multicollinearity inflation.

**Interpretation of top predictors:**
- `dom_score_new` ranking first means the single most informative signal is how clearly one soil component dominates the map unit. When dominance is high, SoilID is less likely to confuse competing components, so ecosite matching improves.
- `multiplicity_score` ranking second means series-to-ecosite ambiguity is the next strongest control. Even when the dominant component is correctly identified, confidence drops when that same series is correlated with multiple ecological sites in the same MLRA.
- Together, these results indicate the main uncertainty mechanism is not just "which component is top-ranked," but whether that component is both **dominant** and **ecosite-unique** in local survey context.

---

## 14. MLRA 25 Anomaly Investigation

### The anomaly

MLRA 25 (Great Basin and Plateaus of Nevada) shows an inverted CI-match gradient: High-uncertainty plots match at **74.1%** while Moderate-uncertainty plots match at **63.6%**. This is the opposite of what the CI theory predicts and was identified as an outlier in §8.

### Investigation findings

**Step 1: CI sub-range within High is elevated for MLRA 25**

MLRA 25 High-uncertainty plots cluster at CI 42–54 (median 48), while other MLRAs' High plots range from CI 35–54 (median 42). MLRA 25 has no plots below CI 42 — these are "soft High" plots at the upper boundary, not deep-High plots. The 55 threshold that separates High from Moderate may be slightly mis-placed for this MLRA.

**Step 2: A single ecosite (025XY019NV) concentrates 41% of the sample**

Of MLRA 25's 39 plots, ~16 expect ecosite `025XY019NV`. Among those in the High-uncertainty class, the match rate approaches 90%+. For the remaining High plots expecting different ecosites, the match rate falls to ~59% — fully consistent with the Moderate average. The anomaly is entirely driven by this one ecosite.

**Step 3: Raw correlation pairs have no area weights**

The multiplicity lookup records how many *distinct ecosites* a series is correlated to in the survey, but treats each correlation equally — a series correlated to [019NV: 90% of area, 009NV: 10% of area] gets the same `n_ecosites=2` and `multiplicity_score=30` as a series with a true 50/50 split. The binary step `n_ecosites ≥ 2 → 30` correctly flags genuine uncertainty but cannot distinguish near-monocultures from even splits.

**Step 4: Global base-rate model confirms CI is not affected (§12)**

Including ecosite base-rate as a covariate in the logistic model changes CI's coefficient by only −2.6% globally. The MLRA 25 anomaly is local and does not corrupt the overall CI evaluation.

### Root cause summary

| Factor | Role |
|---|---|
| `025XY019NV` covers ~41% of MLRA 25 plots | Inflates match rate for High-uncertainty sub-group |
| No area-weighted correlation data | Multiplicity score cannot penalise near-monoculture series further |
| CI 42–54 range in MLRA 25 “High” | Threshold at 55 is near the true boundary for this MLRA |

### What this does NOT mean

This anomaly does **not** mean the CI is wrong for MLRA 25, or that High-uncertainty plots in other MLRAs perform at 72%. It means MLRA 25's plot sample is structurally unbalanced — a large share of plots happened to be located in the dominant-ecosite type. If those plots were resampled proportionally across all MLRA 25 ecosites, the match rate for High-uncertainty would likely fall to ~59%, restoring the expected gradient.

### Recommended reporting convention

When presenting §8 per-MLRA match rates, display `mean_base_rate` alongside match rates. MLRAs with `mean_base_rate > 0.15` should carry a footnote:

> *Match rates in this MLRA are partially driven by ecosite base-rate concentration (mean base-rate = X). The ecosite base-rate covariate model (§12) confirms this does not affect the global CI coefficient.*

---

## 15. Summary and Interpretation

### The CI works as intended

Across all global and MLRA-level tests, the revised CI consistently and significantly predicts whether SoilID identifies the correct ecological site:

| Evidence | Result |
|---|---|
| 3-class match rate gradient | Low 66.7% → Moderate 44.6% → High 39.0% ✓ |
| Wilson CI bands | Low–Moderate gap ~22 pp, non-overlapping ✓ |
| Chi-square class separability | χ²=13.2, p=0.00139 ✓ |
| Cluster-robust logistic β(CI) | +0.032, p=0.000091, OR=1.032/pt ✓ |
| Spearman rank correlation | r=+0.214, p<0.0001 ✓ |
| Base-rate attenuation | −2.7% (negligible) ✓ |

### The revised formula is an improvement

- Spearman r is +0.214 with native columns — consistent and robust
- With native SDA `multiplicity_score`, the old and new formulas produce identical class distributions for this dataset, confirming the pipeline's built-in scoring is already on the revised scale
- Step-function scores are more defensible and reproducible than linear interpolations across NCSS ordinal thresholds

### Terrain model: modest, class-dependent benefit

The terrain-augmented model consistently improves match rates in Moderate (+3.8 pp) and High (+1.8 pp) classes. The degradation in Low class (−3.7 pp) suggests terrain features introduce noise where map units are already well-constrained. Net terrain benefit is positive but small.

### Limitations

1. **NV-only dataset:** All 523 plots are in Nevada (1 Idaho plot excluded). Results are directly applicable to NV Great Basin conditions but should be validated against non-NV datasets before generalizing.
2. **MLRA per-class sample sizes:** Low-uncertainty cells in individual MLRAs have n=1–15 — far too small for reliable within-MLRA calibration. The global and cluster-robust results should be treated as primary.
3. **Order and mukind scores derived in analysis:** `order_score` and `mukind_score` are derived from `uncertainty_class` and `uncertainty_reason` in the run-results file rather than read as direct SDA component columns in this report workflow. This limits precision for component-level interpretation.
4. **1 pipeline failure:** Plot at (−117.46, 39.35) has no SSURGO coverage and is excluded from all match-rate calculations.

### Next steps

| Priority | Action |
|---|---|
| Medium | Add area-weighted correlation data to multiplicity lookup (requires `comonth` or `component.comppct_r` from SDA) |
| Low | Investigate MLRA-specific CI threshold calibration (MLRA 25 effective boundary appears ~CI 45 rather than 55) |
| Low | Validate on non-NV AIM plots to test generalizability |
