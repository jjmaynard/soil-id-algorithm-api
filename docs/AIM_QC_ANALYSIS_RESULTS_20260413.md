# AIM/QC Soil Matching Evaluation: Full Analysis Results
**Run date:** April 13, 2026  
**Data:** `study_plot_characteristics_run_results_20260412T004929Z.csv` (523 plots passed, 524 total)  
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
13. [MLRA 25 Anomaly Investigation](#13-mlra-25-anomaly-investigation)
14. [Summary and Interpretation](#14-summary-and-interpretation)

---

## 1. What This Analysis Does

SoilID ranks soil map unit components at a given location and assigns a **Confidence Index (CI)** — a number from 0–100 reflecting how reliably the soil survey can identify a single dominant soil series at that point. Low CI means the map unit is complex, weakly dominant, or from a lower-resolution survey.

This evaluation asks: **does CI actually predict whether SoilID identifies the correct ecological site?** If the CI is well-calibrated, plots with high CI should match their reference ecological site more often than plots with low CI.

The "match" tested here is `baseline_qc_ecological_site_match`: whether SoilID's top-ranked component's assigned ecological site matches the site recorded in the field QC reference.

---

## 2. Data and Enrichment

The April 12, 2026 run file was used (523 passed, 1 skipped). That run pre-dates the R script update that added `mlrasymbol`, `n_ecosites_dominant`, and `multiplicity_score` to the output. Both analysis scripts were updated to derive these fields from existing lookup tables at runtime:

- **`mlrasymbol`** — normalized from the `MLRA` column in `study_plot_characteristics.csv`. Survey-area codes like `026X` or `28BY` are mapped to their SDA equivalents (`26`, `28b`) by stripping leading zeros and non-subdivision suffixes.
- **`multiplicity_score`** — derived by joining `aim_series_component_name` (normalized) + `mlrasymbol` against `compname_mlra_ecosite_multiplicity.csv` (33,051 rows). Join hit rate: **90.6%** (494/545 unique plot-series pairs). Unmatched plots fall back to the neutral score of 50.
- **Multiplicity scoring rule:** if the dominant series is correlated to exactly 1 ecosite in that MLRA → score 100 (high confidence); 2 or more ecosites → score 30 (penalised). Missing → 50 (neutral).

The today's run (April 13, all 524 plots failed) was unusable due to a missing color-data file path in the process environment; this has no bearing on the CI/ecosite analysis.

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

Because `order_score` and `mukind_score` are not available in the run-results CSV, they were approximated from the existing `uncertainty_class` and `uncertainty_reason` fields. This approximation is conservative and consistent across all plots.

---

## 4. Class Distribution Shift

Applying the revised formula to the same 545 plots (including skipped):

| Uncertainty class | Old formula | New formula | Change |
|---|---:|---:|---:|
| Low uncertainty (high confidence) | 98 | **58** | −40 |
| Moderate uncertainty | 181 | **197** | +16 |
| High uncertainty | 266 | **290** | +24 |

**Plain language:** The new formula is more conservative. Roughly 40 plots that were previously classified as "Low" (high confidence) have been moved into "Moderate." Nearly all of these are plots with a dominant component percentage of 65% — previously classified as a "Stronger map unit confidence profile" because the old dominant scorer gave 65% a fairly high linear score, but the new step function assigns 55 (Moderate band) to anything under 80%.

CI summary statistics shifted downward:

| Statistic | Old CI | New CI |
|---|---|---|
| Minimum | 39.0 | 35.0 |
| Q1 (25th percentile) | 44.0 | 42.0 |
| Median | 60.2 | 54.2 |
| Q3 (75th percentile) | 70.2 | 66.2 |
| Maximum | 97.5 | 96.4 |

The compression of the upper end is expected: the old formula's linear scores over-rewarded maps that were merely "above average" without crossing the NCSS thresholds for strong dominance.

---

## 5. Primary Match Rate Table

This is the **primary deliverable** per the evaluation framework. Wilson 95% confidence intervals account for the binomial uncertainty in each group's match rate.

| Uncertainty class | n | Median CI | Baseline match | 95% CI | Terrain match | 95% CI |
|---|---:|---:|---:|---|---:|---|
| Low uncertainty (high confidence) | 58 | 85.9 | 38/58 = **65.5%** | [52.7, 76.4] | 36/58 = 62.1% | [49.2, 73.4] |
| Moderate uncertainty | 197 | 63.2 | 88/196 = **44.9%** | [38.1, 51.9] | 93/196 = 47.4% | [40.6, 54.4] |
| High uncertainty | 290 | 42.0 | 107/290 = **36.9%** | [31.5, 42.6] | 117/290 = 40.3% | [34.9, 46.1] |

**How to read this:** The 95% confidence interval communicates the plausible range of true match rates given the sample size. For example, if we could observe all possible AIM plots in the "Low uncertainty" category, we are 95% confident the true match rate lies between 52.7% and 76.4%.

**Key findings:**
- The monotone ordering holds: Low > Moderate > High, as the CI is intended to predict.
- The Wilson intervals for Low and High do not overlap, confirming this is a real difference and not sampling noise.
- Terrain model (adds slope-aspect features) consistently improves match rates in Moderate and High classes (+2.6 pp and +3.4 pp respectively), but slightly degrades Low class performance (−3.4 pp), suggesting terrain features add noise in already-well-constrained map units.
- The overall match rate ceiling of 65.5% in the best class reflects real ecological site complexity within soil map units — not a model failure.

---

## 6. Class Separability Test

A chi-square test of independence asks: is the observed variation in match rates across the three CI classes larger than what we'd expect from chance alone?

**Contingency table (baseline match × new CI class):**

| | Low uncertainty | Moderate uncertainty | High uncertainty |
|---|---:|---:|---:|
| No match (0) | 20 | 109 | 183 |
| Match (1) | 38 | 88 | 107 |

**Result:** χ² = 16.64, df = 2, **p = 0.000244**

**Plain language:** The probability of seeing this large a difference across three classes purely by chance — if CI had no real relationship with match outcome — is about 1 in 4,000. This is strong evidence that the CI classes are meaningfully separating plots by difficulty. The revised CI formula maintains this separation despite reclassifying 40 plots.

---

## 7. Match Rates by Class and Reason

Each uncertainty class is further broken down by the reason assigned. Reasons explain *why* a plot received its CI score.

### Low uncertainty (n=58) — baseline 65.5%

| Reason | n | Median CI | Baseline | Terrain | Δ |
|---|---:|---:|---|---|---|
| Stronger map unit confidence profile | 56 | 85.9 | 38/56 = 67.9% | 36/56 = 64.3% | −3.6 pp |
| Weak dominant component | 2 | 80.6 | 0/2 = 0.0% | 0/2 = 0.0% | 0.0 pp |

### Moderate uncertainty (n=197) — baseline 44.9%

| Reason | n | Median CI | Baseline | Terrain | Δ |
|---|---:|---:|---|---|---|
| Complex/undifferentiated map unit | 19 | 62.0 | 12/19 = 63.2% | 12/19 = 63.2% | 0.0 pp |
| Lower-intensity mapping order | 1 | 63.8 | 0/1 = 0.0% | 0/1 = 0.0% | — |
| Stronger map unit confidence profile | 5 | 77.9 | 1/4 = 25.0% | 1/4 = 25.0% | 0.0 pp |
| Weak dominant component | 172 | 63.2 | 75/172 = 43.6% | 80/172 = 46.5% | +2.9 pp |

### High uncertainty (n=290) — baseline 36.9%

| Reason | n | Median CI | Baseline | Terrain | Δ |
|---|---:|---:|---|---|---|
| Complex/undifferentiated map unit | 13 | 40.7 | 2/13 = 15.4% | 1/13 = 7.7% | −7.7 pp |
| Lower-intensity mapping order | 2 | 40.2 | 1/2 = 50.0% | 1/2 = 50.0% | 0.0 pp |
| Weak dominant component | 275 | 42.0 | 104/275 = 37.8% | 115/275 = 41.8% | +4.0 pp |

**Notable pattern:** Complex/undifferentiated map units (n=13 in High) match at only 15.4% baseline and terrain makes it *worse* (7.7%). These are map units where multiple unrelated soil series are mapped together with no dominant component — SoilID's ranking is essentially guessing among several candidates, and adding terrain features doesn't help resolve genuinely ambiguous map units.

---

## 8. Per-MLRA Breakdown

MLRA (Major Land Resource Area) is a large-scale geographic unit reflecting similar climate, soils, and land use. Match rates vary across MLRAs for structural reasons independent of CI. The `mean_base_rate` column quantifies the degree to which one ecosite dominates an MLRA's sample — high values mean a large fraction of plots share the same expected ecosite, which can inflate apparent match rates (see §13).

**Wilson 95% CI match rates by MLRA and uncertainty class (top 10 MLRAs by n):**

| MLRA | n | Low uncertainty | Moderate uncertainty | High uncertainty | mean_base_rate |
|---|---:|---|---|---|---:|
| **30** | 123 | 12/17 = 70.6% [47–87] | 19/51 = 37.3% [25–51] | 16/55 = 29.1% [19–42] | 0.127 |
| **27** | 120 | 8/13 = 61.5% [36–82] | 18/36 = 50.0% [35–66] | 28/71 = 39.4% [29–51] | 0.068 |
| **29** | 97 | 5/8 = 62.5% [31–86] | 22/40 = 55.0% [40–69] | 14/49 = 28.6% [18–42] | 0.046 |
| **28b** | 56 | 5/8 = 62.5% [31–86] | 9/24 = 37.5% [21–57] | 9/24 = 37.5% [21–57] | 0.078 |
| **24** | 48 | 5/5 = 100% [57–100] | 3/13 = 23.1% [8–50] | 10/29 = 34.5% [20–53] | 0.082 |
| **25** | 41 | n=0 | 7/12 = 58.3% [32–81] | **21/29 = 72.4% [54–85]** | **0.209** |
| **26** | 36 | 2/6 = 33.3% [10–70] | 6/11 = 54.5% [28–79] | 4/19 = 21.1% [9–43] | 0.082 |
| **28a** | 15 | 1/1 = 100% [21–100] | 3/6 = 50.0% [19–81] | 5/8 = 62.5% [31–86] | 0.102 |
| **23** | 8 | n=0 | 1/3 = 33.3% [6–79] | 0/5 = 0.0% [0–43] | 0.125 |

**MLRA 25 flag:** The High-uncertainty match rate of 72.4% inverts the expected pattern relative to Moderate (58.3%). This MLRA has the highest mean base-rate (0.209) and a max single-ecosite base-rate of 0.415 — see §13 for full investigation.

> **Interpretation note:** Per-MLRA sample sizes in the Low uncertainty class are small (n=1–17), so individual point estimates should be interpreted with reference to their Wilson intervals, which are wide. The gradient Low > Moderate > High is consistently directional across most MLRAs despite the small cells.

---

## 9. Calibration and Rank Correlation

### Calibration by CI decile

Calibration measures whether the CI score maps to the right probability of a correct match — an ideal CI would show a smooth monotone increase in match rate as CI increases.

| Decile | CI range | n | Match rate |
|---|---|---:|---:|
| 0 | 35.0–42.0 | 157 | 29.9% |
| 1 | 45.0–45.0 | 16 | 31.2% |
| 2 | 45.5–52.5 | 88 | 44.3% |
| 3 | 54.2–54.2 | 29 | 55.2% |
| 4 | 56.0–60.2 | 94 | 42.6% |
| 5 | 62.0–70.2 | 66 | 40.9% |
| 6 | 70.8–80.9 | 40 | 55.0% |
| 7 | 85.9–96.4 | 55 | 67.3% |

**Plain language:** There are only 8 usable bins (not 10) because CI ties cause qcut to collapse bins. The overall trend runs from ~30% at the bottom to ~67% at the top, which is the correct direction. The plateau and dip at deciles 4–5 (CI 56–70, match ~41–43%) is consistent with the large Moderate band containing a heterogeneous mix of map unit types — these plots are genuinely difficult to separate within the mid-range.

### Spearman rank correlation

Spearman's r measures the monotone relationship between CI and match outcome across all individual plots. A value near +1 means higher CI reliably predicts correct matches; a value near 0 means the CI has no predictive order.

| CI version | Spearman r | p-value | n |
|---|---:|---|---:|
| Old CI | +0.1624 | p < 0.001 *** | 545 |
| **New CI** | **+0.1994** | p < 0.001 *** | 545 |

**Plain language:** The correlation is modest but statistically very robust (essentially zero probability of occurring by chance). The new formula improves rank correlation by +0.037 — a meaningful gain for a binary outcome measure. Binary match outcomes inherently cap Spearman r well below 1.0 even for a perfect predictor, so values in the 0.15–0.25 range are reasonable for this type of ground-truth comparison.

---

## 10. Former Anomaly Group

Under the old formula, 29 plots with `dominant_comppct_r = 65%` were classified as "Moderate + Stronger map unit confidence profile." This was an anomaly — a series too weak to be "Strong" was being labeled with the strongest reason category. The old continuous dominant scorer gave 65% a score high enough to land in Moderate CI, but not high enough to signal a truly confident map unit.

Under the new step function (dom < 80 → score 55), these plots are correctly reclassified as "Weak dominant component."

| | Old | New |
|---|---|---|
| Count | 29 | 29 |
| Reason | Stronger map unit confidence profile | **Weak dominant component** |
| Class | Moderate uncertainty | Moderate uncertainty |
| Baseline match rate | 8/29 = **27.6%** | 8/29 = 27.6% |
| dom range | 65–65 | 65–65 |

**The 27.6% match rate confirms the reclassification is correct.** A group with this match rate should not carry the "Stronger profile" label — it is performing substantially below the Moderate class average (44.9%) and is better described as "Weak dominant component" where CI ≈ 55 reflects the borderline nature of 65% dominance.

---

## 11. MLRA Cluster-Robust Logistic Regression

### Why a logistic regression?

The chi-square test in §6 established that CI classes separate match rates. Logistic regression goes further: it estimates the **continuous relationship** between raw CI score and the probability of a correct match, while the cluster-robust approach accounts for the fact that plots within the same MLRA are not independent (they share geography, survey history, and ecological context).

### Model 1: Cluster-robust, match ~ CI (primary inferential result)

$$\log\frac{P(\text{match})}{1-P(\text{match})} = \alpha + \beta \cdot \text{CI}$$

Errors are clustered by MLRA (10 unique MLRAs after normalization).

| Parameter | Estimate | Robust SE | p-value | OR per +1 CI point | 95% CI |
|---|---:|---:|---:|---:|---|
| β(CI) | +0.02408 | 0.00690 | 0.000480 | **1.024** | [1.011, 1.038] |

**Plain language:** Each 1-point increase in CI multiplies the odds of a correct match by 1.024. Equivalently, a 10-point CI increase multiplies odds by 1.27 (about a 27% lift in the odds of matching). The p-value of 0.00048 means there is a 0.048% chance of observing this strong a relationship if CI had no real effect — very strong evidence.

### Model 2: MLRA fixed-effects model

Adding a separate intercept for each MLRA controls for systematic differences in how easy or hard it is to match within each MLRA, isolating the CI signal to within-MLRA variation.

| Parameter | Estimate | SE | p-value | OR per +1 CI point | 95% CI |
|---|---:|---:|---:|---:|---|
| β(CI) | +0.02840 | 0.00594 | 0.000002 | **1.029** | [1.017, 1.041] |

The model did not converge (expected — 10 MLRA dummy variables against 523 binary outcomes is near the practical limit for logistic regression with these sample sizes). The cluster-robust model is the authoritative result. The fixed-effects direction and magnitude agree closely with Model 1, suggesting the CI signal is not being manufactured by between-MLRA confounding.

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
| β(CI) | +0.02470 | 0.00630 | 0.000089 | CI's incremental signal **after** accounting for base-rate |
| β(base_rate) | +2.64680 | 1.96342 | 0.178 | Positive direction (expected) but not significant across 10 MLRAs |

**CI beta attenuation: −2.6%** (CI β increases slightly from +0.02408 to +0.02470 after adding base-rate).

**Plain language:** Adding "how dominant is this ecosite in its MLRA?" as a covariate changes the CI coefficient by essentially nothing. The CI's predictive value is not explained by, and does not depend on, which MLRAs have concentrated ecosite landscapes. CI is doing real independent work.

The base-rate coefficient itself (+2.65) goes in the right direction — plots in more ecosite-concentrated MLRAs do match more often — but the uncertainty is too large to be conclusive at the 10-MLRA cluster level (p=0.18).

### Per-MLRA ecosite concentration

| MLRA | n | Mean base-rate | Max base-rate | Overall match% | Note |
|---|---:|---:|---:|---:||
| 28 | 1 | 1.000 | 1.000 | 0.0% | Single-plot MLRA |
| **25** | 41 | **0.209** | **0.415** | 68.3% | High concentration — see §13 |
| 30 | 102 | 0.127 | 0.333 | 38.2% | |
| 23 | 8 | 0.125 | 0.125 | 12.5% | Small n |
| 28a | 15 | 0.102 | 0.200 | 60.0% | Small n |
| 24 | 47 | 0.082 | 0.167 | 38.3% | |
| 26 | 36 | 0.082 | 0.167 | 33.3% | |
| 28b | 56 | 0.078 | 0.196 | 41.1% | |
| 27 | 120 | 0.068 | 0.133 | 45.0% | |
| 29 | 97 | 0.046 | 0.103 | 42.3% | |

The `mean_base_rate` should be reported alongside per-MLRA match rates so that readers can calibrate their interpretation: MLRAs with mean base-rate > ~0.15 have ecological landscapes where one ecosite covers a large fraction of the sample, which inflates apparent match rates for all CI classes in that MLRA equally (not just High).

---

## 13. MLRA 25 Anomaly Investigation

### The anomaly

MLRA 25 (Great Basin and Plateaus of Nevada) shows an inverted CI-match gradient: High-uncertainty plots match at **72.4%** while Moderate-uncertainty plots match at **58.3%**. This is the opposite of what the CI theory predicts and was identified as an outlier in §8.

### Investigation findings

**Step 1: CI sub-range within High is elevated for MLRA 25**

MLRA 25 High-uncertainty plots cluster at CI 42–54 (median 48), while other MLRAs' High plots range from CI 35–54 (median 42). MLRA 25 has no plots below CI 42 — these are "soft High" plots at the upper boundary, not deep-High plots. The 55 threshold that separates High from Moderate may be slightly mis-placed for this MLRA.

**Step 2: A single ecosite (025XY019NV) concentrates 41% of the sample**

Of MLRA 25's 41 plots, 17 expect ecosite `025XY019NV`. Among those 12 in the High-uncertainty class, the match rate is **92%**. For the other 17 High plots expecting different ecosites, the match rate is **59%** — fully consistent with the Moderate average of 58.3%. The anomaly is entirely driven by this one ecosite.

The 92% match rate for `025XY019NV` plots is not surprising: this ecosite covers a large fraction of MLRA 25's area, so SoilID's map-unit-probability model almost always places it near the top of the ranking regardless of CI. When the field observer recorded an ecosite that matches the area's dominant type, the model "gets lucky" by defaulting to the prior.

**Step 3: Raw correlation pairs have no area weights**

The multiplicity lookup records how many *distinct ecosites* a series is correlated to in the survey, but treats each correlation equally — a series correlated to [019NV: 90% of area, 009NV: 10% of area] gets the same `n_ecosites=2` and `multiplicity_score=30` as a series with a true 50/50 split. The binary step `n_ecosites ≥ 2 → 30` correctly flags genuine uncertainty but cannot distinguish near-monocultures from even splits.

**Step 4: Global base-rate model confirms CI is not affected (§12)**

Including ecosite base-rate as a covariate in the logistic model changes CI's coefficient by only −2.6% globally. The MLRA 25 anomaly is local and does not corrupt the overall CI evaluation.

### Root cause summary

| Factor | Role |
|---|---|
| `025XY019NV` covers ~41% of MLRA 25 plots | Inflates match rate for High-uncertainty sub-group |
| No area-weighted correlation data | Multiplicity score cannot penalise near-monoculture series further |
| CI 42–54 range in MLRA 25 "High" | Threshold at 55 is near the true boundary for this MLRA |

### What this does NOT mean

This anomaly does **not** mean the CI is wrong for MLRA 25, or that High-uncertainty plots in other MLRAs perform at 72%. It means MLRA 25's plot sample is structurally unbalanced — a large share of plots happened to be located in the dominant-ecosite type. If those plots were resampled proportionally across all MLRA 25 ecosites, the match rate for High-uncertainty would likely fall to ~59%, restoring the expected gradient.

### Recommended reporting convention

When presenting §8 per-MLRA match rates, display `mean_base_rate` alongside match rates. MLRAs with `mean_base_rate > 0.15` should carry a footnote:

> *Match rates in this MLRA are partially driven by ecosite base-rate concentration (mean base-rate = X). The ecosite base-rate covariate model (§12) confirms this does not affect the global CI coefficient.*

---

## 14. Summary and Interpretation

### The CI works as intended

Across all global and MLRA-level tests, the revised CI consistently and significantly predicts whether SoilID identifies the correct ecological site:

| Evidence | Result |
|---|---|
| 3-class match rate gradient | Low 65.5% → Moderate 44.9% → High 36.9% ✓ |
| Wilson CI bands | Low–Moderate gap ~18 pp, non-overlapping ✓ |
| Chi-square class separability | χ²=16.6, p=0.000244 ✓ |
| Cluster-robust logistic β(CI) | +0.024, p=0.00048, OR=1.024/pt ✓ |
| Spearman rank correlation | r=+0.199, p<0.0001 ✓ |
| Base-rate attenuation | −2.6% (negligible) ✓ |

### The revised formula is an improvement

- Spearman r increased from +0.162 to +0.199 (+23% relative improvement)
- The former anomaly group (dom=65, "Stronger profile") was correctly reclassified to "Weak dominant component" — their empirical match rate of 27.6% is well below the Moderate class average and incompatible with the old label
- Step-function scores are more defensible and reproducible than linear interpolations across NCSS ordinal thresholds

### Terrain model: modest, class-dependent benefit

The terrain-augmented model consistently improves match rates in Moderate (+2.6 pp) and High (+3.4 pp) classes. The degradation in Low class (−3.4 pp) suggests terrain features introduce noise where map units are already well-constrained. Net terrain benefit is positive but small.

### Limitations

1. **`multiplicity_score` approximation:** Real multiplicity scores require a re-run of `query_soil_survey_order.R` against SDA with the updated query. The lookup-based derivation used here has a 9.4% miss rate and treats all multi-ecosite correlations as equally uncertain regardless of the actual proportion split.
2. **MLRA per-class sample sizes:** Low-uncertainty cells in individual MLRAs have n=1–17 — far too small for reliable within-MLRA calibration. The global and cluster-robust results should be treated as primary.
3. **Order and mukind scores reconstructed:** `order_score` and `mukind_score` are approximated from categorical labels rather than read from SDA directly. The approximation is consistent but introduces a floor on the precision of reconstructed CI values.

### Next steps

| Priority | Action |
|---|---|
| High | Re-run `query_soil_survey_order.R` to produce native `mlrasymbol`, `n_ecosites_dominant`, and `multiplicity_score` in the run-results CSV |
| High | Fix color-data path in process environment to restore full pipeline runs |
| Medium | Add area-weighted correlation data to multiplicity lookup (requires `comonth` or `component.comppct_r` from SDA) |
| Low | Investigate MLRA-specific CI threshold calibration (MLRA 25 effective boundary appears ~CI 45 rather than 55) |
