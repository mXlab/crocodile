# Experimental Results — Cross-Subject Alignment

Results from running the "Cross-subject alignment" workflow in
[README.md](README.md) against real data: Erin (`emotion_biodata_erin_2026-02-09_labeled.csv`)
as the calibration subject, Laurence's combined sessions
(`emotion_biodata_laurence_main_*.csv`) as the reference/actress data, on the
three emotions they share (`anx`, `neu`, `sad`). See README.md for the
commands used to reproduce any of this.

## Closed-set benchmark

Calibrate and evaluate on the *same* emotions — the standard workflow-3
Step 2/3 usage.

| Method | RMSE\_norm | NPA | RF accuracy |
|--------|-----------|-----|-------------|
| Ridge | 0.397 | 52.8% | 59.3% |
| OT global | 0.455 | 37.4% | 30.2% |
| CORAL | 0.441 | 40.5% | 31.8% |
| OT class-conditional | **0.000** | **54.2%** | **77.0%** |

Class-conditional OT wins on all metrics in this closed-set benchmark. CORAL
lands close to OT global — both are unsupervised, class-blind, moment-matching
linear maps, so neither corrects per-emotion structure, and both underperform
Ridge (which is also global but at least trained toward the per-emotion
prototypes it does have).

This benchmark only tells you about the closed-set case. For a deployment
where the calibration set covers a handful of elicited emotions but the
participant is later observed in states outside that set, the class-blind
methods (CORAL, OT global, and — with a caveat — Ridge) have no per-class
assignment step to fail on unseen states, whereas class-conditional OT
force-assigns every sample to its nearest *calibrated* prototype and applies
that emotion's specific map, which is untested outside the calibrated set.
That trade-off is measured next.

## Held-out-emotion generalization test

Simulates the deployment scenario where a participant is observed in a state
outside the calibrated set (`scripts/validate_heldout_emotion.py`, workflow-3
Step 4): for each shared emotion, fit every method on the *other* shared
emotions only, then check how each one handles the emotion it never saw.

Two metrics per held-out emotion:
- **Held-out RMSE_norm** — distance from the transformed held-out samples to
  the actress' *true* prototype for that emotion (which the transformer never
  saw), normalized on the same scale as the closed-set RMSE_norm above.
- **RF recall(h)** — a Random Forest trained on the reference's full
  known-emotion data classifies the transformed held-out samples; the
  fraction correctly labeled as the true (held-out) emotion.

**Result, holding out each of anx/neu/sad in turn (2 calibrated emotions per split):**

| Held out | Ridge | OT global | OT class-conditional | CORAL |
|---|---|---|---|---|
| anx | RMSE_norm 0.463, recall 0% | 2.262, 1.2% | 2.377, **39.7%** | 2.470, 21.2% |
| neu | RMSE_norm **5.597**, recall 5.6% | 13.441, 5.0% | 14.509, 3.4% | 28.177, 2.2% |
| sad | RMSE_norm **1.131**, recall 0% | 2.472, 0% | 3.167, 0% | 2.361, 0% |

Two things stand out, and they cut in different directions:

1. **On raw distance (RMSE_norm), Ridge extrapolates far more conservatively than the other three** in every split — consistent with it being the most heavily regularized, lowest-capacity map (fit from just 2 prototype pairs). The OT-based methods and CORAL, which fit full covariance structure from the same 2 calibrated classes, swing much further off target when asked to place data outside what they were fit on (up to 28x the inter-class scale for CORAL on `neu`).
2. **But no method reliably lands the held-out samples in the correct RF decision region.** Recall is near-zero-to-single-digits for `neu` and exactly 0% for `sad` across all four methods — the transformed points systematically get misclassified as one of the *other* known emotions (`sad`→`neu` almost 100% of the time for three of the four methods; `neu`→`anx`/`sad` for all four). Class-conditional OT's 39.7% on `anx` is the one exception, but it comes from a wilder, higher-RMSE transform that happens to spread samples across all three known buckets rather than concentrating them in one wrong one — not evidence it's actually generalizing.

**Takeaway: none of the four methods generalize reliably to an emotion outside the calibration set** on this data. Ridge's smoothness protects against wild, physiologically-implausible outputs (the safer failure mode for driving a face in real time) but doesn't mean the transformed signal is usable for anything downstream trying to recognize what the participant is actually feeling. With only 3 emotions shared between Erin and the actress, each split calibrates on just 2 classes — a small, noisy test — but the direction of the result (universal failure to generalize, not "method X solves it") is unlikely to be a data-size artifact alone.

## Does reducing dimensionality fix it? (ANOVA feature selection)

All 73 features are estimated from as few as ~130-680 samples per emotion;
the covariance-based methods (OT global, class-conditional OT, CORAL) each
fit a 73×73 covariance matrix (2,701 free parameters) per class from that,
which is exactly the kind of under-sampled estimate that produces unstable
inverses. `validate_heldout_emotion.py --n-features N` re-runs the same
held-out test with ANOVA feature selection (reusing Ridge's existing
`--n-features` convention), re-selected per split from the calibration
emotions' reference data only. See [FEATURES.md](FEATURES.md) for the full
ANOVA ranking of all 73 features on this same reference data.

> Correction: the first version of this feature-selection code (in both
> `--n-features` here and in Ridge's pre-existing `--n-features` option in
> `train_transformer.py`) had a real bug -- `np.argsort(f_scores)[::-1]`
> sorts `NaN` scores (from features that are constant within a calibration
> split) to the *front* instead of the back, so a degenerate feature could
> get selected ahead of a genuinely informative one. Fixed in both places
> (`np.nan_to_num(f_scores, nan=-np.inf)` before ranking); the numbers below
> are post-fix.

**Held-out RMSE_norm at 73 vs. 20 vs. 10 features:**

| Held out | Ridge | OT global | OT class-conditional | CORAL |
|---|---|---|---|---|
| anx | 0.463 / 0.421 / 0.753 | 2.262 / 1.596 / **0.909** | 2.377 / 1.650 / 1.698 | 2.470 / 1.719 / **1.046** |
| neu | 5.597 / 1.435 / **1.497** | 13.441 / 9.274 / **6.830** | 14.509 / 9.272 / **7.741** | 28.177 / 11.947 / **8.994** |
| sad | 1.131 / 1.088 / 1.048 | 2.472 / 2.297 / **1.369** | 3.167 / 2.402 / 1.534 | 2.361 / 2.224 / 1.351 |

(each cell: 73 features / 20 features / 10 features)

20 features lands where you'd expect — a real but partial version of the 10-feature effect:

- **Ridge is close to flat across all three settings** (e.g. `anx`: 0.463 → 0.421 → 0.753) — it never estimates a covariance, so cutting dimensionality mostly just changes *which* 2-point regression it fits, not how stable the fit is. The wobble at 10 features on `anx` is ANOVA selection optimizing for the two calibration classes (`neu`/`sad`) and occasionally picking features that generalize worse to the third, unrelated one.
- **The covariance-based methods improve monotonically as dimensionality drops, and 20 features already captures most of the direction of the effect** (OT global on `neu`: 13.441 → 9.274 → 6.830; CORAL: 28.177 → 11.947 → 8.994) — going all the way to 10 buys further improvement but with diminishing returns past 20, and at higher risk of the ANOVA-selection instability visible in Ridge's wobble.
- **Recall on the true held-out class doesn't track RMSE_norm.** It only improves meaningfully for the `neu` split (e.g. CORAL recall ~2% at 73 features → ~20% at both 10 and 20), and stays at or near 0% for `anx`/`sad` regardless of feature count — the systematic misrouting (`sad`→`neu`, `anx`→`sad`) doesn't change direction at any dimensionality tested.

**Interpretation, and 20 vs. 10 as a choice:** dimensionality is a real contributor to *how unstable* the covariance-based maps are — 20 features already recovers most of that benefit without cutting as aggressively as 10, so it's a reasonable middle ground if you want to keep more physiological detail. But dimensionality isn't the main reason held-out emotions get mapped to the *wrong* known emotion — that misrouting is consistent from 10 through 73 features, which points at a genuine structural relationship in the raw feature space (`sad` and `neu` sit closer to each other physiologically, for this subject pair, than either does to the actress' true `sad`/`neu` targets) rather than a noisy high-dimensional artifact. That's consistent with a valence-arousal reframing of the alignment problem (discrete calibrated emotions replaced by a continuous target space) being a more promising direction than further tuning of these four methods — dimensionality reduction can stabilize the map, but it doesn't give the target space any structure that would tell the model *why* `sad` and `neu` are related and in which direction a truly novel state should be placed relative to them.
