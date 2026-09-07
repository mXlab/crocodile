"""Regression tests for OnlineFeatureExtractor's live-readiness properties.

These are the properties most likely to silently regress in a future edit
(e.g. someone reintroduces a whole-array recompute for "simplicity"), so
they're checked here rather than relying on the one-off manual verification
done when this was first built (see PIPELINE.md's live-readiness section).

Checks:
1. Strict causality: feeding a session prefix and the full session must
   give byte-identical output on the overlapping rows. A genuinely causal
   system cannot do otherwise -- a row's features must never depend on
   samples that arrive after that row's own timestamp.
2. Live-simulation equivalence: driving push() with small, boundary-
   misaligned chunks must give byte-identical output to process_session()
   on the same data -- proves push() is truly chunk-size-agnostic, not
   just "close enough."
3. Boundedness (informational, not a hard pass/fail): the causal
   filter/detector step (the part this work specifically fixed) should
   cost roughly the same per row late in a session as early on. Full
   end-to-end push() cost is allowed to grow with session length (some
   features are inherently full-history quantities -- trend/max "_full"
   features, _hold_last_value's array reconstruction -- deliberately not
   optimized further, see online_feature_extractor.py's module docstring
   and PIPELINE.md's scoping) as long as it stays well under the 1Hz
   feature-interval budget for realistic (minutes-scale) session lengths.
4. Calibrate-then-push consistency: calling calibrate(prefix) then
   push(remainder) must give the same features as process_session() on
   the whole [prefix + remainder] session -- calibrate() extends the
   growing signal arrays but (until fixed) didn't advance the row counter
   those arrays are indexed against, silently reading calibration data as
   if it were the live tail for the first several rows after calibration.
   Caught by scripts/live_pipeline.py's own end-to-end testing (very large,
   implausible W-vector norms), not by the first three checks above, since
   none of them exercise calibrate().

Usage (from biodata_pipeline/):
    python scripts/test_online_causality.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.online_feature_extractor import OnlineFeatureExtractor

SESSION_PATH = PROJECT_ROOT / 'data' / 'raw' / 'emotion_biodata_laurence_main_1.csv'


def test_strict_causality(prefix_len=50000):
    df = pd.read_csv(SESSION_PATH)
    out_full = OnlineFeatureExtractor(sampling_rate=100).process_session(df)
    out_prefix = OnlineFeatureExtractor(sampling_rate=100).process_session(
        df.iloc[:prefix_len].reset_index(drop=True))

    n_common = len(out_prefix)
    feature_cols = [c for c in out_full.columns if '.' in c]
    common_full = out_full.iloc[:n_common]
    max_diff = max(np.nanmax(np.abs(out_prefix[c].values - common_full[c].values))
                    for c in feature_cols)
    status = "PASS" if max_diff == 0.0 else "FAIL"
    print(f"[{status}] strict causality: prefix ({n_common} rows) vs full "
          f"({len(out_full)} rows) session -- max abs diff on overlap = {max_diff}")
    return max_diff == 0.0


def test_live_simulation_equivalence(n_samples=20000, chunk_size=37):
    """chunk_size is deliberately NOT a divisor of the 100-sample feature
    step, to prove push() doesn't secretly assume step-aligned input."""
    df = pd.read_csv(SESSION_PATH).iloc[:n_samples].reset_index(drop=True)

    out_batch = OnlineFeatureExtractor(sampling_rate=100).process_session(df)

    ext_live = OnlineFeatureExtractor(sampling_rate=100)
    rows = []
    for start in range(0, len(df), chunk_size):
        rows.extend(ext_live.push(df.iloc[start:start + chunk_size]))
    out_live = pd.DataFrame(rows)

    feature_cols = [c for c in out_batch.columns if '.' in c]
    same_rows = len(out_batch) == len(out_live)
    max_diff = max(np.nanmax(np.abs(out_live[c].values - out_batch[c].values))
                    for c in feature_cols) if same_rows else float('nan')
    status = "PASS" if same_rows and max_diff == 0.0 else "FAIL"
    print(f"[{status}] live-simulation equivalence: {chunk_size}-sample chunks vs "
          f"process_session() -- {len(out_batch)} vs {len(out_live)} rows, max abs diff = {max_diff}")
    return same_rows and max_diff == 0.0


def test_detector_boundedness(growth_factor_limit=5.0):
    """Informational: confirms the specific fix (SCR/peak detector state)
    stays roughly flat per-row cost, isolated from the full-history
    aggregate math this work deliberately left unoptimized (see module
    docstring)."""
    df = pd.read_csv(SESSION_PATH)
    fs, step = 100, 100
    ext = OnlineFeatureExtractor(sampling_rate=100)

    detector_times = []
    n_rows = len(df) // step
    for t in range(n_rows):
        chunk = df.iloc[t * step:(t + 1) * step]
        ext._pending['gsr'].extend(chunk['gsr'].values.astype(float).tolist())
        ext._pending['ppg'].extend(chunk['heart'].values.astype(float).tolist())
        ext._pending['resp'].extend(chunk['respiration'].values.astype(float).tolist())
        gsr_chunk = np.array(ext._pending['gsr'][:step]); ext._pending['gsr'] = ext._pending['gsr'][step:]
        ppg_chunk = np.array(ext._pending['ppg'][:step]); ext._pending['ppg'] = ext._pending['ppg'][step:]
        resp_chunk = np.array(ext._pending['resp'][:step]); ext._pending['resp'] = ext._pending['resp'][step:]

        start = time.perf_counter()
        new_gsr_clean = ext._eda_clean_filter.extend(gsr_chunk)
        ext._gsr_clean = np.concatenate([ext._gsr_clean, new_gsr_clean])
        new_tonic = ext._eda_tonic_filter.extend(new_gsr_clean)
        ext._tonic = np.concatenate([ext._tonic, new_tonic])
        new_phasic = new_gsr_clean - new_tonic
        ext._phasic = np.concatenate([ext._phasic, new_phasic])
        new_ppg_clean = ext._ppg_filter.extend(ppg_chunk)
        ext._ppg_clean = np.concatenate([ext._ppg_clean, new_ppg_clean])
        new_resp_clean = ext._resp_filter.extend(resp_chunk)
        ext._resp_clean = np.concatenate([ext._resp_clean, new_resp_clean])
        end = len(ext._gsr_clean)
        ext._scr_detector.extend(new_phasic)
        ext._cardiac_peak_detector.step(ext._ppg_clean, end)
        ext._resp_peak_detector.step(ext._resp_clean, end)
        ext._resp_trough_detector.step(-ext._resp_clean, end)
        detector_times.append(time.perf_counter() - start)

    detector_times = np.array(detector_times)
    early = detector_times[:10].mean()
    late = detector_times[-10:].mean()
    growth = late / early if early > 0 else float('inf')
    status = "PASS" if growth < growth_factor_limit else "WARN"
    print(f"[{status}] detector boundedness: first-10-rows mean {early*1000:.3f}ms -> "
          f"last-10-rows mean {late*1000:.3f}ms (growth {growth:.1f}x, "
          f"limit {growth_factor_limit}x) over a {n_rows}s session")
    return growth < growth_factor_limit


def test_calibrate_then_push_consistency(calib_len=6000, live_len=10000):
    df = pd.read_csv(SESSION_PATH)
    calib_df = df.iloc[:calib_len].reset_index(drop=True)
    live_df = df.iloc[calib_len:calib_len + live_len].reset_index(drop=True)
    combined_df = df.iloc[:calib_len + live_len].reset_index(drop=True)

    out_combined = OnlineFeatureExtractor(sampling_rate=100).process_session(combined_df)

    ext = OnlineFeatureExtractor(sampling_rate=100)
    ext.calibrate(calib_df)
    rows = ext.push(live_df)
    out_live = pd.DataFrame(rows)

    # Rows finalized during push() correspond to the tail of out_combined
    # (calibration doesn't itself finalize rows) -- compare that overlap.
    n_live_rows = len(out_live)
    tail = out_combined.iloc[-n_live_rows:].reset_index(drop=True)
    feature_cols = [c for c in out_combined.columns if '.' in c]
    max_diff = max(np.nanmax(np.abs(out_live[c].values - tail[c].values)) for c in feature_cols)
    status = "PASS" if max_diff == 0.0 else "FAIL"
    print(f"[{status}] calibrate-then-push consistency: {n_live_rows} live rows vs "
          f"process_session() on the combined session -- max abs diff = {max_diff}")
    return max_diff == 0.0


if __name__ == '__main__':
    results = [
        test_strict_causality(),
        test_live_simulation_equivalence(),
        test_detector_boundedness(),
        test_calibrate_then_push_consistency(),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
