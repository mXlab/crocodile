"""Online (real-time-safe) counterpart to BatchFeatureExtractor -- computes
the SAME 51 NeuroKit2-schema features (see batch_feature_extractor.py) but
derives the underlying physiological signals (EDA tonic/phasic, cardiac
rate/quality, respiratory amplitude/RVT/symmetry/phase) using only past and
current samples, never NeuroKit2's offline whole-session algorithms
(cvxEDA, elgendi PPG cleaning, RSP_Rate from full-session peak detection).

Architecture: BatchFeatureExtractor already splits each modality into
`_*_derive_signals()` (NeuroKit2-based, offline) and `_*_aggregate()` (the
per-second windowed feature formulas -- already online-safe, since each
second `t` only ever reads its inputs up to `end = (t+1)*step`). This class
subclasses BatchFeatureExtractor and overrides only the three
`_*_derive_signals()` methods with online-safe algorithms, so the 51 feature
*definitions* (names, window sizes, formulas) are guaranteed identical --
only the underlying signal quality differs. This is what makes a fair,
feature-by-feature comparison possible (see scripts/compare_extractors.py).

A note on what "online-safe" means for the implementation below: some
signal-level steps use `scipy.signal.sosfilt` over an entire raw array in a
single call rather than a literal sample-by-sample Python loop. This is NOT
a shortcut that leaks future data -- `sosfilt` is a strictly causal
recursive (IIR) filter, so sample i of its output depends only on samples
0..i of the input; handing it the whole array in one call produces exactly
the output a true streaming implementation would produce sample-by-sample,
just faster to compute in Python/NumPy. Peak/trough/SCR-onset detection
below, by contrast, genuinely needs a forward-only algorithm (a plain
Python loop, or a fixed lookback-window rescan repeated every second) since
`scipy.signal.find_peaks` run on a whole array WOULD leak future context.

Known approximations vs. NeuroKit2 (documented so the comparison script's
results are interpretable, not just "worse"):
- EDA tonic/phasic split uses a causal lowpass/highpass filter pair instead
  of cvxEDA's global convex optimization -- the same category of method as
  NeuroKit2's own `eda_phasic(method="highpass")` alternative to cvxEDA.
- SCR onset/peak/amplitude/rise-time/recovery-time detection is a bespoke
  slope-threshold state machine (trough -> sustained rise -> confirmed
  decline), calibrated from each session's own first 30s of phasic signal
  (mirrors the calibration-phase design discussed for real-time deployment),
  not NeuroKit2's `eda_findpeaks`.
- Cardiac/respiratory peak and trough detection use a rolling z-scored
  lookback-window rescan (adapted from the existing
  EnhancedContinuousFeatureExtractor's approach) instead of NeuroKit2's
  `ppg_findpeaks`/`rsp_findpeaks`.
- `cardiac.quality_mean_10s` is a rough IBI-regularity proxy (deviation from
  the local median inter-beat interval), not NeuroKit2's template-matching
  signal quality index -- this is the single feature with the weakest
  conceptual match to its NeuroKit2 namesake.
- Respiratory RVT is approximated as amplitude / cycle-duration rather than
  NeuroKit2's smoothed Harrison et al. (2021) formulation.
"""

import warnings

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, find_peaks

from modules.batch_feature_extractor import BatchFeatureExtractor

# Expected before the first cardiac/respiratory event is confirmed (no
# online-detected peaks yet -> genuinely empty/all-NaN windows), unlike
# BatchFeatureExtractor where NeuroKit2's offline detection has full-session
# context from the first sample. Matches the existing suppression in
# EnhancedContinuousFeatureExtractor for the same class of cold-start warning.
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')


def _causal_filter(x, fs, btype, cutoff, order):
    """Single-pass causal IIR filter (see module docstring: not a lookahead
    shortcut -- sosfilt is strictly causal). `cutoff` is a scalar for
    low/high, a (low, high) pair for 'band'.

    Initializes the filter's internal state to its steady-state response to
    a constant input at x[0] (`sosfilt_zi(sos) * x[0]`) instead of the
    default all-zero state -- with zero-init, a slow lowpass (e.g. the 0.05Hz
    tonic filter) takes tens of seconds to ramp up from 0 to the signal's
    actual DC level, producing a huge spurious cold-start transient in the
    high-pass/phasic residual (measured: first-30s phasic std ~195 vs. a
    true in-session ~30-60, large enough to blow out any fixed SCR amplitude
    threshold). Steady-state init removes this without needing any lookahead."""
    sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')
    zi = sosfilt_zi(sos) * x[0]
    y, _ = sosfilt(sos, x, zi=zi)
    return y


def _detect_events_online(signal, fs, min_distance_s, lookback_s, height_z=0.4,
                           step_s=1.0, confirm_margin_s=0.2):
    """Generic causal peak detector: at each second, z-scores a fixed
    lookback window ending at the current sample and re-runs find_peaks on
    it, accepting only NEWLY-seen peaks that already have `confirm_margin_s`
    of decline after them (so a peak sitting right at the current sample
    isn't accepted before we've actually seen it start declining -- the
    real-time equivalent of "you can't confirm a peak until it's past").
    Used for both cardiac PPG peaks and respiratory peaks/troughs (pass
    `-signal` for troughs).
    """
    n = len(signal)
    step = int(step_s * fs)
    lookback = int(lookback_s * fs)
    min_distance = max(1, int(min_distance_s * fs))
    confirm_margin = int(confirm_margin_s * fs)

    confirmed = []
    last_idx = -min_distance
    for end in range(step, n + step, step):
        end = min(end, n)
        start = max(0, end - lookback)
        window = signal[start:end]
        if len(window) < 2 * min_distance:
            continue
        sigma = window.std()
        if sigma == 0:
            continue
        z = (window - window.mean()) / sigma
        pk, _ = find_peaks(z, distance=min_distance, height=height_z)
        for p in pk:
            global_idx = start + p
            if global_idx <= last_idx or (end - global_idx) < confirm_margin:
                continue
            confirmed.append(global_idx)
            last_idx = global_idx
    return np.array(confirmed, dtype=int)


def _hold_last_value(known_from_idx, values, n_samples):
    """Zero-order hold: value[k] becomes visible starting at sample
    known_from_idx[k] and stays until the next one -- the causal analogue of
    interpolating between events (a real-time system can't interpolate
    towards a future event it hasn't seen yet, only hold the last one)."""
    out = np.full(n_samples, np.nan)
    if len(known_from_idx) == 0:
        return out
    idx = np.arange(n_samples)
    pos = np.searchsorted(known_from_idx, idx, side='right') - 1
    valid = pos >= 0
    out[valid] = np.asarray(values)[pos[valid]]
    return out


class OnlineFeatureExtractor(BatchFeatureExtractor):
    """Online/real-time-safe extractor producing the same 51-feature schema
    as BatchFeatureExtractor. See module docstring for the approximations
    made relative to NeuroKit2's offline algorithms."""

    # ------------------------------------------------------------------
    # EDA
    # ------------------------------------------------------------------
    def _eda_derive_signals(self, gsr, fs):
        clean = _causal_filter(gsr, fs, 'low', 3.0, order=4)
        tonic = _causal_filter(clean, fs, 'low', 0.05, order=1)
        phasic = clean - tonic
        onset_times, amplitude, rise_time, recovery_time = self._eda_detect_scrs(phasic, fs)
        return tonic, phasic, onset_times, amplitude, rise_time, recovery_time

    def _eda_detect_scrs(self, phasic, fs, calib_s=30, k_amplitude=3.0,
                          rise_start_frac=0.3, decline_confirm_s=0.3):
        """Causal SCR onset/peak/amplitude/rise-time/recovery-time detection
        via a slope-threshold state machine (see module docstring). The
        amplitude threshold is calibrated from the session's own first
        `calib_s` seconds of phasic signal, since raw sensor scale isn't
        comparable across sessions (see PIPELINE.md discussion)."""
        n = len(phasic)
        calib_n = max(2, min(n, int(calib_s * fs)))
        baseline_std = np.std(phasic[:calib_n])
        amplitude_min = max(k_amplitude * baseline_std, 1e-9)
        rise_start_thresh = rise_start_frac * baseline_std
        decline_confirm = max(1, int(decline_confirm_s * fs))

        trough_val, trough_idx = phasic[0], 0
        in_rise = False
        cand_val, cand_idx = -np.inf, 0
        decline_count = 0
        onsets = []  # each: [onset_idx, peak_idx, amplitude, rise_time, recovery_time, half_level, recovered]

        for i in range(1, n):
            v = phasic[i]
            if not in_rise:
                if v < trough_val:
                    trough_val, trough_idx = v, i
                elif v - trough_val > rise_start_thresh:
                    in_rise = True
                    cand_val, cand_idx = v, i
                    decline_count = 0
            else:
                if v > cand_val:
                    cand_val, cand_idx = v, i
                    decline_count = 0
                else:
                    decline_count += 1
                    if decline_count >= decline_confirm:
                        amp = cand_val - trough_val
                        if amp >= amplitude_min:
                            onsets.append([trough_idx, cand_idx, amp,
                                          (cand_idx - trough_idx) / fs, np.nan,
                                          cand_val - 0.5 * amp, False])
                        in_rise = False
                        trough_val, trough_idx = v, i

            for rec in onsets:
                if not rec[6] and i > rec[1] and v <= rec[5]:
                    rec[4] = (i - rec[1]) / fs
                    rec[6] = True

        if not onsets:
            empty = np.array([])
            return empty, empty, empty, empty
        onsets = np.array([[o[0], o[2], o[3], o[4]] for o in onsets])
        onset_times = onsets[:, 0] / fs
        amplitude = onsets[:, 1]
        rise_time = onsets[:, 2]
        recovery_time = onsets[:, 3]
        return onset_times, amplitude, rise_time, recovery_time

    # ------------------------------------------------------------------
    # Cardiac (PPG)
    # ------------------------------------------------------------------
    def _cardiac_derive_signals(self, ppg, fs):
        clean = _causal_filter(ppg, fs, 'band', (0.5, 8.0), order=3)
        peak_idx = _detect_events_online(clean, fs, min_distance_s=0.4, lookback_s=15)
        peak_times = peak_idx / fs

        hr = np.full(len(ppg), np.nan)
        quality = np.full(len(ppg), np.nan)
        if len(peak_times) >= 2:
            ibi = np.diff(peak_times)
            inst_hr = 60.0 / ibi
            known_from = peak_idx[1:]
            hr = _hold_last_value(known_from, inst_hr, len(ppg))

            # Quality proxy: how close each beat's IBI is to the recent
            # (last 8 beats) median IBI -- not NeuroKit2's template-matching
            # SQI, but a reasonable causal regularity proxy (see module
            # docstring).
            quality_vals = np.ones(len(ibi))
            for k in range(len(ibi)):
                lo = max(0, k - 8)
                recent_median = np.median(ibi[lo:k + 1])
                if recent_median > 0:
                    quality_vals[k] = 1.0 - min(abs(ibi[k] - recent_median) / recent_median, 1.0)
            quality = _hold_last_value(known_from, quality_vals, len(ppg))

        return hr, quality, clean, peak_times

    # ------------------------------------------------------------------
    # Respiratory
    # ------------------------------------------------------------------
    def _respiratory_derive_signals(self, resp, fs):
        clean = _causal_filter(resp, fs, 'band', (0.05, 3.0), order=3)
        # height_z=1.2 (vs. cardiac/EDA's more permissive defaults): a plain
        # 0.4 threshold over-detected breaths on this signal (calibrated by
        # comparing trough counts to NeuroKit2's on the same sessions -- see
        # PIPELINE.md's extractor comparison section), inflating
        # respiratory.rate_* well above physiological plausibility.
        peak_idx = _detect_events_online(clean, fs, min_distance_s=2.0, lookback_s=30, height_z=1.2)
        trough_idx = _detect_events_online(-clean, fs, min_distance_s=2.0, lookback_s=30, height_z=1.2)
        amplitude, rvt, symmetry, phase = self._respiratory_cycle_signals(
            clean, peak_idx, trough_idx, fs, len(resp))
        return amplitude, rvt, symmetry, phase, trough_idx.astype(float)

    def _respiratory_cycle_signals(self, clean, peak_idx, trough_idx, fs, n_samples):
        """Per-sample amplitude/RVT/symmetry/phase via zero-order hold from
        completed trough->peak->trough breath cycles -- a cycle's amplitude/
        RVT/symmetry only becomes known once its closing trough is observed
        (causal), then holds until the next cycle completes."""
        phase = np.full(n_samples, np.nan)
        events = sorted([(int(i), 0) for i in trough_idx] + [(int(i), 1) for i in peak_idx])
        for (i0, k0), (i1, _k1) in zip(events[:-1], events[1:]):
            phase[i0:i1] = 1.0 if k0 == 0 else 0.0
        if events:
            last_i, last_k = events[-1]
            phase[last_i:] = 1.0 if last_k == 0 else 0.0

        troughs_sorted = np.sort(trough_idx).astype(int)
        peaks_sorted = np.sort(peak_idx).astype(int)
        known_from, amp_vals, rvt_vals, sym_vals = [], [], [], []
        for k in range(len(troughs_sorted) - 1):
            t0, t1 = troughs_sorted[k], troughs_sorted[k + 1]
            candidates = peaks_sorted[(peaks_sorted > t0) & (peaks_sorted < t1)]
            if len(candidates) == 0:
                continue
            p = candidates[0]
            amp = clean[p] - min(clean[t0], clean[t1])
            cycle_s = (t1 - t0) / fs
            if cycle_s <= 0:
                continue
            known_from.append(t1)
            amp_vals.append(amp)
            rvt_vals.append(amp / cycle_s)
            sym_vals.append((p - t0) / fs / cycle_s)

        known_from = np.array(known_from, dtype=int)
        amplitude = _hold_last_value(known_from, amp_vals, n_samples)
        rvt = _hold_last_value(known_from, rvt_vals, n_samples)
        symmetry = _hold_last_value(known_from, sym_vals, n_samples)
        return amplitude, rvt, symmetry, phase
