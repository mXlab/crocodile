"""Online (real-time-safe) counterpart to BatchFeatureExtractor -- computes
the SAME NeuroKit2-schema features (see batch_feature_extractor.py) but
derives the underlying physiological signals (EDA tonic/phasic, cardiac
rate/quality, respiratory amplitude/RVT/symmetry/phase) using only past and
current samples, never NeuroKit2's offline whole-session algorithms
(cvxEDA, elgendi PPG cleaning, RSP_Rate from full-session peak detection).

Two ways to call it:
- `process_session(df)`: one-shot, offline replay of a whole recorded
  session -- for training-data generation and comparison against
  BatchFeatureExtractor (see scripts/compare_extractors.py). Matches
  BatchFeatureExtractor's signature so the two are drop-in alternatives.
- `calibrate(calibration_df)` + repeated `push(chunk_df)`: genuine
  incremental/live use -- feed new raw samples as they arrive (any chunk
  size), get back zero or more newly-finalized one-second feature rows.
  `calibrate()` is optional and primes filter state + the SCR amplitude
  threshold from a separate calibration recording (see PIPELINE.md's
  exhibition-calibration-period discussion) instead of the live session's
  own first 30s.

`process_session()` is implemented ON TOP OF `calibrate()`/`push()` (not a
separate parallel implementation) -- see PIPELINE.md's "Live-readiness"
section for why: one code path guarantees offline-replay numbers and live
behavior can never drift apart, and fixes a real "backdating" bug that used
to exist in one-shot processing (a row's features could depend on samples
that arrived after that row's own timestamp -- caught by feeding the same
extractor a session prefix and the full session and finding the overlapping
rows didn't match exactly, which they must for a genuinely causal system).
`push()`'s discipline -- advance every stateful component by exactly one
new second's worth of samples, then immediately finalize and never revisit
that row -- is what fixes this, whether called from a live loop or from
`process_session()` handing over a whole array at once.

A note on what "online-safe" means for the implementation below: some
signal-level steps use `scipy.signal.sosfilt` with persisted filter state
(`zi`) across calls rather than a literal sample-by-sample Python loop.
This is NOT a shortcut that leaks future data -- `sosfilt` is a strictly
causal recursive (IIR) filter, so sample i of its output depends only on
samples 0..i of the input; carrying `zi` across incremental calls produces
exactly the output a true streaming implementation would produce
sample-by-sample. Peak/trough/SCR-onset detection, by contrast, genuinely
needs a forward-only algorithm (a resumable state machine, or a fixed
lookback-window rescan repeated every second) since `scipy.signal.find_peaks`
run on a whole array would leak future context.

Known approximations vs. NeuroKit2 (documented so the comparison script's
results are interpretable, not just "worse"):
- EDA tonic/phasic split uses a causal lowpass/highpass filter pair instead
  of cvxEDA's global convex optimization -- the same category of method as
  NeuroKit2's own `eda_phasic(method="highpass")` alternative to cvxEDA.
- SCR onset/peak/amplitude/rise-time/recovery-time detection is a bespoke
  slope-threshold state machine (trough -> sustained rise -> confirmed
  decline), calibrated from a calibration recording's phasic signal (or the
  live session's own first 30s if no separate calibration is given), not
  NeuroKit2's `eda_findpeaks`.
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

Explicitly out of scope (see PIPELINE.md): bounded-memory ring buffers
(growing arrays are fine for realistic exhibition session lengths --
minutes, not hours) and wiring to actual live sensor hardware (this module
only makes the extractor itself callable incrementally; a real acquisition
loop calling `push()` is separate future work).
"""

import warnings

import numpy as np
import pandas as pd
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

DEFAULT_SIGNAL_COLS = {'eda': 'gsr', 'ppg': 'heart', 'resp': 'respiration'}


# ---------------------------------------------------------------------------
# Stateful, incremental signal-derivation primitives
# ---------------------------------------------------------------------------

class _CausalFilterState:
    """Persistent causal IIR filter state across incremental pushes.
    `.extend(new_chunk)` filters only the new samples (via sosfilt's `zi`
    carryover) and returns them -- the caller appends to its own growing
    output array. Steady-state init (`zi * x[0]`, not the default all-zero
    state) happens on the very first chunk it ever sees: with zero-init, a
    slow lowpass (e.g. the 0.05Hz tonic filter) takes tens of seconds to
    ramp up from 0 to the signal's actual DC level, producing a huge
    spurious cold-start transient in the phasic residual (measured: first-
    30s phasic std ~195 vs. a true in-session ~30-60, large enough to blow
    out any fixed SCR amplitude threshold)."""

    def __init__(self, fs, btype, cutoff, order):
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')
        self.zi = None

    def extend(self, new_chunk):
        new_chunk = np.asarray(new_chunk, dtype=float)
        if len(new_chunk) == 0:
            return np.array([])
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos) * new_chunk[0]
        y, self.zi = sosfilt(self.sos, new_chunk, zi=self.zi)
        return y


class _ScrDetectorState:
    """Resumable causal SCR onset/peak/amplitude/rise-time/recovery-time
    detector via a slope-threshold state machine (trough -> sustained rise
    -> confirmed decline). `.extend(new_phasic_chunk)` continues the state
    machine from wherever it left off, processing only the new samples --
    the whole point being this is O(new samples), not O(session so far),
    unlike calling the equivalent whole-array loop fresh every time.

    The amplitude threshold is calibrated once, from the first `calib_s`
    seconds of whatever is pushed through `.extend()` -- buffered until
    enough samples arrive, then replayed through the state machine now that
    the threshold is known (`set_threshold_from()` does the actual
    computation). Whether those first `calib_s` seconds come from a
    dedicated calibration recording or the live session's own start is
    entirely up to the caller -- OnlineFeatureExtractor.calibrate() is just
    a `push()` call over calibration data with the returned rows discarded,
    not a separate priming path, so this buffer-then-compute logic is the
    only place the threshold ever gets set either way."""

    def __init__(self, fs, calib_s=30, k_amplitude=3.0, rise_start_frac=0.3,
                 decline_confirm_s=0.3):
        self.fs = fs
        self.calib_n = max(2, int(calib_s * fs))
        self.k_amplitude = k_amplitude
        self.rise_start_frac = rise_start_frac
        self.decline_confirm = max(1, int(decline_confirm_s * fs))

        self.amplitude_min = None
        self.rise_start_thresh = None
        self._calib_buffer = []

        self._initialized = False
        self.trough_val = None
        self.trough_idx = 0
        self.in_rise = False
        self.cand_val = -np.inf
        self.cand_idx = 0
        self.decline_count = 0
        self.next_i = 0
        # each: [onset_idx, peak_idx, amplitude, rise_time, recovery_time, half_level, recovered]
        self.onsets = []

    def set_threshold_from(self, phasic_calibration_array):
        """Explicit calibration from a separate recording (see
        OnlineFeatureExtractor.calibrate())."""
        baseline_std = np.std(phasic_calibration_array)
        self.amplitude_min = max(self.k_amplitude * baseline_std, 1e-9)
        self.rise_start_thresh = self.rise_start_frac * baseline_std

    def extend(self, new_phasic_chunk):
        chunk = np.asarray(new_phasic_chunk, dtype=float)
        if len(chunk) == 0:
            return
        if self.amplitude_min is None:
            self._calib_buffer.extend(chunk.tolist())
            if len(self._calib_buffer) < self.calib_n:
                return  # still buffering -- nothing to process yet
            calib_array = np.array(self._calib_buffer)
            self.set_threshold_from(calib_array)
            self._calib_buffer = None
            self._process_chunk(calib_array)
        else:
            self._process_chunk(chunk)

    def _process_chunk(self, chunk):
        for v in chunk:
            i = self.next_i
            if not self._initialized:
                self.trough_val, self.trough_idx = v, i
                self._initialized = True
                self.next_i += 1
                continue

            if not self.in_rise:
                if v < self.trough_val:
                    self.trough_val, self.trough_idx = v, i
                elif v - self.trough_val > self.rise_start_thresh:
                    self.in_rise = True
                    self.cand_val, self.cand_idx = v, i
                    self.decline_count = 0
            else:
                if v > self.cand_val:
                    self.cand_val, self.cand_idx = v, i
                    self.decline_count = 0
                else:
                    self.decline_count += 1
                    if self.decline_count >= self.decline_confirm:
                        amp = self.cand_val - self.trough_val
                        if amp >= self.amplitude_min:
                            self.onsets.append([self.trough_idx, self.cand_idx, amp,
                                                (self.cand_idx - self.trough_idx) / self.fs,
                                                np.nan, self.cand_val - 0.5 * amp, False])
                        self.in_rise = False
                        self.trough_val, self.trough_idx = v, i

            for rec in self.onsets:
                if not rec[6] and i > rec[1] and v <= rec[5]:
                    rec[4] = (i - rec[1]) / self.fs
                    rec[6] = True

            self.next_i += 1

    def as_arrays(self):
        if not self.onsets:
            empty = np.array([])
            return empty, empty, empty, empty
        arr = np.array([[o[0], o[2], o[3], o[4]] for o in self.onsets])
        onset_times = arr[:, 0] / self.fs
        return onset_times, arr[:, 1], arr[:, 2], arr[:, 3]


class _PeakDetectorState:
    """Resumable causal peak detector: `.step(signal_so_far, end)` does
    exactly ONE bounded lookback-window rescan (z-scores the last
    `lookback_s` seconds ending at `end` and re-runs find_peaks on it),
    accepting only newly-seen peaks that already have `confirm_margin_s` of
    decline after them (so a peak sitting right at the current sample isn't
    accepted before it's actually been seen to decline -- the real-time
    equivalent of "you can't confirm a peak until it's past"). `last_idx`
    persists across calls so each `.step()` only does O(lookback_s) work,
    not O(session so far) -- the caller must pass `signal_so_far` truncated
    to exactly `end` samples; this never looks beyond what it's given. Used
    for both cardiac PPG peaks and respiratory peaks/troughs (pass a
    `_PeakDetectorState` fed `-signal` for troughs)."""

    def __init__(self, fs, min_distance_s, lookback_s, height_z=0.4, confirm_margin_s=0.2):
        self.lookback = int(lookback_s * fs)
        self.min_distance = max(1, int(min_distance_s * fs))
        self.height_z = height_z
        self.confirm_margin = int(confirm_margin_s * fs)
        self.last_idx = -self.min_distance
        self.confirmed = []

    def step(self, signal_so_far, end):
        start = max(0, end - self.lookback)
        window = signal_so_far[start:end]
        if len(window) < 2 * self.min_distance:
            return
        sigma = window.std()
        if sigma == 0:
            return
        z = (window - window.mean()) / sigma
        pk, _ = find_peaks(z, distance=self.min_distance, height=self.height_z)
        for p in pk:
            global_idx = start + p
            if global_idx <= self.last_idx or (end - global_idx) < self.confirm_margin:
                continue
            self.confirmed.append(global_idx)
            self.last_idx = global_idx


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
    """Online/real-time-safe extractor producing the same feature schema as
    BatchFeatureExtractor. See module docstring for the calibrate()/push()
    live API, process_session()'s relationship to it, and the
    approximations made relative to NeuroKit2's offline algorithms."""

    def __init__(self, sampling_rate: int = 100):
        super().__init__(sampling_rate)
        self._reset_state()

    def _reset_state(self):
        fs = self.sampling_rate
        self._eda_clean_filter = _CausalFilterState(fs, 'low', 3.0, order=4)
        self._eda_tonic_filter = _CausalFilterState(fs, 'low', 0.05, order=1)
        self._ppg_filter = _CausalFilterState(fs, 'band', (0.5, 8.0), order=3)
        self._resp_filter = _CausalFilterState(fs, 'band', (0.05, 3.0), order=3)

        self._gsr_clean = np.array([])
        self._tonic = np.array([])
        self._phasic = np.array([])
        self._ppg_clean = np.array([])
        self._resp_clean = np.array([])

        self._scr_detector = _ScrDetectorState(fs)
        self._cardiac_peak_detector = _PeakDetectorState(fs, min_distance_s=0.4, lookback_s=15)
        # height_z=1.2 (vs. cardiac/EDA's more permissive defaults): a plain
        # 0.4 threshold over-detected breaths on this signal (calibrated by
        # comparing trough counts to NeuroKit2's on the same sessions -- see
        # PIPELINE.md's extractor comparison section), inflating
        # respiratory.rate_* well above physiological plausibility.
        self._resp_peak_detector = _PeakDetectorState(fs, min_distance_s=2.0, lookback_s=30, height_z=1.2)
        self._resp_trough_detector = _PeakDetectorState(fs, min_distance_s=2.0, lookback_s=30, height_z=1.2)

        self._pending = {'gsr': [], 'ppg': [], 'resp': [], 'emotion': [], 'feeling_it': []}
        self._n_rows_emitted = 0
        self._feature_interval_s = 1.0
        self._calibrated = False

    # ------------------------------------------------------------------
    # Live API
    # ------------------------------------------------------------------
    def calibrate(self, calibration_df: pd.DataFrame, signal_cols: dict = None):
        """Prime filter state, the SCR amplitude threshold, and cardiac/
        respiratory peak-detector history from a separate calibration
        recording, instead of using the live session's own first 30s
        (process_session()'s default when this isn't called). Emits no
        feature rows to the caller -- only readies state. Call once, before
        the first `push()`.

        Implemented as a plain `push()` call over the calibration data with
        the returned rows discarded -- not a separate, bespoke priming path.
        An earlier version only extended the causal filters here, which left
        the cardiac/respiratory peak detectors never having been stepped
        through the calibration period at all: their first live-triggered
        `.step()` call would catch only the last `lookback_s` seconds of
        calibration in one lookback rescan, not the properly incremental,
        one-row-at-a-time history `push()` builds. That mismatch showed up
        as real discrepancies in HRV/HR features between calibrate()+push()
        and an equivalent process_session() call -- caught by
        scripts/test_online_causality.py's calibrate-then-push consistency
        check. Delegating to push() here removes the possibility of the two
        paths diverging again."""
        if signal_cols is None:
            signal_cols = DEFAULT_SIGNAL_COLS
        self.push(calibration_df, signal_cols=signal_cols, feature_interval_s=1.0)
        self._calibrated = True

    def recalibrate(self):
        """Reset just the EDA/SCR amplitude threshold so the next `calib_s`
        seconds of pushed phasic data recompute it -- the same buffer-then-
        compute logic `_ScrDetectorState.set_threshold_from()` already uses
        on first calibration, just re-armed. Lets a long-running live
        session adapt to EDA baseline drift (temperature, sweat, electrode
        contact) without discarding the extractor entirely.

        Only the SCR threshold needs this: the causal filters (IIR state
        carried forward forever) and the cardiac/respiratory peak detectors
        (rolling lookback-window re-scan on every `.step()`) are already
        continuously self-adapting and don't need an explicit reset. The
        SCR detector's rise/decline state machine and onset history are
        untouched here -- only `amplitude_min`/`rise_start_thresh` and the
        re-buffering flag reset, so no detector state or in-flight SCR
        event is lost.

        A naive rolling threshold (recomputed continuously from a trailing
        window) was deliberately not used instead: it would fold real SCR
        events into its own baseline-noise estimate, inflating the
        threshold right after a genuine response burst -- a feedback loop
        that suppresses detection when it matters most. Calling this
        on-demand, ideally during another calm moment, avoids that."""
        self._scr_detector.amplitude_min = None
        self._scr_detector.rise_start_thresh = None
        self._scr_detector._calib_buffer = []

    def push(self, chunk_df: pd.DataFrame, signal_cols: dict = None,
             feature_interval_s: float = 1.0) -> list:
        """Feed new raw samples (any chunk size). Returns a list of zero or
        more newly-finalized one-second feature-row dicts. Each row is
        finalized using state that has seen precisely up to that row's own
        sample boundary -- never more, even if this call's chunk contains
        samples belonging to later rows too -- which is what keeps this
        causally exact rather than an approximation of it (see module
        docstring's note on the "backdating" fix)."""
        if signal_cols is None:
            signal_cols = DEFAULT_SIGNAL_COLS
        self._feature_interval_s = feature_interval_s
        fs = self.sampling_rate
        step = int(feature_interval_s * fs)

        self._pending['gsr'].extend(chunk_df[signal_cols['eda']].values.astype(float).tolist())
        self._pending['ppg'].extend(chunk_df[signal_cols['ppg']].values.astype(float).tolist())
        self._pending['resp'].extend(chunk_df[signal_cols['resp']].values.astype(float).tolist())
        has_emotion = 'emotion' in chunk_df.columns
        has_feeling_it = 'feeling_it' in chunk_df.columns
        if has_emotion:
            self._pending['emotion'].extend(chunk_df['emotion'].values.tolist())
        if has_feeling_it:
            self._pending['feeling_it'].extend(chunk_df['feeling_it'].values.tolist())

        rows = []
        while len(self._pending['gsr']) >= step:
            gsr_chunk = np.array(self._pending['gsr'][:step]); self._pending['gsr'] = self._pending['gsr'][step:]
            ppg_chunk = np.array(self._pending['ppg'][:step]); self._pending['ppg'] = self._pending['ppg'][step:]
            resp_chunk = np.array(self._pending['resp'][:step]); self._pending['resp'] = self._pending['resp'][step:]
            emotion_val = None
            if self._pending['emotion']:
                emotion_val = self._pending['emotion'][0]
                self._pending['emotion'] = self._pending['emotion'][step:]
            feeling_it_val = None
            if self._pending['feeling_it']:
                feeling_it_val = self._pending['feeling_it'][0]
                self._pending['feeling_it'] = self._pending['feeling_it'][step:]

            rows.append(self._finalize_row(gsr_chunk, ppg_chunk, resp_chunk, fs, step,
                                            emotion_val, feeling_it_val))
            self._n_rows_emitted += 1

        return rows

    def _finalize_row(self, gsr_chunk, ppg_chunk, resp_chunk, fs, step, emotion_val, feeling_it_val):
        # 1. Extend causal filters by exactly this new chunk.
        new_gsr_clean = self._eda_clean_filter.extend(gsr_chunk)
        self._gsr_clean = np.concatenate([self._gsr_clean, new_gsr_clean])
        new_tonic = self._eda_tonic_filter.extend(new_gsr_clean)
        self._tonic = np.concatenate([self._tonic, new_tonic])
        new_phasic = new_gsr_clean - new_tonic
        self._phasic = np.concatenate([self._phasic, new_phasic])

        new_ppg_clean = self._ppg_filter.extend(ppg_chunk)
        self._ppg_clean = np.concatenate([self._ppg_clean, new_ppg_clean])
        new_resp_clean = self._resp_filter.extend(resp_chunk)
        self._resp_clean = np.concatenate([self._resp_clean, new_resp_clean])

        end = len(self._gsr_clean)  # == (t+1)*step, t = self._n_rows_emitted

        # 2. Extend event detectors by exactly this new chunk / up to this new end.
        self._scr_detector.extend(new_phasic)
        self._cardiac_peak_detector.step(self._ppg_clean, end)
        self._resp_peak_detector.step(self._resp_clean, end)
        self._resp_trough_detector.step(-self._resp_clean, end)

        # 3. Build the (small, event-count-bounded) derived arrays the
        # shared one-row aggregate functions need. Recomputed fresh each
        # row from the confirmed-events-so-far lists -- O(n_events), not
        # O(session samples), so this is cheap regardless of session length.
        onset_times, scr_amplitude, rise_time, recovery_time = self._scr_detector.as_arrays()

        peak_idx = np.array(self._cardiac_peak_detector.confirmed, dtype=int)
        peak_times = peak_idx / fs
        hr = np.full(end, np.nan)
        quality = np.full(end, np.nan)
        if len(peak_times) >= 2:
            ibi = np.diff(peak_times)
            inst_hr = 60.0 / ibi
            known_from = peak_idx[1:]
            hr = _hold_last_value(known_from, inst_hr, end)
            quality_vals = np.ones(len(ibi))
            for k in range(len(ibi)):
                lo = max(0, k - 8)
                recent_median = np.median(ibi[lo:k + 1])
                if recent_median > 0:
                    quality_vals[k] = 1.0 - min(abs(ibi[k] - recent_median) / recent_median, 1.0)
            quality = _hold_last_value(known_from, quality_vals, end)

        resp_peak_idx = np.array(self._resp_peak_detector.confirmed, dtype=int)
        resp_trough_idx = np.array(self._resp_trough_detector.confirmed, dtype=int)
        amplitude, rvt, symmetry, phase = self._respiratory_cycle_signals(
            self._resp_clean, resp_peak_idx, resp_trough_idx, fs, end)

        # 4. One-row aggregate functions (BatchFeatureExtractor, inherited).
        # _respiratory_prepare de-duplicates troughs internally (same as the
        # batch path) -- pass the raw confirmed-troughs list, not pre-deduped.
        t = self._n_rows_emitted
        ibi_full, pulse_amplitude, pulse_amp_times = self._cardiac_prepare(self._ppg_clean, peak_times, fs)
        rate, breath_times, breath_intervals, breath_depths = \
            self._respiratory_prepare(end, amplitude, resp_trough_idx.astype(float), fs)

        row = {'timestamp': t * self._feature_interval_s, 'sample_idx': t * step}
        row.update(self._eda_aggregate_one_row(self._tonic, self._phasic, onset_times, scr_amplitude,
                                                rise_time, recovery_time, fs, t, step))
        row.update(self._cardiac_aggregate_one_row(hr, quality, peak_times, ibi_full, pulse_amplitude,
                                                    pulse_amp_times, fs, t, step))
        row.update(self._respiratory_aggregate_one_row(rate, amplitude, rvt, symmetry, phase,
                                                         breath_times, breath_intervals,
                                                         breath_depths, fs, t, step))
        if emotion_val is not None:
            row['emotion'] = emotion_val
        if feeling_it_val is not None:
            row['feeling_it'] = feeling_it_val
        return row

    def _respiratory_cycle_signals(self, clean, peak_idx, trough_idx, fs, n_samples):
        """Per-sample amplitude/RVT/symmetry/phase via zero-order hold from
        completed trough->peak->trough breath cycles -- a cycle's amplitude/
        RVT/symmetry only becomes known once its closing trough is observed
        (causal), then holds until the next cycle completes. Recomputed
        fresh from the full peak/trough index lists each row -- O(n_cycles),
        not O(session samples), so cheap regardless of session length."""
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

    # ------------------------------------------------------------------
    # Offline replay, implemented on top of the live API (see module docstring)
    # ------------------------------------------------------------------
    def process_session(self, session_df: pd.DataFrame, feature_interval_s: float = 1.0,
                        signal_cols: dict = None) -> pd.DataFrame:
        """Extract features for one whole recorded session by replaying it
        through push() in one call. This is NOT a separate implementation --
        push()'s row-by-row discipline (advance state by exactly one new
        row's worth of samples, finalize, never revisit) applies here
        exactly as it would to genuinely live data arriving incrementally,
        even though the whole array is technically already in memory. That
        discipline is what fixes the backdating bug this used to have (see
        module docstring)."""
        self._reset_state()
        if signal_cols is None:
            signal_cols = DEFAULT_SIGNAL_COLS
        rows = self.push(session_df, signal_cols=signal_cols, feature_interval_s=feature_interval_s)
        return pd.DataFrame(rows)
