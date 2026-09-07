"""Batch (offline) feature extractor -- NeuroKit2-based alternative to
EnhancedContinuousFeatureExtractor.

Why this exists as a SEPARATE module (see PIPELINE.md / session discussion):
continuous_feature_extractor.py was deliberately built as an online,
never-resetting streaming processor -- it computes each second's features
using only that second's and earlier samples, never looking ahead -- so the
same code could eventually run in a real-time installation. That constraint
has two costs when used for OFFLINE training data (which is all it's used
for today -- the live installation doesn't exist yet):

1. Cold-start artifacts: online filters/rolling windows have no history at
   the start of a recording, producing statistically unstable (sometimes
   extreme) values in the first several seconds. A batch processor sees the
   whole session at once and has no such warm-up period.
2. EDA specifically has no raw-signal normalization at all (unlike cardiac,
   which z-normalizes its input, and respiratory, which adaptively rescales
   to an observed min/max) -- confirmed by direct code inspection. EDA
   amplitude is not comparable across sessions/days as a result: ambient
   temperature, electrode contact, and skin hydration all shift the raw
   microsiemens scale with no correction.

This module fixes both by using NeuroKit2's validated offline algorithms
(cvxEDA convex-optimization phasic/tonic decomposition, proper PPG/RSP peak
detection) run once over a session's full raw signal, then computes
per-second aggregate features from the result -- no filter state, no
look-only-backward constraint (each second's features can use the whole
recording, past and future).

Explicitly NOT a replacement: continuous_feature_extractor.py is untouched
and still the one to use if/when a real-time-compatible extractor is needed.
This module is for offline analysis and training data only. Output goes to
continuous_features_batch.csv (via extract_continuous_features_batch.py),
never overwriting continuous_features.csv, so both remain available to
compare directly.

Normalization is deliberately NOT baked into feature computation here
(unlike the original extractor's *_normalized_*/*_scaled_* patterns) --
these features are in raw physical units (BPM, microsiemens, breaths/min,
seconds). Per-session vs. pooled-across-sessions normalization is a modeling
decision, applied downstream (e.g. in Stage 5), not a feature-extraction one.
"""

import numpy as np
import pandas as pd
import neurokit2 as nk


class BatchFeatureExtractor:
    """Offline feature extractor using NeuroKit2 -- sees the whole session's
    signal at once (past and future samples both available), unlike the
    online/real-time-compatible EnhancedContinuousFeatureExtractor.

    Unlike EnhancedContinuousFeatureExtractor, this has no persistent filter
    state across calls -- process_session() is a pure function of the whole
    session's raw signal.
    """

    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate

    def process_session(self, session_df: pd.DataFrame, feature_interval_s: float = 1.0,
                        signal_cols: dict = None) -> pd.DataFrame:
        """Extract batch features for one session.

        Parameters mirror EnhancedContinuousFeatureExtractor.process_session
        so the two are drop-in alternatives from the calling script's side.
        """
        if signal_cols is None:
            signal_cols = {'eda': 'gsr', 'ppg': 'heart', 'resp': 'respiration'}

        fs = self.sampling_rate
        gsr = session_df[signal_cols['eda']].values.astype(float)
        ppg = session_df[signal_cols['ppg']].values.astype(float)
        resp = session_df[signal_cols['resp']].values.astype(float)
        n_seconds = int(len(gsr) // fs)
        step = int(feature_interval_s * fs)

        eda_feats = self._eda_features(gsr, fs, n_seconds, step)
        cardiac_feats = self._cardiac_features(ppg, fs, n_seconds, step)
        resp_feats = self._respiratory_features(resp, fs, n_seconds, step)

        n_rows = n_seconds
        out = pd.DataFrame({
            'timestamp': np.arange(n_rows) * feature_interval_s,
            'sample_idx': np.arange(n_rows) * step,
        })
        for feats in (eda_feats, cardiac_feats, resp_feats):
            for col, values in feats.items():
                out[col] = values[:n_rows]

        # Pass through emotion/feeling_it the same way the row-per-second
        # cadence expects (one label per feature_interval_s block).
        if 'emotion' in session_df.columns:
            idx = np.arange(n_rows) * step
            idx = np.clip(idx, 0, len(session_df) - 1)
            out['emotion'] = session_df['emotion'].values[idx]
        if 'feeling_it' in session_df.columns:
            idx = np.arange(n_rows) * step
            idx = np.clip(idx, 0, len(session_df) - 1)
            out['feeling_it'] = session_df['feeling_it'].values[idx]

        return out

    # ------------------------------------------------------------------
    # EDA
    # ------------------------------------------------------------------
    def _eda_features(self, gsr, fs, n_seconds, step):
        tonic, phasic, onset_times, amplitude, rise_time, recovery_time = \
            self._eda_derive_signals(gsr, fs)
        return self._eda_aggregate(tonic, phasic, onset_times, amplitude,
                                    rise_time, recovery_time, fs, n_seconds, step)

    def _eda_derive_signals(self, gsr, fs):
        """Offline tonic/phasic decomposition + SCR event detection via
        NeuroKit2 (cvxEDA + eda_findpeaks) -- sees the whole session at once.
        Override point for an online-safe equivalent (see
        OnlineFeatureExtractor)."""
        signals, info = nk.eda_process(gsr, sampling_rate=fs)
        tonic = signals['EDA_Tonic'].values
        phasic = signals['EDA_Phasic'].values
        onset_idx = np.asarray(info['SCR_Onsets'], dtype=float)
        onset_times = onset_idx / fs
        amplitude = np.asarray(info['SCR_Amplitude'], dtype=float)
        rise_time = np.asarray(info['SCR_RiseTime'], dtype=float)
        recovery_time = np.asarray(info['SCR_RecoveryTime'], dtype=float)
        return tonic, phasic, onset_times, amplitude, rise_time, recovery_time

    def _eda_aggregate(self, tonic, phasic, onset_times, amplitude, rise_time,
                        recovery_time, fs, n_seconds, step):
        """Per-second windowed EDA features from already-derived tonic/phasic/
        SCR-event signals. Only ever reads tonic/phasic/onset_times up to the
        current second `t` (via `end`) -- this loop is itself online-safe
        regardless of how the input signals were derived.
        """
        # First derivative of the phasic signal, for an instability measure
        # analogous to the old extractor's eda_instability_10s.
        phasic_deriv = np.diff(phasic, prepend=phasic[0])

        row_times = np.arange(n_seconds)
        feats = {
            'eda.tonic_level': np.zeros(n_seconds),
            'eda.tonic_std_10s': np.zeros(n_seconds),
            'eda.tonic_range_10s': np.zeros(n_seconds),
            'eda.tonic_trend_10s': np.zeros(n_seconds),
            'eda.tonic_trend_full': np.zeros(n_seconds),
            'eda.phasic_mean_10s': np.zeros(n_seconds),
            'eda.phasic_std_10s': np.zeros(n_seconds),
            'eda.instability_10s': np.zeros(n_seconds),
            'eda.scr_rate_60s': np.zeros(n_seconds),
            'eda.scr_event_count_10s': np.zeros(n_seconds),
            'eda.scr_recent_max_amplitude_5s': np.zeros(n_seconds),
            'eda.scr_event_clustering_60s': np.zeros(n_seconds),
            'eda.seconds_since_onset': np.zeros(n_seconds),
            'eda.last_onset_amplitude': np.zeros(n_seconds),
            'eda.last_onset_risetime': np.zeros(n_seconds),
            'eda.last_onset_recoverytime': np.zeros(n_seconds),
            'eda.mean_onset_amplitude_full': np.zeros(n_seconds),
        }

        # idx_of_last_onset[t] = index into onset_times of the most recent
        # onset at or before second t (-1 if none yet)
        idx_of_last = np.searchsorted(onset_times, row_times, side='right') - 1

        for t in row_times:
            end = min((t + 1) * step, len(tonic))
            start_5s = max(0, end - 5 * fs)
            start_10s = max(0, end - 10 * fs)
            start_60s = max(0, end - 60 * fs)

            feats['eda.tonic_level'][t] = tonic[start_10s:end].mean() if end > start_10s else tonic[0]
            feats['eda.tonic_std_10s'][t] = tonic[start_10s:end].std() if end > start_10s else 0.0
            feats['eda.tonic_range_10s'][t] = np.ptp(tonic[start_10s:end]) if end > start_10s else 0.0
            feats['eda.phasic_mean_10s'][t] = phasic[start_10s:end].mean() if end > start_10s else 0.0
            feats['eda.phasic_std_10s'][t] = phasic[start_10s:end].std() if end > start_10s else 0.0
            feats['eda.instability_10s'][t] = phasic_deriv[start_10s:end].var() if end > start_10s else 0.0

            if end - start_10s > fs:
                x = np.arange(start_10s, end)
                feats['eda.tonic_trend_10s'][t] = np.polyfit(x, tonic[start_10s:end], 1)[0]
            if end > 30 * fs:  # require >=30s of history for a stable full-session trend
                x = np.arange(0, end)
                feats['eda.tonic_trend_full'][t] = np.polyfit(x, tonic[0:end], 1)[0]

            onsets_60s = onset_times[(onset_times >= start_60s / fs) & (onset_times <= t)]
            feats['eda.scr_rate_60s'][t] = len(onsets_60s)
            onsets_10s = onset_times[(onset_times >= start_10s / fs) & (onset_times <= t)]
            feats['eda.scr_event_count_10s'][t] = len(onsets_10s)
            onsets_5s_mask = (onset_times >= start_5s / fs) & (onset_times <= t)
            if onsets_5s_mask.any():
                feats['eda.scr_recent_max_amplitude_5s'][t] = np.nanmax(amplitude[onsets_5s_mask])
            if len(onsets_60s) >= 3:
                inter_onset = np.diff(onsets_60s)
                cv = inter_onset.std() / inter_onset.mean() if inter_onset.mean() > 0 else 0.0
                feats['eda.scr_event_clustering_60s'][t] = 1.0 - min(cv, 1.0)

            i = int(idx_of_last[t])
            if i >= 0:
                feats['eda.seconds_since_onset'][t] = t - onset_times[i]
                feats['eda.last_onset_amplitude'][t] = amplitude[i]
                feats['eda.last_onset_risetime'][t] = rise_time[i] if not np.isnan(rise_time[i]) else 0.0
                feats['eda.last_onset_recoverytime'][t] = recovery_time[i] if not np.isnan(recovery_time[i]) else 0.0
                feats['eda.mean_onset_amplitude_full'][t] = np.nanmean(amplitude[:i + 1])
            else:
                feats['eda.seconds_since_onset'][t] = t  # no onset yet -> time since session start

        return feats

    # ------------------------------------------------------------------
    # Cardiac (PPG)
    # ------------------------------------------------------------------
    def _cardiac_features(self, ppg, fs, n_seconds, step):
        hr, quality, clean, peak_times = self._cardiac_derive_signals(ppg, fs)
        return self._cardiac_aggregate(hr, quality, clean, peak_times, fs, n_seconds, step)

    def _cardiac_derive_signals(self, ppg, fs):
        """Offline cleaning, peak detection, instantaneous rate, and quality
        index via NeuroKit2 -- sees the whole session at once. Override point
        for an online-safe equivalent (see OnlineFeatureExtractor)."""
        signals, info = nk.ppg_process(ppg, sampling_rate=fs)
        hr = signals['PPG_Rate'].values
        quality = signals['PPG_Quality'].values
        clean = signals['PPG_Clean'].values
        peaks = np.asarray(info['PPG_Peaks'], dtype=float)
        peak_times = peaks / fs
        return hr, quality, clean, peak_times

    def _cardiac_aggregate(self, hr, quality, clean, peak_times, fs, n_seconds, step):
        """Per-second windowed cardiac features from already-derived hr/
        quality/clean/peak_times. Only ever reads these up to the current
        second `t` (via `end`) -- online-safe regardless of how the inputs
        were derived."""
        ibi = np.diff(peak_times)  # inter-beat intervals, seconds
        hr_sample_deltas = np.abs(np.diff(hr))  # sample-to-sample HR change, for max_acceleration

        # PPG waveform amplitude (peak-to-trough of the pulse itself, distinct
        # from heart RATE) -- not given directly by nk.ppg_process, so find
        # the trough preceding each peak in the cleaned signal.
        peaks_int = np.round(peak_times * fs).astype(int)
        pulse_amplitude = np.full(len(peak_times), np.nan)
        pulse_amp_times = peak_times.copy()
        for j, p in enumerate(peaks_int):
            lo = int(peaks_int[j - 1]) if j > 0 else max(0, p - int(1.5 * fs))
            segment = clean[lo:p + 1]
            if len(segment):
                pulse_amplitude[j] = clean[p] - segment.min()

        row_times = np.arange(n_seconds)
        feats = {
            'cardiac.hr_mean_10s': np.zeros(n_seconds),
            'cardiac.hr_std_10s': np.zeros(n_seconds),
            'cardiac.hr_median_10s': np.zeros(n_seconds),
            'cardiac.hr_trend_10s': np.zeros(n_seconds),
            'cardiac.hr_trend_full': np.zeros(n_seconds),
            'cardiac.hr_delta_10s': np.zeros(n_seconds),
            'cardiac.hr_recent_max_10s': np.zeros(n_seconds),
            'cardiac.hr_recent_spike_10s': np.zeros(n_seconds),
            'cardiac.hr_max_acceleration_full': np.zeros(n_seconds),
            'cardiac.quality_mean_10s': np.zeros(n_seconds),
            'cardiac.hrv_sdnn_60s': np.zeros(n_seconds),
            'cardiac.hrv_rmssd_60s': np.zeros(n_seconds),
            'cardiac.hrv_pnn50_60s': np.zeros(n_seconds),
            'cardiac.hrv_cv_60s': np.zeros(n_seconds),
            'cardiac.bpm_cv_60s': np.zeros(n_seconds),
            'cardiac.ppg_amplitude_mean_10s': np.zeros(n_seconds),
            'cardiac.ppg_amplitude_cv_10s': np.zeros(n_seconds),
        }

        for t in row_times:
            end = min((t + 1) * step, len(hr))
            start_10s = max(0, end - 10 * fs)
            start_60s = max(0, end - 60 * fs)

            window_10s = hr[start_10s:end]
            feats['cardiac.hr_mean_10s'][t] = np.nanmean(window_10s) if len(window_10s) else np.nan
            feats['cardiac.hr_std_10s'][t] = np.nanstd(window_10s) if len(window_10s) else 0.0
            feats['cardiac.hr_median_10s'][t] = np.nanmedian(window_10s) if len(window_10s) else np.nan
            feats['cardiac.quality_mean_10s'][t] = np.nanmean(quality[start_10s:end]) if end > start_10s else 0.0

            if end - start_10s > fs:
                x = np.arange(start_10s, end)
                valid = ~np.isnan(hr[start_10s:end])
                if valid.sum() > 2:
                    feats['cardiac.hr_trend_10s'][t] = np.polyfit(x[valid], hr[start_10s:end][valid], 1)[0]
            if end > 30 * fs:  # require >=30s of history for a stable full-session trend
                x = np.arange(0, end)
                valid = ~np.isnan(hr[0:end])
                if valid.sum() > 2:
                    feats['cardiac.hr_trend_full'][t] = np.polyfit(x[valid], hr[0:end][valid], 1)[0]

            if start_10s > fs:
                feats['cardiac.hr_delta_10s'][t] = np.nan_to_num(hr[end - 1] - hr[start_10s])

            if len(window_10s) and not np.all(np.isnan(window_10s)):
                recent_max = np.nanmax(window_10s)
                feats['cardiac.hr_recent_max_10s'][t] = recent_max
                session_median_so_far = np.nanmedian(hr[:end]) if end > 0 else np.nan
                feats['cardiac.hr_recent_spike_10s'][t] = recent_max - session_median_so_far

            if end > 30 * fs:
                feats['cardiac.hr_max_acceleration_full'][t] = np.nanmax(hr_sample_deltas[:end - 1]) \
                    if end - 1 > 0 and not np.all(np.isnan(hr_sample_deltas[:end - 1])) else 0.0

            window_60s = hr[start_60s:end]
            valid_60 = window_60s[~np.isnan(window_60s)]
            if len(valid_60) > 1 and valid_60.mean() != 0:
                feats['cardiac.bpm_cv_60s'][t] = valid_60.std() / valid_60.mean() * 100

            ibi_mask = (peak_times[1:] >= start_60s / fs) & (peak_times[1:] <= t)
            recent_ibi = ibi[ibi_mask]
            if len(recent_ibi) >= 3:
                feats['cardiac.hrv_sdnn_60s'][t] = recent_ibi.std() * 1000  # ms
                diffs_ms = np.diff(recent_ibi) * 1000
                feats['cardiac.hrv_rmssd_60s'][t] = np.sqrt(np.mean(diffs_ms ** 2))
                feats['cardiac.hrv_pnn50_60s'][t] = (np.abs(diffs_ms) > 50).mean() * 100
                feats['cardiac.hrv_cv_60s'][t] = recent_ibi.std() / recent_ibi.mean() * 100 if recent_ibi.mean() > 0 else 0.0

            amp_mask = (pulse_amp_times >= start_10s / fs) & (pulse_amp_times <= t)
            recent_amp = pulse_amplitude[amp_mask]
            recent_amp = recent_amp[~np.isnan(recent_amp)]
            if len(recent_amp) > 1 and recent_amp.mean() != 0:
                feats['cardiac.ppg_amplitude_mean_10s'][t] = recent_amp.mean()
                feats['cardiac.ppg_amplitude_cv_10s'][t] = recent_amp.std() / recent_amp.mean() * 100

        return feats

    # ------------------------------------------------------------------
    # Respiratory
    # ------------------------------------------------------------------
    def _respiratory_features(self, resp, fs, n_seconds, step):
        amplitude, rvt, symmetry, phase, troughs = self._respiratory_derive_signals(resp, fs)
        return self._respiratory_aggregate(len(resp), amplitude, rvt, symmetry, phase,
                                            troughs, fs, n_seconds, step)

    def _respiratory_derive_signals(self, resp, fs):
        """Offline cleaning, amplitude/RVT/symmetry/phase, and trough
        detection via NeuroKit2 -- sees the whole session at once. Override
        point for an online-safe equivalent (see OnlineFeatureExtractor).
        Trough de-duplication (below) is intentionally shared with any
        override -- it's a generic safety net, not NeuroKit-specific."""
        signals, info = nk.rsp_process(resp, sampling_rate=fs)
        amplitude = signals['RSP_Amplitude'].values
        rvt = signals['RSP_RVT'].values
        symmetry = signals['RSP_Symmetry_RiseDecay'].values
        # RSP_Phase: 1 during inhale, 0 during exhale (NeuroKit2 convention),
        # NaN before the first detected cycle.
        phase = signals['RSP_Phase'].values
        troughs = np.asarray(info['RSP_Troughs'], dtype=float)
        return amplitude, rvt, symmetry, phase, troughs

    def _respiratory_aggregate(self, n_samples, amplitude, rvt, symmetry, phase,
                                troughs, fs, n_seconds, step):
        """Per-second windowed respiratory features from already-derived
        amplitude/RVT/symmetry/phase/troughs. Only ever reads these up to the
        current second `t` (via `end`) -- online-safe regardless of how the
        inputs were derived.
        """
        # Trough detection (NeuroKit2's default, or any override) can
        # over-fire on noisy stretches of this signal -- e.g. troughs 0.3-0.6s
        # apart (100-200 breaths/min, not physiologically possible) observed
        # in a noisy segment. De-duplicate any trough within min_interval_s of
        # the previous KEPT one (~40/min ceiling, matching the plausibility-
        # bound pattern already used elsewhere in this codebase for cardiac
        # R-R and breath intervals), then rebuild the rate signal from the
        # cleaned troughs rather than trusting any upstream rate estimate
        # (which would be derived from the same over-detected troughs). This
        # does NOT fix the opposite failure mode (a missed breath inflating
        # one interval, seen separately) -- that needs smarter re-detection,
        # not de-duplication, and remains a known residual limitation.
        min_interval_s = 1.5
        if len(troughs) > 1:
            keep = [troughs[0]]
            for tr in troughs[1:]:
                if (tr - keep[-1]) / fs >= min_interval_s:
                    keep.append(tr)
            troughs = np.array(keep)

        breath_times = troughs / fs
        breath_intervals = np.diff(breath_times)
        if len(breath_times) >= 2:
            inst_rate = 60.0 / breath_intervals
            rate = np.interp(np.arange(n_samples), troughs[1:], inst_rate,
                             left=inst_rate[0], right=inst_rate[-1])
        else:
            rate = np.full(n_samples, np.nan)

        # Depth of each individual breath cycle (amplitude sampled at each
        # trough), for sigh/spike detection -- distinct from the continuous
        # amplitude signal used for the *_mean_10s features above.
        trough_idx = troughs.astype(int)
        breath_depths = amplitude[trough_idx] if len(trough_idx) else np.array([])

        row_times = np.arange(n_seconds)
        feats = {
            'respiratory.rate_mean_10s': np.zeros(n_seconds),
            'respiratory.rate_median_10s': np.zeros(n_seconds),
            'respiratory.rate_std_10s': np.zeros(n_seconds),
            'respiratory.rate_trend_10s': np.zeros(n_seconds),
            'respiratory.rate_trend_full': np.zeros(n_seconds),
            'respiratory.amplitude_mean_10s': np.zeros(n_seconds),
            'respiratory.amplitude_median_10s': np.zeros(n_seconds),
            'respiratory.amplitude_std_10s': np.zeros(n_seconds),
            'respiratory.amplitude_cv_10s': np.zeros(n_seconds),
            'respiratory.amplitude_range_10s': np.zeros(n_seconds),
            'respiratory.amplitude_spike_5s': np.zeros(n_seconds),
            'respiratory.rvt_mean_10s': np.zeros(n_seconds),
            'respiratory.symmetry_risedecay_mean_10s': np.zeros(n_seconds),
            'respiratory.exhale_ratio_10s': np.zeros(n_seconds),
            'respiratory.sigh_count_5s': np.zeros(n_seconds),
            'respiratory.sigh_frequency_5s': np.zeros(n_seconds),
            'respiratory.pause_detected_5s': np.zeros(n_seconds),
            'respiratory.gasp_detected_5s': np.zeros(n_seconds),
            'respiratory.cv_60s': np.zeros(n_seconds),
        }

        for t in row_times:
            end = min((t + 1) * step, len(rate))
            start_5s = max(0, end - 5 * fs)
            start_10s = max(0, end - 10 * fs)
            start_60s = max(0, end - 60 * fs)

            feats['respiratory.rate_mean_10s'][t] = np.nanmean(rate[start_10s:end]) if end > start_10s else np.nan
            feats['respiratory.rate_median_10s'][t] = np.nanmedian(rate[start_10s:end]) if end > start_10s else np.nan
            feats['respiratory.rate_std_10s'][t] = np.nanstd(rate[start_10s:end]) if end > start_10s else 0.0
            feats['respiratory.amplitude_mean_10s'][t] = np.nanmean(amplitude[start_10s:end]) if end > start_10s else np.nan
            feats['respiratory.amplitude_median_10s'][t] = np.nanmedian(amplitude[start_10s:end]) if end > start_10s else np.nan
            feats['respiratory.amplitude_std_10s'][t] = np.nanstd(amplitude[start_10s:end]) if end > start_10s else 0.0
            amp_window = amplitude[start_10s:end]
            amp_valid = amp_window[~np.isnan(amp_window)]
            if len(amp_valid) > 1 and amp_valid.mean() != 0:
                feats['respiratory.amplitude_cv_10s'][t] = amp_valid.std() / amp_valid.mean() * 100
            feats['respiratory.amplitude_range_10s'][t] = np.ptp(amplitude[start_10s:end]) if end > start_10s else 0.0
            feats['respiratory.rvt_mean_10s'][t] = np.nanmean(rvt[start_10s:end]) if end > start_10s else np.nan
            feats['respiratory.symmetry_risedecay_mean_10s'][t] = np.nanmean(symmetry[start_10s:end]) if end > start_10s else np.nan

            phase_window = phase[start_10s:end]
            valid_phase = phase_window[~np.isnan(phase_window)]
            if len(valid_phase):
                feats['respiratory.exhale_ratio_10s'][t] = (valid_phase == 0).mean()

            if end - start_10s > fs:
                x = np.arange(start_10s, end)
                valid = ~np.isnan(rate[start_10s:end])
                if valid.sum() > 2:
                    feats['respiratory.rate_trend_10s'][t] = np.polyfit(x[valid], rate[start_10s:end][valid], 1)[0]
            if end > 30 * fs:  # require >=30s of history for a stable full-session trend
                x = np.arange(0, end)
                valid = ~np.isnan(rate[0:end])
                if valid.sum() > 2:
                    feats['respiratory.rate_trend_full'][t] = np.polyfit(x[valid], rate[0:end][valid], 1)[0]

            interval_mask = (breath_times[1:] >= start_60s / fs) & (breath_times[1:] <= t)
            recent_intervals = breath_intervals[interval_mask]
            if len(recent_intervals) > 1 and recent_intervals.mean() != 0:
                feats['respiratory.cv_60s'][t] = recent_intervals.std() / recent_intervals.mean() * 100

            # Sigh / pause / gasp: compare recent breaths to the session's
            # own distribution so far (mirrors the original extractor's
            # mean +/- k*std plausibility-bound pattern).
            depths_so_far_mask = breath_times <= t
            depths_so_far = breath_depths[depths_so_far_mask]
            if len(depths_so_far) >= 5:
                # nanmean/nanstd (not plain mean/std): breath_depths can carry
                # a single leading NaN when amplitude is only known from a
                # cycle's *closing* trough onward (true for the online
                # extractor -- its very first trough precedes any completed
                # cycle). A plain .mean() would let that one NaN poison
                # d_mean/d_std -- and therefore amplitude_spike_5s -- for the
                # rest of the session. NeuroKit2's own amplitude signal
                # rarely has this gap, so this is a no-op for the batch path.
                d_mean, d_std = np.nanmean(depths_so_far), np.nanstd(depths_so_far)
                recent_depth_mask = (breath_times >= start_5s / fs) & (breath_times <= t)
                recent_depths = breath_depths[recent_depth_mask]
                n_sighs = int((recent_depths > d_mean + 2 * d_std).sum())
                feats['respiratory.sigh_count_5s'][t] = n_sighs
                feats['respiratory.sigh_frequency_5s'][t] = n_sighs / 5.0
                if len(depths_so_far) >= 3:
                    feats['respiratory.amplitude_spike_5s'][t] = recent_depths[-3:].max() - d_mean \
                        if len(recent_depths) else 0.0

            intervals_so_far_mask = breath_times[1:] <= t
            intervals_so_far = breath_intervals[intervals_so_far_mask]
            if len(intervals_so_far) >= 5:
                i_mean, i_std = intervals_so_far.mean(), intervals_so_far.std()
                recent_interval_mask = (breath_times[1:] >= start_5s / fs) & (breath_times[1:] <= t)
                recent_int = breath_intervals[recent_interval_mask]
                if len(recent_int):
                    feats['respiratory.pause_detected_5s'][t] = float(np.any(recent_int > i_mean + 2 * i_std))
                    feats['respiratory.gasp_detected_5s'][t] = float(
                        np.any((recent_int < i_mean - 1.5 * i_std) & (recent_int > 1.0)))

        return feats
