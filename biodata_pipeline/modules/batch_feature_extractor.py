"""Batch (offline) feature extractor -- NeuroKit2-based alternative to
EnhancedContinuousFeatureExtractor.

Why this exists as a SEPARATE module (see PIPELINE.md / session discussion):
continuous_feature_extractor.py was deliberately built as a causal, never-
resetting streaming processor so the same code could eventually run in a
real-time installation. That constraint has two costs when used for OFFLINE
training data (which is all it's used for today -- the runtime pipeline
doesn't exist yet):

1. Cold-start artifacts: causal filters/rolling windows have no history at
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
online/causal constraint.

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
    """Offline, non-causal feature extractor using NeuroKit2.

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
        signals, info = nk.eda_process(gsr, sampling_rate=fs)
        tonic = signals['EDA_Tonic'].values
        phasic = signals['EDA_Phasic'].values
        onset_idx = np.asarray(info['SCR_Onsets'], dtype=float)
        onset_times = onset_idx / fs
        amplitude = np.asarray(info['SCR_Amplitude'], dtype=float)
        rise_time = np.asarray(info['SCR_RiseTime'], dtype=float)
        recovery_time = np.asarray(info['SCR_RecoveryTime'], dtype=float)

        row_times = np.arange(n_seconds)
        feats = {
            'eda.tonic_level': np.zeros(n_seconds),
            'eda.tonic_std_10s': np.zeros(n_seconds),
            'eda.tonic_trend_10s': np.zeros(n_seconds),
            'eda.tonic_trend_full': np.zeros(n_seconds),
            'eda.phasic_mean_10s': np.zeros(n_seconds),
            'eda.phasic_std_10s': np.zeros(n_seconds),
            'eda.scr_rate_60s': np.zeros(n_seconds),
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
            start_10s = max(0, end - 10 * fs)
            start_60s = max(0, end - 60 * fs)

            feats['eda.tonic_level'][t] = tonic[start_10s:end].mean() if end > start_10s else tonic[0]
            feats['eda.tonic_std_10s'][t] = tonic[start_10s:end].std() if end > start_10s else 0.0
            feats['eda.phasic_mean_10s'][t] = phasic[start_10s:end].mean() if end > start_10s else 0.0
            feats['eda.phasic_std_10s'][t] = phasic[start_10s:end].std() if end > start_10s else 0.0

            if end - start_10s > fs:
                x = np.arange(start_10s, end)
                feats['eda.tonic_trend_10s'][t] = np.polyfit(x, tonic[start_10s:end], 1)[0]
            if end > 30 * fs:  # require >=30s of history for a stable full-session trend
                x = np.arange(0, end)
                feats['eda.tonic_trend_full'][t] = np.polyfit(x, tonic[0:end], 1)[0]

            recent_onsets = onset_times[(onset_times >= start_60s / fs) & (onset_times <= t)]
            feats['eda.scr_rate_60s'][t] = len(recent_onsets) / 1.0  # per-60s-window count

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
        signals, info = nk.ppg_process(ppg, sampling_rate=fs)
        hr = signals['PPG_Rate'].values
        quality = signals['PPG_Quality'].values
        peaks = np.asarray(info['PPG_Peaks'], dtype=float)
        peak_times = peaks / fs
        ibi = np.diff(peak_times)  # inter-beat intervals, seconds

        row_times = np.arange(n_seconds)
        feats = {
            'cardiac.hr_mean_10s': np.zeros(n_seconds),
            'cardiac.hr_std_10s': np.zeros(n_seconds),
            'cardiac.hr_median_10s': np.zeros(n_seconds),
            'cardiac.hr_trend_10s': np.zeros(n_seconds),
            'cardiac.hr_trend_full': np.zeros(n_seconds),
            'cardiac.hr_delta_10s': np.zeros(n_seconds),
            'cardiac.quality_mean_10s': np.zeros(n_seconds),
            'cardiac.hrv_sdnn_60s': np.zeros(n_seconds),
            'cardiac.hrv_rmssd_60s': np.zeros(n_seconds),
            'cardiac.bpm_cv_60s': np.zeros(n_seconds),
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

            window_60s = hr[start_60s:end]
            valid_60 = window_60s[~np.isnan(window_60s)]
            if len(valid_60) > 1 and valid_60.mean() != 0:
                feats['cardiac.bpm_cv_60s'][t] = valid_60.std() / valid_60.mean() * 100

            ibi_mask = (peak_times[1:] >= start_60s / fs) & (peak_times[1:] <= t)
            recent_ibi = ibi[ibi_mask]
            if len(recent_ibi) >= 3:
                feats['cardiac.hrv_sdnn_60s'][t] = recent_ibi.std() * 1000  # ms
                feats['cardiac.hrv_rmssd_60s'][t] = np.sqrt(np.mean(np.diff(recent_ibi) ** 2)) * 1000

        return feats

    # ------------------------------------------------------------------
    # Respiratory
    # ------------------------------------------------------------------
    def _respiratory_features(self, resp, fs, n_seconds, step):
        signals, info = nk.rsp_process(resp, sampling_rate=fs)
        amplitude = signals['RSP_Amplitude'].values
        rvt = signals['RSP_RVT'].values
        symmetry = signals['RSP_Symmetry_RiseDecay'].values
        troughs = np.asarray(info['RSP_Troughs'], dtype=float)

        # NeuroKit2's default trough detection over-fires on noisy stretches
        # of this signal -- found troughs 0.3-0.6s apart (100-200 breaths/min,
        # not physiologically possible) in a noisy segment. De-duplicate any
        # trough within min_interval_s of the previous KEPT one (~40/min
        # ceiling, matching the plausibility-bound pattern already used
        # elsewhere in this codebase for cardiac R-R and breath intervals),
        # then rebuild the rate signal from the cleaned troughs rather than
        # trusting signals['RSP_Rate'] (which is derived from the same
        # over-detected troughs). This does NOT fix the opposite failure mode
        # (a missed breath inflating one interval, seen separately) -- that
        # needs smarter re-detection, not de-duplication, and remains a known
        # residual limitation of this prototype.
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
            rate = np.interp(np.arange(len(resp)), troughs[1:], inst_rate,
                             left=inst_rate[0], right=inst_rate[-1])
        else:
            rate = np.full(len(resp), np.nan)

        row_times = np.arange(n_seconds)
        feats = {
            'respiratory.rate_mean_10s': np.zeros(n_seconds),
            'respiratory.rate_std_10s': np.zeros(n_seconds),
            'respiratory.rate_trend_10s': np.zeros(n_seconds),
            'respiratory.rate_trend_full': np.zeros(n_seconds),
            'respiratory.amplitude_mean_10s': np.zeros(n_seconds),
            'respiratory.amplitude_std_10s': np.zeros(n_seconds),
            'respiratory.rvt_mean_10s': np.zeros(n_seconds),
            'respiratory.symmetry_risedecay_mean_10s': np.zeros(n_seconds),
            'respiratory.cv_60s': np.zeros(n_seconds),
        }

        for t in row_times:
            end = min((t + 1) * step, len(rate))
            start_10s = max(0, end - 10 * fs)
            start_60s = max(0, end - 60 * fs)

            feats['respiratory.rate_mean_10s'][t] = np.nanmean(rate[start_10s:end]) if end > start_10s else np.nan
            feats['respiratory.rate_std_10s'][t] = np.nanstd(rate[start_10s:end]) if end > start_10s else 0.0
            feats['respiratory.amplitude_mean_10s'][t] = np.nanmean(amplitude[start_10s:end]) if end > start_10s else np.nan
            feats['respiratory.amplitude_std_10s'][t] = np.nanstd(amplitude[start_10s:end]) if end > start_10s else 0.0
            feats['respiratory.rvt_mean_10s'][t] = np.nanmean(rvt[start_10s:end]) if end > start_10s else np.nan
            feats['respiratory.symmetry_risedecay_mean_10s'][t] = np.nanmean(symmetry[start_10s:end]) if end > start_10s else np.nan

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

        return feats
