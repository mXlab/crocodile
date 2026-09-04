import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

class EmotionFeatureExtractor:
    """
    Extract emotion-discriminative features from physiological signals.
    Optimized for 100 Hz sampling rate.
    """
    
    def __init__(self, sampling_rate=100):
        self.sr = sampling_rate

    def _safe_divide(self, numerator, denominator, default=0.0):
        """Safe division that handles zero denominator and invalid results."""
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return default
        if np.abs(result) > 1e10:
            return default
        return result

    def _validate_feature(self, value, feature_name="unknown", default=0.0):
        """Validate a feature value and replace invalid values."""
        if np.isnan(value):
            return default
        if np.isinf(value):
            return default
        if np.abs(value) > 1e10:
            return default
        return value

    def _validate_feature_dict(self, feature_dict, prefix=""):
        """Validate all features in a dictionary."""
        validated = {}
        for key, value in feature_dict.items():
            feature_name = f"{prefix}.{key}" if prefix else key
            validated[key] = self._validate_feature(value, feature_name)
        return validated

    def extract_all_features(self, eda_raw, ppg_raw, resp_raw, 
                            window_size_s=30, baseline=None):
        """Extract all features from raw physiological signals."""
        
        # Validate input lengths
        min_length = min(len(eda_raw), len(ppg_raw), len(resp_raw))
        eda_raw = np.array(eda_raw[:min_length])
        ppg_raw = np.array(ppg_raw[:min_length])
        resp_raw = np.array(resp_raw[:min_length])
        
        print(f"Processing {min_length} samples ({min_length/self.sr:.1f}s)")
        
        # Initialize feature dictionary
        features = {
            'eda': {},
            'cardiac': {},
            'respiratory': {},
            'multimodal': {}
        }
        
        # Extract features
        print("Extracting EDA features...")
        features['eda'] = self._extract_eda_features(eda_raw, window_size_s)
        
        print("Extracting cardiac features...")
        features['cardiac'] = self._extract_cardiac_features(ppg_raw, window_size_s)
        
        print("Extracting respiratory features...")
        features['respiratory'] = self._extract_respiratory_features(resp_raw, window_size_s)
        
        print("Computing multimodal features...")
        features['multimodal'] = self._compute_multimodal_features(features)
        
        # Validate all features
        features['eda'] = self._validate_feature_dict(features['eda'], 'eda')
        features['cardiac'] = self._validate_feature_dict(features['cardiac'], 'cardiac')
        features['respiratory'] = self._validate_feature_dict(features['respiratory'], 'respiratory')
        features['multimodal'] = self._validate_feature_dict(features['multimodal'], 'multimodal')

        # Normalize to baseline if provided
        if baseline is not None:
            features = self._normalize_to_baseline(features, baseline)

        return features
    
    def _extract_eda_features(self, eda_raw, window_size_s):
        """Extract EDA features"""
        
        features = {}
        
        # Filter EDA signal (0.05-5 Hz)
        filtered = self._bandpass_filter(eda_raw, 0.05, 5.0)
        
        # Decompose into SCL (tonic) and SCR (phasic)
        scl = self._lowpass_filter(filtered, cutoff=0.05)
        scr = filtered - scl
        
        # Detect SCR peaks
        scr_positive = np.maximum(scr, 0)
        peaks, properties = find_peaks(
            scr_positive, 
            prominence=0.01,
            distance=int(0.5 * self.sr)
        )
        
        # Ultra-short features (5s)
        ultra_short_window = int(5 * self.sr)
        recent_peaks = peaks[peaks >= (len(scr) - ultra_short_window)]
        features['scr_onset_count_5s'] = len(recent_peaks)
        
        if len(scr[-ultra_short_window:]) > 1:
            scr_diff = np.diff(scr[-ultra_short_window:])
            features['scr_max_rise_rate'] = np.max(scr_diff) if len(scr_diff) > 0 else 0
        else:
            features['scr_max_rise_rate'] = 0
        
        features['scr_recent_max_amplitude'] = np.max(np.abs(scr[-ultra_short_window:]))
        features['scr_recent_mean_amplitude'] = np.mean(np.abs(scr[-ultra_short_window:]))
        
        # Short features (10s)
        short_window = int(10 * self.sr)
        features['scl_trend_10s'] = self._compute_linear_slope(scl[-short_window:])
        
        recent_peaks_10s = peaks[peaks >= (len(scr) - short_window)]
        features['scr_event_count_10s'] = len(recent_peaks_10s)
        
        if len(recent_peaks_10s) > 1:
            intervals = np.diff(recent_peaks_10s) / self.sr
            features['scr_event_clustering'] = 1.0 / (np.mean(intervals) + 0.1)
        else:
            features['scr_event_clustering'] = 0
        
        features['eda_instability_10s'] = np.std(np.diff(scl[-short_window:]))
        
        # Medium features (full window)
        medium_window = min(int(window_size_s * self.sr), len(scl))
        
        features['scl_mean'] = np.mean(scl[-medium_window:])
        features['scl_median'] = np.median(scl[-medium_window:])
        features['scl_std'] = np.std(scl[-medium_window:])
        features['scl_range'] = np.ptp(scl[-medium_window:])
        
        features['scr_mean'] = np.mean(np.abs(scr[-medium_window:]))
        features['scr_std'] = np.std(scr[-medium_window:])
        
        window_peaks = peaks[peaks >= (len(scr) - medium_window)]
        features['scr_frequency'] = self._safe_divide(len(window_peaks), window_size_s)
        
        if len(window_peaks) > 0 and len(properties['prominences']) > 0:
            window_prominences = properties['prominences'][peaks >= (len(scr) - medium_window)]
            if len(window_prominences) > 0:
                features['scr_mean_peak_amplitude'] = np.mean(window_prominences)
            else:
                features['scr_mean_peak_amplitude'] = 0
        else:Ok coo
            features['scr_mean_peak_amplitude'] = 0
        
        features['scl_trend_full'] = self._compute_linear_slope(scl[-medium_window:])
        
        return features
    
    def _extract_cardiac_features(self, ppg_raw, window_size_s):
        """Extract cardiac features from PPG"""
        
        features = {}
        
        # Filter PPG
        ppg_filtered = self._bandpass_filter(ppg_raw, 0.5, 8.0)
        
        # Detect peaks
        peaks_pos, _ = find_peaks(ppg_filtered, distance=int(0.4*self.sr), prominence=0.1)
        peaks_neg, _ = find_peaks(-ppg_filtered, distance=int(0.4*self.sr), prominence=0.1)
        
        if len(peaks_pos) >= len(peaks_neg):
            peaks = peaks_pos
            ppg_troughs = peaks_neg
        else:
            peaks = peaks_neg
            ppg_troughs = peaks_pos
        
        # Compute heart rate
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / self.sr
            heart_rate = 60.0 / rr_intervals
            valid_hr = heart_rate[(heart_rate >= 40) & (heart_rate <= 200)]
            if len(valid_hr) == 0:
                valid_hr = heart_rate
            heart_rate = valid_hr
        else:
            heart_rate = np.array([])
        
        # Ultra-short features
        if len(heart_rate) > 0:
            n_samples_5s = max(1, int(5 * len(heart_rate) / (len(ppg_raw) / self.sr)))
            hr_5s = heart_rate[-n_samples_5s:] if len(heart_rate) > n_samples_5s else heart_rate
            
            features['hr_recent_max'] = np.max(hr_5s)
            features['hr_recent_spike'] = features['hr_recent_max'] - np.median(heart_rate)
            
            if len(hr_5s) > 1:
                features['hr_max_acceleration'] = np.max(np.diff(hr_5s))
            else:
                features['hr_max_acceleration'] = 0
        else:
            features['hr_recent_max'] = 0
            features['hr_recent_spike'] = 0
            features['hr_max_acceleration'] = 0
        
        # Short features
        if len(heart_rate) > 0:
            n_samples_10s = max(1, int(10 * len(heart_rate) / (len(ppg_raw) / self.sr)))
            hr_10s = heart_rate[-n_samples_10s:] if len(heart_rate) > n_samples_10s else heart_rate
            
            features['hr_trend_10s'] = self._compute_linear_slope(hr_10s)
            
            if len(hr_10s) > 1:
                features['hr_delta_10s'] = hr_10s[-1] - hr_10s[0]
            else:
                features['hr_delta_10s'] = 0
        else:
            features['hr_trend_10s'] = 0
            features['hr_delta_10s'] = 0
        
        # Medium features
        if len(heart_rate) > 0:
            features['hr_mean'] = np.mean(heart_rate)
            features['hr_median'] = np.median(heart_rate)
            features['hr_std'] = np.std(heart_rate)
            
            if len(peaks) > 1:
                rr_intervals_ms = np.diff(peaks) / self.sr * 1000
                
                features['hrv_rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals_ms)**2))
                features['hrv_sdnn'] = np.std(rr_intervals_ms)
                features['hrv_pnn50'] = self._compute_pnn50(rr_intervals_ms)
                features['hrv_cv'] = self._safe_divide(features['hrv_sdnn'], np.mean(rr_intervals_ms))
            else:
                features['hrv_rmssd'] = 0
                features['hrv_sdnn'] = 0
                features['hrv_pnn50'] = 0
                features['hrv_cv'] = 0
            
            features['hr_trend_full'] = self._compute_linear_slope(heart_rate)
        else:
            features['hr_mean'] = 0
            features['hr_median'] = 0
            features['hr_std'] = 0
            features['hrv_rmssd'] = 0
            features['hrv_sdnn'] = 0
            features['hrv_pnn50'] = 0
            features['hrv_cv'] = 0
            features['hr_trend_full'] = 0

        # ================================================================
        # Enhanced cardiac features (PPG amplitude dynamics, BPM dynamics)
        # Parallel to enhanced respiratory features
        # ================================================================

        n_beats_for_change = 5

        # Per-beat amplitudes: peak-to-trough on filtered PPG signal
        beat_amplitudes = []
        for i in range(len(peaks) - 1):
            troughs_between = ppg_troughs[
                (ppg_troughs > peaks[i]) & (ppg_troughs < peaks[i + 1])
            ]
            if len(troughs_between) > 0:
                amp = ppg_filtered[peaks[i]] - ppg_filtered[troughs_between[0]]
                beat_amplitudes.append(abs(amp))
        beat_amplitudes = np.array(beat_amplitudes) if beat_amplitudes else np.array([])

        # Beat intervals in seconds
        if len(peaks) > 1:
            beat_intervals_s = np.diff(peaks) / self.sr
            beat_bpms = 60.0 / beat_intervals_s
        else:
            beat_intervals_s = np.array([])
            beat_bpms = np.array([])

        # --- PPG amplitude rate of change (units/min over last N beats) ---
        if len(beat_amplitudes) >= 2:
            n = min(n_beats_for_change, len(beat_amplitudes))
            amp_delta = beat_amplitudes[-1] - beat_amplitudes[-n]
            n_ivl = min(n - 1, len(beat_intervals_s))
            time_span_s = np.sum(beat_intervals_s[-n_ivl:]) if n_ivl > 0 else 0
            features['ppg_amplitude_rate_of_change'] = self._safe_divide(amp_delta, time_span_s) * 60
        else:
            features['ppg_amplitude_rate_of_change'] = 0.0

        # --- PPG amplitude coefficient of variation (%) ---
        if len(beat_amplitudes) > 1:
            features['ppg_amplitude_coefficient_of_variation'] = \
                self._safe_divide(np.std(beat_amplitudes), np.mean(beat_amplitudes)) * 100
        else:
            features['ppg_amplitude_coefficient_of_variation'] = 0.0

        # --- PPG amplitude level indicator (0-1, current beat vs distribution) ---
        if len(beat_amplitudes) > 1 and np.std(beat_amplitudes) > 0:
            z = (beat_amplitudes[-1] - np.mean(beat_amplitudes)) / np.std(beat_amplitudes)
            features['ppg_amplitude_level_indicator'] = float(np.clip((z + 1) / 2, 0, 1))
        else:
            features['ppg_amplitude_level_indicator'] = 0.5

        # --- BPM rate of change (BPM/min over last N beats) ---
        if len(beat_bpms) >= 2:
            n = min(n_beats_for_change, len(beat_bpms))
            bpm_delta = beat_bpms[-1] - beat_bpms[-n]
            n_ivl = min(n - 1, len(beat_intervals_s))
            time_span_s = np.sum(beat_intervals_s[-n_ivl:]) if n_ivl > 0 else 0
            features['bpm_rate_of_change'] = self._safe_divide(bpm_delta, time_span_s) * 60
        else:
            features['bpm_rate_of_change'] = 0.0

        # --- BPM coefficient of variation (%) ---
        if len(beat_bpms) > 1:
            features['bpm_coefficient_of_variation'] = \
                self._safe_divide(np.std(beat_bpms), np.mean(beat_bpms)) * 100
        else:
            features['bpm_coefficient_of_variation'] = 0.0

        return features

    def _extract_respiratory_features(self, resp_raw, window_size_s):
        """Extract respiratory features"""
        
        features = {}
        
        # Filter respiratory signal
        filtered = self._bandpass_filter(resp_raw, 0.1, 0.5)
        
        # Detect breathing cycles
        peaks, _ = find_peaks(filtered, distance=int(1.5*self.sr))
        troughs, _ = find_peaks(-filtered, distance=int(1.5*self.sr))
        
        # Compute respiratory rate
        if len(peaks) > 1:
            breath_intervals = np.diff(peaks) / self.sr
            resp_rate = 60.0 / breath_intervals
            valid_rate = resp_rate[(resp_rate >= 5) & (resp_rate <= 40)]
            if len(valid_rate) == 0:
                valid_rate = resp_rate
            resp_rate = valid_rate
        else:
            resp_rate = np.array([])
        
        # Compute amplitude envelope
        amplitude_envelope = np.abs(filtered - np.median(filtered))
        
        # Ultra-short features
        ultra_short_window = int(5 * self.sr)
        
        features['resp_sigh_count_5s'] = self._detect_sighs(
            filtered[-ultra_short_window:],
            amplitude_envelope[-ultra_short_window:]
        )
        
        features['resp_pause_detected_5s'] = int(self._detect_respiratory_pause(
            filtered[-ultra_short_window:]
        ))
        
        features['resp_gasp_detected_5s'] = int(self._detect_gasp(
            filtered[-ultra_short_window:],
            amplitude_envelope[-ultra_short_window:]
        ))
        
        if len(amplitude_envelope) > ultra_short_window:
            features['resp_amplitude_spike_5s'] = (
                np.max(amplitude_envelope[-ultra_short_window:]) - 
                np.median(amplitude_envelope[-ultra_short_window:])
            )
        else:
            features['resp_amplitude_spike_5s'] = 0
        
        # Short features
        short_window = int(10 * self.sr)
        
        if len(resp_rate) > 0:
            n_samples_10s = max(1, int(10 * len(resp_rate) / (len(filtered) / self.sr)))
            resp_rate_10s = resp_rate[-n_samples_10s:] if len(resp_rate) > n_samples_10s else resp_rate
            
            features['resp_rate_trend_10s'] = self._compute_linear_slope(resp_rate_10s)
            
            if len(resp_rate_10s) > 1:
                features['resp_variability_cv_10s'] = self._safe_divide(np.std(resp_rate_10s), np.mean(resp_rate_10s))
            else:
                features['resp_variability_cv_10s'] = 0
        else:
            features['resp_rate_trend_10s'] = 0
            features['resp_variability_cv_10s'] = 0
        
        if len(amplitude_envelope) > short_window:
            features['resp_amplitude_trend_10s'] = self._compute_linear_slope(
                amplitude_envelope[-short_window:]
            )
            features['resp_amplitude_variability_10s'] = np.std(amplitude_envelope[-short_window:])
        else:
            features['resp_amplitude_trend_10s'] = 0
            features['resp_amplitude_variability_10s'] = 0
        
        # Medium features
        medium_window = min(int(window_size_s * self.sr), len(filtered))
        
        if len(resp_rate) > 0:
            features['resp_rate_mean'] = np.mean(resp_rate)
            features['resp_rate_median'] = np.median(resp_rate)
            features['resp_rate_std'] = np.std(resp_rate)
            features['resp_variability_cv'] = self._safe_divide(features['resp_rate_std'], features['resp_rate_mean'])
        else:
            features['resp_rate_mean'] = 0
            features['resp_rate_median'] = 0
            features['resp_rate_std'] = 0
            features['resp_variability_cv'] = 0
        
        if len(amplitude_envelope) > 0:
            amp_window = amplitude_envelope[-medium_window:] if len(amplitude_envelope) > medium_window else amplitude_envelope
            features['resp_amplitude_mean'] = np.mean(amp_window)
            features['resp_amplitude_median'] = np.median(amp_window)
            features['resp_amplitude_std'] = np.std(amp_window)
            features['resp_amplitude_range'] = np.ptp(amp_window)
        else:
            features['resp_amplitude_mean'] = 0
            features['resp_amplitude_median'] = 0
            features['resp_amplitude_std'] = 0
            features['resp_amplitude_range'] = 0
        
        features['resp_sigh_frequency'] = self._safe_divide(
            self._detect_sighs(filtered[-medium_window:], amplitude_envelope[-medium_window:]),
            window_size_s
        )
        
        if len(resp_rate) > 0:
            features['resp_rate_trend_full'] = self._compute_linear_slope(resp_rate)
        else:
            features['resp_rate_trend_full'] = 0
        
        if len(amplitude_envelope) > 0:
            amp_window = amplitude_envelope[-medium_window:] if len(amplitude_envelope) > medium_window else amplitude_envelope
            features['resp_amplitude_trend_full'] = self._compute_linear_slope(amp_window)
        else:
            features['resp_amplitude_trend_full'] = 0

        # ================================================================
        # Enhanced respiratory features (rate of change, CV, phase)
        # Based on BioData library + Luana Belinsky MX2403FR research
        # Uses adaptive MinMax normalization for robust breath detection
        # ================================================================

        n_breaths_for_change = 5
        min_breath_distance = int(1.5 * self.sr)

        # Adaptive normalization for breath detection (BioData MinMax approach)
        resp_normalized = self._adaptive_minmax_normalize(resp_raw)

        # Detect breaths on normalized signal
        enh_peaks, _ = find_peaks(resp_normalized, distance=min_breath_distance, height=0.5)
        enh_troughs, _ = find_peaks(-resp_normalized, distance=min_breath_distance, height=-0.5)

        # Per-breath amplitudes: peak-to-trough for each breath cycle
        breath_amplitudes = []
        for i in range(len(enh_peaks) - 1):
            troughs_between = enh_troughs[(enh_troughs > enh_peaks[i]) & (enh_troughs < enh_peaks[i + 1])]
            if len(troughs_between) > 0:
                amp = resp_normalized[enh_peaks[i]] - resp_normalized[troughs_between[0]]
                breath_amplitudes.append(abs(amp))
        breath_amplitudes = np.array(breath_amplitudes) if breath_amplitudes else np.array([])

        # Breath intervals and per-breath RPM
        if len(enh_peaks) > 1:
            breath_intervals_s = np.diff(enh_peaks) / self.sr
            breath_rpms = 60.0 / breath_intervals_s
        else:
            breath_intervals_s = np.array([])
            breath_rpms = np.array([])

        # --- Amplitude rate of change (units/min over last N breaths) ---
        if len(breath_amplitudes) >= 2:
            n = min(n_breaths_for_change, len(breath_amplitudes))
            amp_delta = breath_amplitudes[-1] - breath_amplitudes[-n]
            n_ivl = min(n - 1, len(breath_intervals_s))
            time_span_s = np.sum(breath_intervals_s[-n_ivl:]) if n_ivl > 0 else 0
            features['resp_amplitude_rate_of_change'] = self._safe_divide(amp_delta, time_span_s) * 60
        else:
            features['resp_amplitude_rate_of_change'] = 0.0

        # --- Amplitude coefficient of variation (%) ---
        if len(breath_amplitudes) > 1:
            features['resp_amplitude_coefficient_of_variation'] = \
                self._safe_divide(np.std(breath_amplitudes), np.mean(breath_amplitudes)) * 100
        else:
            features['resp_amplitude_coefficient_of_variation'] = 0.0

        # --- Amplitude level indicator (0-1, current breath vs distribution) ---
        if len(breath_amplitudes) > 1 and np.std(breath_amplitudes) > 0:
            z = (breath_amplitudes[-1] - np.mean(breath_amplitudes)) / np.std(breath_amplitudes)
            features['resp_amplitude_level_indicator'] = float(np.clip((z + 1) / 2, 0, 1))
        else:
            features['resp_amplitude_level_indicator'] = 0.5

        # --- RPM rate of change (RPM/min over last N breaths) ---
        if len(breath_rpms) >= 2:
            n = min(n_breaths_for_change, len(breath_rpms))
            rpm_delta = breath_rpms[-1] - breath_rpms[-n]
            n_ivl = min(n - 1, len(breath_intervals_s))
            time_span_s = np.sum(breath_intervals_s[-n_ivl:]) if n_ivl > 0 else 0
            features['resp_rpm_rate_of_change'] = self._safe_divide(rpm_delta, time_span_s) * 60
        else:
            features['resp_rpm_rate_of_change'] = 0.0

        # --- RPM coefficient of variation (%) ---
        if len(breath_rpms) > 1:
            features['resp_rpm_coefficient_of_variation'] = \
                self._safe_divide(np.std(breath_rpms), np.mean(breath_rpms)) * 100
        else:
            features['resp_rpm_coefficient_of_variation'] = 0.0

        # --- RPM level indicator (0-1, current RPM vs distribution) ---
        if len(breath_rpms) > 1 and np.std(breath_rpms) > 0:
            z = (breath_rpms[-1] - np.mean(breath_rpms)) / np.std(breath_rpms)
            features['resp_rpm_level_indicator'] = float(np.clip((z + 1) / 2, 0, 1))
        else:
            features['resp_rpm_level_indicator'] = 0.5

        # --- Phase features: exhale ratio + current phase ---
        # Between trough→peak = exhaling, peak→trough = inhaling
        if len(enh_peaks) > 0 and len(enh_troughs) > 0:
            events = [(p, 'peak') for p in enh_peaks] + [(t, 'trough') for t in enh_troughs]
            events.sort(key=lambda x: x[0])
            exhale_samples = 0
            inhale_samples = 0
            for i in range(len(events) - 1):
                seg_len = events[i + 1][0] - events[i][0]
                if events[i][1] == 'trough':
                    exhale_samples += seg_len
                else:
                    inhale_samples += seg_len
            total_phase = exhale_samples + inhale_samples
            features['resp_exhale_ratio'] = self._safe_divide(exhale_samples, total_phase, default=0.5)
            features['resp_currently_exhaling'] = float(events[-1][1] == 'trough')
        else:
            features['resp_exhale_ratio'] = 0.5
            features['resp_currently_exhaling'] = 0.0

        return features
    
    def _compute_multimodal_features(self, features):
        """Compute composite features"""
        
        multimodal = {}
        
        # Arousal index
        arousal_components = []
        if features['eda']['scl_mean'] != 0:
            arousal_components.append(0.4 * self._safe_normalize(features['eda']['scl_mean']))
        if features['cardiac']['hr_mean'] != 0:
            arousal_components.append(0.3 * self._safe_normalize(features['cardiac']['hr_mean']))
        if features['respiratory']['resp_rate_mean'] != 0:
            arousal_components.append(0.3 * self._safe_normalize(features['respiratory']['resp_rate_mean']))
        
        multimodal['arousal_index'] = np.mean(arousal_components) if arousal_components else 0
        
        # Regulation index
        reg_hrv = features['cardiac'].get('hrv_rmssd', 0)
        reg_resp_cv = features['respiratory']['resp_variability_cv']
        reg_scr_freq = features['eda']['scr_frequency']
        
        multimodal['regulation_index'] = (
            0.4 * self._safe_normalize(reg_hrv) +
            0.3 * (1.0 - self._safe_normalize(reg_resp_cv)) +
            0.3 * (1.0 - self._safe_normalize(reg_scr_freq))
        )
        
        # Instability index
        multimodal['instability_index'] = (
            0.4 * self._safe_normalize(features['respiratory']['resp_variability_cv']) +
            0.3 * self._safe_normalize(features['eda']['eda_instability_10s']) +
            0.3 * self._safe_normalize(features['cardiac']['hr_std'])
        )
        
        # Valence proxy
        multimodal['valence_proxy'] = (
            -0.3 * np.clip(features['cardiac']['hr_trend_10s'], -1, 1) +
            0.4 * np.clip(features['respiratory']['resp_amplitude_mean'] / 10, -1, 1) +
            -0.3 * np.clip(features['eda']['scl_trend_10s'], -1, 1)
        )
        
        # Event detected
        multimodal['event_detected'] = int(
            features['eda']['scr_onset_count_5s'] > 0 or
            features['respiratory']['resp_pause_detected_5s'] > 0 or
            features['respiratory']['resp_gasp_detected_5s'] > 0
        )
        
        return multimodal
    
    # Helper functions
    def _bandpass_filter(self, signal, lowcut, highcut, order=4):
        nyq = 0.5 * self.sr
        low = max(0.001, min(lowcut / nyq, 0.999))
        high = max(0.001, min(highcut / nyq, 0.999))
        if low >= high:
            high = low + 0.1
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)
    
    def _lowpass_filter(self, signal, cutoff, order=4):
        nyq = 0.5 * self.sr
        normal_cutoff = max(0.001, min(cutoff / nyq, 0.999))
        b, a = butter(order, normal_cutoff, btype='low')
        return filtfilt(b, a, signal)
    
    def _compute_linear_slope(self, signal):
        if len(signal) < 2:
            return 0
        x = np.arange(len(signal))
        slope, _ = np.polyfit(x, signal, 1)
        return slope
    
    def _compute_pnn50(self, rr_intervals):
        if len(rr_intervals) < 2:
            return 0
        diff_rr = np.abs(np.diff(rr_intervals))
        return 100.0 * np.sum(diff_rr > 50) / len(diff_rr)
    
    def _detect_sighs(self, resp_signal, amplitude_envelope):
        if len(amplitude_envelope) < self.sr:
            return 0
        
        baseline_amplitude = np.median(amplitude_envelope)
        threshold = 2.0 * baseline_amplitude
        
        peaks, _ = find_peaks(amplitude_envelope, height=threshold, distance=int(2*self.sr))
        
        sigh_count = 0
        for peak_idx in peaks:
            if peak_idx + int(1.0 * self.sr) < len(resp_signal):
                post_peak = resp_signal[peak_idx:peak_idx + int(1.0*self.sr)]
                slope = self._compute_linear_slope(post_peak)
                if slope < -0.005:
                    sigh_count += 1
        
        return sigh_count
    
    def _detect_respiratory_pause(self, resp_signal, duration_threshold_s=1.0):
        if len(resp_signal) < self.sr:
            return False
        
        variation = np.abs(np.diff(resp_signal))
        pause_threshold = np.std(variation) * 0.15
        
        is_pause = variation < pause_threshold
        
        pause_start = None
        for i, pause_state in enumerate(is_pause):
            if pause_state and pause_start is None:
                pause_start = i
            elif not pause_state and pause_start is not None:
                pause_duration = (i - pause_start) / self.sr
                if pause_duration >= duration_threshold_s:
                    return True
                pause_start = None
        
        return False
    
    def _detect_gasp(self, resp_signal, amplitude_envelope):
        if len(amplitude_envelope) < 10:
            return False
        
        amplitude_diff = np.diff(amplitude_envelope)
        threshold = np.std(amplitude_diff) * 2.5
        
        gasps = amplitude_diff > threshold
        return np.any(gasps)
    
    def _adaptive_minmax_normalize(self, signal):
        """
        Adaptive MinMax normalization from BioData library (MinMax.h).
        Min/Max bounds adapt slowly to track signal drift.
        Output is in [0, 1].
        """
        adaptation_rate = 0.05
        adapt_factor = adaptation_rate ** 2

        normalized = np.zeros(len(signal), dtype=float)
        current_min = float(signal[0])
        current_max = float(signal[0])

        for i in range(len(signal)):
            value = float(signal[i])
            if value > current_max:
                current_max = value
            if value < current_min:
                current_min = value
            current_min += (value - current_min) * adapt_factor
            current_max += (value - current_max) * adapt_factor
            if current_max == current_min:
                normalized[i] = 0.5
            else:
                normalized[i] = (value - current_min) / (current_max - current_min)

        return normalized

    def _safe_normalize(self, value, scale=100.0):
        return np.clip(self._safe_divide(value, scale), 0, 1)
    
    def _normalize_to_baseline(self, features, baseline):
        normalized = {}
        
        for modality in features:
            normalized[modality] = {}
            for feature_name, value in features[modality].items():
                if modality in baseline and feature_name in baseline[modality]:
                    baseline_mean = baseline[modality][feature_name]['mean']
                    baseline_std = baseline[modality][feature_name]['std']
                    
                    normalized[modality][feature_name] = self._safe_divide(
                        value - baseline_mean, baseline_std
                    )
                else:
                    normalized[modality][feature_name] = value
        
        return normalized
    
    def features_to_vector(self, features):
        vector = []
        feature_names = []
        
        for modality in ['eda', 'cardiac', 'respiratory', 'multimodal']:
            if modality in features:
                for feature_name in sorted(features[modality].keys()):
                    vector.append(features[modality][feature_name])
                    feature_names.append(f"{modality}.{feature_name}")
        
        return np.array(vector), feature_names
