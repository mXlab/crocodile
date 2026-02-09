"""
Continuous Feature Extractor

This module extracts features sample-by-sample or at regular intervals,
maintaining filter state across the entire session (not resetting at emotion boundaries).

This matches real-time deployment architecture where filters run continuously.

Key differences from EmotionFeatureExtractor:
1. Processes entire session continuously (not per-segment)
2. Maintains filter states across emotion boundaries
3. Returns per-sample or per-interval features
4. Can be sliced into windows AFTER feature extraction

Usage:
    extractor = ContinuousFeatureExtractor(sampling_rate=100)
    features_df = extractor.process_session(
        session_data,
        feature_interval_s=1.0  # Extract features every 1 second
    )
    
    # Then slice features by emotion segments (see data_slicer.slice_features_into_windows)
    from modules.data_slicer import slice_features_into_windows
    windows = slice_features_into_windows(features_df, ...)
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from collections import deque
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ContinuousFeatureExtractor:
    """
    Extracts features continuously across entire session.
    
    Maintains filter states and temporal context, matching real-time
    operation where filters never reset.
    """
    
    def __init__(self, sampling_rate: int = 100):
        """
        Initialize continuous feature extractor.
        
        Parameters
        ----------
        sampling_rate : int
            Sampling rate in Hz (default: 100)
        """
        self.sampling_rate = sampling_rate
        
        # Initialize filter states (will persist across samples)
        self.reset_states()
    
    def reset_states(self):
        """Reset all filter states (call between sessions, not within!)"""
        
        # EDA filter states
        self.eda_scl_lpf_state = 0.0
        self.eda_scr_hpf_state = 0.0
        self.eda_scr_prev_input = 0.0
        
        # Cardiac filter states
        self.hr_history = deque(maxlen=int(10 * self.sampling_rate))  # 10s history
        self.rr_intervals = deque(maxlen=20)  # Last 20 R-R intervals
        self.last_r_peak_idx = None
        
        # Respiratory filter states  
        self.resp_amplitude_history = deque(maxlen=10)  # Last 10 breaths
        self.resp_interval_history = deque(maxlen=10)
        self.last_resp_peak_idx = None
        
        # Adaptive normalization states
        self.resp_min = None
        self.resp_max = None
        self.ppg_min = None
        self.ppg_max = None
    
    def process_session(self,
                       session_df: pd.DataFrame,
                       feature_interval_s: float = 1.0,
                       signal_cols: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Process entire session continuously, extracting features at intervals.
        
        Parameters
        ----------
        session_df : pd.DataFrame
            Raw session data with columns: gsr, heart, respiration, emotion, etc.
        feature_interval_s : float
            Extract features every N seconds (default: 1.0)
        signal_cols : dict, optional
            Mapping of signal names, e.g. {'eda': 'gsr', 'ppg': 'heart', 'resp': 'respiration'}
        
        Returns
        -------
        pd.DataFrame
            Features extracted at regular intervals with timestamps and emotion labels
        """
        
        if signal_cols is None:
            signal_cols = {
                'eda': 'gsr',
                'ppg': 'heart',
                'resp': 'respiration'
            }
        
        # Reset states for new session
        self.reset_states()
        
        # Extract features at intervals
        interval_samples = int(feature_interval_s * self.sampling_rate)
        n_samples = len(session_df)
        
        features_list = []
        
        for idx in range(0, n_samples, interval_samples):
            # Get window of samples for this interval
            window_start = max(0, idx - int(30 * self.sampling_rate))  # Use 30s history
            window_end = min(n_samples, idx + interval_samples)
            
            window_data = session_df.iloc[window_start:window_end]
            
            # Extract signals
            eda_signal = window_data[signal_cols['eda']].values
            ppg_signal = window_data[signal_cols['ppg']].values
            resp_signal = window_data[signal_cols['resp']].values
            
            # Get emotion label at this timestamp
            emotion_label = session_df.iloc[idx].get('emotion', 'unknown')
            
            # Extract features (maintaining filter state!)
            features = self._extract_features_at_timestamp(
                eda_signal,
                ppg_signal,
                resp_signal,
                current_idx=len(window_data) - interval_samples
            )
            
            # Add metadata
            features['timestamp'] = idx / self.sampling_rate
            features['sample_idx'] = idx
            features['emotion'] = emotion_label
            
            # Add other metadata if available
            if 'feeling_it' in session_df.columns:
                features['feeling_it'] = session_df.iloc[idx:window_end]['feeling_it'].mean()
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _extract_features_at_timestamp(self,
                                      eda_signal: np.ndarray,
                                      ppg_signal: np.ndarray,
                                      resp_signal: np.ndarray,
                                      current_idx: int) -> Dict[str, float]:
        """
        Extract features at current timestamp using available history.
        
        Parameters
        ----------
        eda_signal : np.ndarray
            EDA signal history (including current sample)
        ppg_signal : np.ndarray  
            PPG signal history
        resp_signal : np.ndarray
            Respiration signal history
        current_idx : int
            Index of current sample within the provided signals
        
        Returns
        -------
        dict
            Feature dictionary
        """
        
        features = {}
        
        # ================================================================
        # EDA FEATURES (Continuous low-pass and high-pass filtering)
        # ================================================================
        
        eda_features = self._extract_eda_features_continuous(eda_signal, current_idx)
        features.update(eda_features)
        
        # ================================================================
        # CARDIAC FEATURES (Maintain R-R interval history)
        # ================================================================
        
        cardiac_features = self._extract_cardiac_features_continuous(ppg_signal, current_idx)
        features.update(cardiac_features)
        
        # ================================================================
        # RESPIRATORY FEATURES (Adaptive normalization across session)
        # ================================================================
        
        resp_features = self._extract_respiratory_features_continuous(resp_signal, current_idx)
        features.update(resp_features)
        
        return features
    
    def _extract_eda_features_continuous(self, eda_signal: np.ndarray, current_idx: int) -> Dict[str, float]:
        """Extract EDA features with continuous filtering."""
        
        features = {}
        
        # Use last few seconds of signal
        window_samples = min(int(10 * self.sampling_rate), len(eda_signal))
        eda_window = eda_signal[-window_samples:]
        
        # SCL (tonic): Low-pass filter (continuous state)
        # Simple exponential moving average
        alpha_scl = 0.01  # Very slow
        for sample in eda_window:
            self.eda_scl_lpf_state = alpha_scl * sample + (1 - alpha_scl) * self.eda_scl_lpf_state
        
        features['eda.scl_mean'] = self.eda_scl_lpf_state
        features['eda.scl_std'] = np.std(eda_window)
        features['eda.scl_range'] = np.ptp(eda_window)
        
        # SCR (phasic): High-pass filter (continuous state)
        alpha_scr = 0.1
        scr_signal = []
        for sample in eda_window:
            # High-pass filter
            scr = alpha_scr * (self.eda_scr_hpf_state + sample - self.eda_scr_prev_input)
            self.eda_scr_hpf_state = scr
            self.eda_scr_prev_input = sample
            scr_signal.append(max(0, scr))  # Rectify
        
        scr_signal = np.array(scr_signal)
        
        # SCR statistics
        features['eda.scr_mean'] = np.mean(scr_signal)
        features['eda.scr_std'] = np.std(scr_signal)
        features['eda.scr_max'] = np.max(scr_signal)
        
        # SCR events (peaks)
        if len(scr_signal) > 100:
            threshold = np.mean(scr_signal) + 0.5 * np.std(scr_signal)
            peaks, _ = find_peaks(scr_signal, height=threshold, distance=int(0.5 * self.sampling_rate))
            features['eda.scr_frequency'] = len(peaks) / (len(scr_signal) / self.sampling_rate)
        else:
            features['eda.scr_frequency'] = 0.0
        
        return features
    
    def _extract_cardiac_features_continuous(self, ppg_signal: np.ndarray, current_idx: int) -> Dict[str, float]:
        """Extract cardiac features maintaining R-R interval history."""
        
        features = {}
        
        # Use last 10 seconds
        window_samples = min(int(10 * self.sampling_rate), len(ppg_signal))
        ppg_window = ppg_signal[-window_samples:]
        
        # Normalize
        ppg_normalized = (ppg_window - np.mean(ppg_window)) / (np.std(ppg_window) + 1e-10)
        
        # Detect R-peaks
        peaks, _ = find_peaks(ppg_normalized, distance=int(0.4 * self.sampling_rate), height=0.5)
        
        if len(peaks) >= 2:
            # Update R-R interval history (continuous!)
            intervals_ms = np.diff(peaks) / self.sampling_rate * 1000
            for interval in intervals_ms:
                if 300 < interval < 2000:  # Valid physiological range
                    self.rr_intervals.append(interval)
            
            # Heart rate from recent peaks
            if len(peaks) > 1:
                mean_interval_s = np.mean(np.diff(peaks)) / self.sampling_rate
                hr = 60 / mean_interval_s
                features['cardiac.hr_mean'] = hr
            else:
                features['cardiac.hr_mean'] = 0.0
            
            # HRV from accumulated R-R intervals
            if len(self.rr_intervals) >= 3:
                rr_array = np.array(self.rr_intervals)
                
                # RMSSD
                rr_diff = np.diff(rr_array)
                features['cardiac.hrv_rmssd'] = np.sqrt(np.mean(rr_diff ** 2))
                
                # SDNN
                features['cardiac.hrv_sdnn'] = np.std(rr_array)
                
                # pNN50
                nn50 = np.sum(np.abs(rr_diff) > 50)
                features['cardiac.hrv_pnn50'] = (nn50 / len(rr_diff)) * 100
            else:
                features['cardiac.hrv_rmssd'] = 0.0
                features['cardiac.hrv_sdnn'] = 0.0
                features['cardiac.hrv_pnn50'] = 0.0
        else:
            features['cardiac.hr_mean'] = 0.0
            features['cardiac.hrv_rmssd'] = 0.0
            features['cardiac.hrv_sdnn'] = 0.0
            features['cardiac.hrv_pnn50'] = 0.0
        
        return features
    
    def _extract_respiratory_features_continuous(self, resp_signal: np.ndarray, current_idx: int) -> Dict[str, float]:
        """Extract respiratory features with adaptive normalization."""
        
        features = {}
        
        # Use last 30 seconds
        window_samples = min(int(30 * self.sampling_rate), len(resp_signal))
        resp_window = resp_signal[-window_samples:]
        
        # Adaptive Min-Max normalization (continuous!)
        adaptation_rate = 0.05
        for sample in resp_window:
            if self.resp_min is None:
                self.resp_min = sample
                self.resp_max = sample
            else:
                # Update min/max
                if sample < self.resp_min:
                    self.resp_min = sample
                if sample > self.resp_max:
                    self.resp_max = sample
                
                # Adapt towards current value
                adapt_factor = adaptation_rate ** 2
                self.resp_min += (sample - self.resp_min) * adapt_factor
                self.resp_max += (sample - self.resp_max) * adapt_factor
        
        # Normalize
        if self.resp_max > self.resp_min:
            resp_normalized = (resp_window - self.resp_min) / (self.resp_max - self.resp_min)
        else:
            resp_normalized = np.ones_like(resp_window) * 0.5
        
        # Detect breaths (peaks in normalized signal)
        peaks, _ = find_peaks(resp_normalized, distance=int(2 * self.sampling_rate), height=0.3)
        
        if len(peaks) >= 2:
            # Breathing rate
            intervals_s = np.diff(peaks) / self.sampling_rate
            rpm = 60 / np.mean(intervals_s)
            features['respiratory.resp_rate_mean'] = rpm
            
            # Update breath history (continuous!)
            for interval in intervals_s:
                if 2 < interval < 10:  # Valid breath (6-30 breaths/min)
                    self.resp_interval_history.append(interval)
            
            # Amplitude (peak-to-trough)
            amplitudes = []
            for i in range(len(peaks) - 1):
                peak_val = resp_normalized[peaks[i]]
                # Find trough between peaks
                trough_region = resp_normalized[peaks[i]:peaks[i+1]]
                if len(trough_region) > 0:
                    trough_val = np.min(trough_region)
                    amplitudes.append(peak_val - trough_val)
            
            if amplitudes:
                features['respiratory.resp_amplitude_mean'] = np.mean(amplitudes)
                
                # Update amplitude history (continuous!)
                for amp in amplitudes:
                    self.resp_amplitude_history.append(amp)
                
                # Coefficient of variation (from history)
                if len(self.resp_amplitude_history) >= 3:
                    amp_array = np.array(self.resp_amplitude_history)
                    features['respiratory.resp_amplitude_coefficient_of_variation'] = \
                        (np.std(amp_array) / np.mean(amp_array)) * 100 if np.mean(amp_array) > 0 else 0.0
                else:
                    features['respiratory.resp_amplitude_coefficient_of_variation'] = 0.0
            else:
                features['respiratory.resp_amplitude_mean'] = 0.0
                features['respiratory.resp_amplitude_coefficient_of_variation'] = 0.0
            
            # RPM coefficient of variation
            if len(self.resp_interval_history) >= 3:
                interval_array = np.array(self.resp_interval_history)
                rpm_array = 60 / interval_array
                features['respiratory.resp_rpm_coefficient_of_variation'] = \
                    (np.std(rpm_array) / np.mean(rpm_array)) * 100 if np.mean(rpm_array) > 0 else 0.0
            else:
                features['respiratory.resp_rpm_coefficient_of_variation'] = 0.0
        else:
            features['respiratory.resp_rate_mean'] = 0.0
            features['respiratory.resp_amplitude_mean'] = 0.0
            features['respiratory.resp_amplitude_coefficient_of_variation'] = 0.0
            features['respiratory.resp_rpm_coefficient_of_variation'] = 0.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of all feature names."""
        
        # Generate a dummy feature dict to get all keys
        dummy_features = {
            'eda.scl_mean': 0.0,
            'eda.scl_std': 0.0,
            'eda.scl_range': 0.0,
            'eda.scr_mean': 0.0,
            'eda.scr_std': 0.0,
            'eda.scr_max': 0.0,
            'eda.scr_frequency': 0.0,
            'cardiac.hr_mean': 0.0,
            'cardiac.hrv_rmssd': 0.0,
            'cardiac.hrv_sdnn': 0.0,
            'cardiac.hrv_pnn50': 0.0,
            'respiratory.resp_rate_mean': 0.0,
            'respiratory.resp_amplitude_mean': 0.0,
            'respiratory.resp_amplitude_coefficient_of_variation': 0.0,
            'respiratory.resp_rpm_coefficient_of_variation': 0.0,
        }
        
        return list(dummy_features.keys())



if __name__ == "__main__":
    print("Continuous Feature Extractor Module")
    print("="*80)
    print("\nKey advantages:")
    print("  ✓ Maintains filter states across emotion boundaries")
    print("  ✓ Captures transition dynamics")
    print("  ✓ Matches real-time deployment architecture")
    print("  ✓ Extract once, slice many ways")
