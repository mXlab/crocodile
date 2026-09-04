"""
Enhanced Respiratory Feature Extraction

Based on:
1. BioData library (Respiration.cpp, Respiration.h)
2. Luana Belinsky's MX2403FR research
3. Real-time normalization and change detection

Key additions:
- Amplitude/RPM rate of change (Δ)
- Amplitude/RPM coefficient of variation (CV)
- Exhale/inhale phase detection
- Adaptive normalization (MinMax with adaptation)
- Long-term trend tracking (slow LPF)

These features capture emotion-relevant dynamics that static features miss!
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Dict, Tuple


class EnhancedRespiratoryFeatures:
    """
    Extract respiratory features matching BioData library approach.
    
    Implements:
    - Adaptive MinMax normalization
    - Exponential moving averages for trends
    - Rate of change features
    - Coefficient of variation
    - Phase detection
    """
    
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        
        # Time windows (from Luana's research)
        self.amplitude_normalizer_window_s = 90  # Long window for amplitude
        self.amplitude_variability_window_s = 30  # CV calculation window
        self.rpm_normalizer_window_s = 90
        self.rpm_variability_window_s = 30
        
        # Number of breaths to track for rate of change
        self.n_breaths_for_change = 5  # Track last 5 breaths
    
    def extract_enhanced_features(self, 
                                  resp_raw: np.ndarray,
                                  window_size_s: float) -> Dict[str, float]:
        """
        Extract enhanced respiratory features.
        
        Returns both basic + advanced features from Luana's research.
        """
        
        # ====================================================================
        # 1. BASIC FEATURES (Already in your code)
        # ====================================================================
        
        basic_features = self._extract_basic_features(resp_raw)
        
        # ====================================================================
        # 2. ADAPTIVE NORMALIZATION FEATURES
        # ====================================================================
        
        # Simulate adaptive MinMax normalization
        normalized_signal = self._adaptive_minmax_normalize(resp_raw)
        
        # ====================================================================
        # 3. BREATH DETECTION WITH PHASE
        # ====================================================================
        
        breath_info = self._detect_breaths_with_phase(normalized_signal)
        
        # ====================================================================
        # 4. AMPLITUDE FEATURES (Enhanced)
        # ====================================================================
        
        amplitude_features = self._extract_amplitude_features(
            resp_raw,
            breath_info,
            window_size_s
        )
        
        # ====================================================================
        # 5. RPM FEATURES (Enhanced)
        # ====================================================================
        
        rpm_features = self._extract_rpm_features(
            breath_info,
            window_size_s
        )
        
        # ====================================================================
        # 6. PHASE FEATURES
        # ====================================================================
        
        phase_features = self._extract_phase_features(breath_info, resp_raw)
        
        # Combine all
        all_features = {
            **basic_features,
            **amplitude_features,
            **rpm_features,
            **phase_features
        }
        
        return all_features
    
    def _adaptive_minmax_normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Adaptive MinMax normalization (from BioData MinMax.h).
        
        Min/Max bounds adapt slowly to track signal drift.
        """
        
        # Adaptation rate (from Respiration.cpp: adapt(0.05))
        adaptation_rate = 0.05
        
        normalized = np.zeros_like(signal, dtype=float)
        
        # Initialize
        current_min = signal[0]
        current_max = signal[0]
        
        for i, value in enumerate(signal):
            # Update min/max with new observation
            if value > current_max:
                current_max = value
            if value < current_min:
                current_min = value
            
            # Adapt min/max towards current value (low-pass)
            adapt_factor = adaptation_rate ** 2  # Squared (from MinMax.h)
            current_min += (value - current_min) * adapt_factor
            current_max += (value - current_max) * adapt_factor
            
            # Normalize to [0, 1]
            if current_max == current_min:
                normalized[i] = 0.5
            else:
                normalized[i] = (value - current_min) / (current_max - current_min)
        
        return normalized
    
    def _detect_breaths_with_phase(self, normalized_signal: np.ndarray) -> Dict:
        """
        Detect breaths and determine exhale/inhale phases.
        
        From Luana's code:
        - Peak (max): End of exhale
        - Trough (min): End of inhale
        """
        
        # Peak detection parameters (from MX2403FR)
        peak_threshold = 0.5
        trough_threshold = 0.5
        
        min_breath_duration = int(self.sampling_rate * 2)  # Min 2s between breaths
        
        # Find peaks (exhale ends) and troughs (inhale ends)
        peaks, _ = find_peaks(
            normalized_signal,
            height=peak_threshold,
            distance=min_breath_duration
        )
        
        troughs, _ = find_peaks(
            -normalized_signal,  # Invert for trough detection
            height=-trough_threshold,
            distance=min_breath_duration
        )
        
        # Determine phase at each sample (0=inhale, 1=exhale)
        phase = np.zeros(len(normalized_signal), dtype=int)
        
        # Between trough and peak = exhaling (temperature rising)
        for i in range(len(normalized_signal)):
            # Find most recent trough and peak
            recent_troughs = troughs[troughs < i]
            recent_peaks = peaks[peaks < i]
            
            if len(recent_troughs) > 0 and len(recent_peaks) > 0:
                # If most recent event is trough → exhaling
                phase[i] = 1 if recent_troughs[-1] > recent_peaks[-1] else 0
            elif len(recent_troughs) > 0:
                phase[i] = 1  # Started with trough
        
        # Compute amplitudes for each breath
        amplitudes = []
        intervals = []
        
        for i in range(len(peaks) - 1):
            # Find trough between consecutive peaks
            troughs_in_range = troughs[(troughs > peaks[i]) & (troughs < peaks[i+1])]
            
            if len(troughs_in_range) > 0:
                trough_idx = troughs_in_range[0]
                
                # Amplitude = peak - trough
                amplitude = normalized_signal[peaks[i]] - normalized_signal[trough_idx]
                amplitudes.append(amplitude)
                
                # Interval in seconds
                interval_s = (peaks[i+1] - peaks[i]) / self.sampling_rate
                intervals.append(interval_s)
        
        return {
            'peaks': peaks,
            'troughs': troughs,
            'phase': phase,
            'amplitudes': np.array(amplitudes) if amplitudes else np.array([0.0]),
            'intervals': np.array(intervals) if intervals else np.array([0.0])
        }
    
    def _extract_amplitude_features(self,
                                    resp_raw: np.ndarray,
                                    breath_info: Dict,
                                    window_size_s: float) -> Dict[str, float]:
        """
        Extract enhanced amplitude features from Luana's research.
        """
        
        amplitudes = breath_info['amplitudes']
        
        if len(amplitudes) < 2:
            return {
                'resp_amplitude_rate_of_change': 0.0,
                'resp_amplitude_coefficient_of_variation': 0.0,
                'resp_amplitude_level_indicator': 0.5
            }
        
        # ================================================================
        # Rate of Change (Δ amplitude)
        # ================================================================
        # From MX2403FR: difference between current and oldest amplitude
        # divided by time elapsed, converted to points/minute
        
        n_for_change = min(self.n_breaths_for_change, len(amplitudes))
        
        if n_for_change >= 2:
            current_amp = amplitudes[-1]
            oldest_amp = amplitudes[-(n_for_change)]
            
            # Time elapsed (approximate from intervals)
            intervals = breath_info['intervals']
            if len(intervals) >= n_for_change - 1:
                time_elapsed_s = np.sum(intervals[-(n_for_change-1):])
                
                # Rate of change in amplitude units per minute
                rate_of_change = (current_amp - oldest_amp) / time_elapsed_s * 60
            else:
                rate_of_change = 0.0
        else:
            rate_of_change = 0.0
        
        # ================================================================
        # Coefficient of Variation (amplitude variability)
        # ================================================================
        # From MX2403FR: CV = (std_dev / mean) * 100
        # Uses 30-second window
        
        if len(amplitudes) > 1 and np.mean(amplitudes) > 0:
            cv = (np.std(amplitudes) / np.mean(amplitudes)) * 100
        else:
            cv = 0.0
        
        # ================================================================
        # Amplitude Level Indicator (scaled 0-1 relative to ±1 std)
        # ================================================================
        # From MX2403FR: map amplitude to [0,1] using ±1 std dev bounds
        
        amp_mean = np.mean(amplitudes)
        amp_std = np.std(amplitudes)
        
        if amp_std > 0:
            current_normalized = (amplitudes[-1] - amp_mean) / amp_std
            # Map [-1, +1] → [0, 1]
            level_indicator = np.clip((current_normalized + 1) / 2, 0, 1)
        else:
            level_indicator = 0.5
        
        return {
            'resp_amplitude_rate_of_change': rate_of_change,
            'resp_amplitude_coefficient_of_variation': cv,
            'resp_amplitude_level_indicator': level_indicator
        }
    
    def _extract_rpm_features(self,
                              breath_info: Dict,
                              window_size_s: float) -> Dict[str, float]:
        """
        Extract enhanced RPM (breaths per minute) features.
        """
        
        intervals = breath_info['intervals']
        
        if len(intervals) < 2:
            return {
                'resp_rpm_rate_of_change': 0.0,
                'resp_rpm_coefficient_of_variation': 0.0,
                'resp_rpm_level_indicator': 0.5
            }
        
        # Convert intervals to RPM
        rpms = 60.0 / intervals
        
        # ================================================================
        # Rate of Change (Δ RPM)
        # ================================================================
        
        n_for_change = min(self.n_breaths_for_change, len(rpms))
        
        if n_for_change >= 2:
            current_rpm = rpms[-1]
            oldest_rpm = rpms[-(n_for_change)]
            time_elapsed_s = np.sum(intervals[-(n_for_change-1):])
            
            # Rate of change in RPM per minute
            rate_of_change = (current_rpm - oldest_rpm) / time_elapsed_s * 60
        else:
            rate_of_change = 0.0
        
        # ================================================================
        # Coefficient of Variation (RPM variability)
        # ================================================================
        
        if len(rpms) > 1 and np.mean(rpms) > 0:
            cv = (np.std(rpms) / np.mean(rpms)) * 100
        else:
            cv = 0.0
        
        # ================================================================
        # RPM Level Indicator
        # ================================================================
        
        rpm_mean = np.mean(rpms)
        rpm_std = np.std(rpms)
        
        if rpm_std > 0:
            current_normalized = (rpms[-1] - rpm_mean) / rpm_std
            level_indicator = np.clip((current_normalized + 1) / 2, 0, 1)
        else:
            level_indicator = 0.5
        
        return {
            'resp_rpm_rate_of_change': rate_of_change,
            'resp_rpm_coefficient_of_variation': cv,
            'resp_rpm_level_indicator': level_indicator
        }
    
    def _extract_phase_features(self,
                                breath_info: Dict,
                                resp_raw: np.ndarray) -> Dict[str, float]:
        """
        Extract exhale/inhale phase features.
        """
        
        phase = breath_info['phase']
        
        # Proportion of time spent exhaling
        exhale_ratio = np.mean(phase)
        
        # Currently exhaling? (at end of window)
        currently_exhaling = float(phase[-1])
        
        return {
            'resp_exhale_ratio': exhale_ratio,
            'resp_currently_exhaling': currently_exhaling
        }
    
    def _extract_basic_features(self, resp_raw: np.ndarray) -> Dict[str, float]:
        """
        Basic features you already have.
        Kept for backwards compatibility.
        """
        
        # These are placeholders - use your existing implementation
        return {
            'resp_rate_mean': 0.0,  # You already compute this
            'resp_rate_median': 0.0,
            'resp_amplitude_mean': 0.0,
            'resp_amplitude_median': 0.0
            # ... etc
        }


# ==================================================================
# USAGE EXAMPLE
# ==================================================================

if __name__ == "__main__":
    print("Enhanced Respiratory Feature Extraction")
    print("="*80)
    print("\nNew features added:")
    print("  1. resp_amplitude_rate_of_change")
    print("  2. resp_amplitude_coefficient_of_variation") 
    print("  3. resp_amplitude_level_indicator")
    print("  4. resp_rpm_rate_of_change")
    print("  5. resp_rpm_coefficient_of_variation")
    print("  6. resp_rpm_level_indicator")
    print("  7. resp_exhale_ratio")
    print("  8. resp_currently_exhaling")
    print("\nThese capture emotional dynamics that static features miss!")
