"""
Crocodile Biodata Pipeline - Data Slicer Module (Enhanced with Flexible Windowing)

Handles slicing and filtering of emotion-labeled physiological data with flexible
windowing strategies for different use cases.

Key Features:
    - Segment-based windowing (one sample per emotion segment)
    - Sliding window windowing (configurable overlap, may cross boundaries)
    - Hybrid windowing (multiple windows per segment, no boundary crossing)
    - Emotion filtering and quality control
    - Feeling_it pedal handling

Windowing Modes:
    - 'segment': One window per emotion segment (current approach, 76 samples)
    - 'sliding': Fixed windows with stride (crosses boundaries, ~1,000-5,000 samples)
    - 'hybrid': Multiple windows per segment (no crossing, ~500-1,500 samples)

Usage:
    >>> slicer = DataSlicer(sampling_rate=100)
    >>> 
    >>> # Segment mode (original)
    >>> windows = slicer.create_windows(data, 'session1', window_mode='segment')
    >>> 
    >>> # Training mode (hybrid, recommended)
    >>> windows = slicer.create_windows(
    ...     data, 'session1',
    ...     window_mode='hybrid',
    ...     window_size_s=30,
    ...     stride_s=5
    ... )
    >>> 
    >>> # Real-time mode (sliding)
    >>> windows = slicer.create_windows(
    ...     data, 'session1',
    ...     window_mode='sliding',
    ...     window_size_s=30,
    ...     stride_s=1
    ... )

See docs/guides/DataSlicer_Usage_Guide.md for detailed examples.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import warnings


@dataclass
class EmotionWindow:
    """
    Represents a single time window of physiological data with emotion label.
    
    This replaces/extends EmotionSegment to support flexible windowing.
    Can represent either a full segment or a sliding window.
    
    Attributes
    ----------
    window_id : str
        Unique identifier for this window
    session_id : str
        Session this window belongs to
    emotion : str
        Emotion label for this window
    start_idx : int
        Starting sample index in original data
    end_idx : int
        Ending sample index in original data
    start_time : float
        Starting time in seconds
    end_time : float
        Ending time in seconds
    duration : float
        Duration in seconds
    signals : dict
        Dictionary of signal arrays (e.g., {'heart': array, 'gsr': array})
    emotion_purity : float, optional
        Proportion of window that is labeled with this emotion (1.0 = pure)
    feeling_it : bool, optional
        Whether feeling_it pedal was pressed during this window
    feeling_it_ratio : float, optional
        Proportion of window with feeling_it = True
    feeling_it_indices : np.ndarray, optional
        Indices where feeling_it was True
    parent_segment_id : str, optional
        If from hybrid mode, ID of parent segment
    metadata : dict, optional
        Additional metadata
    """
    window_id: str
    session_id: str
    emotion: str
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    duration: float
    signals: Dict[str, np.ndarray]
    emotion_purity: float = 1.0
    feeling_it: bool = False
    feeling_it_ratio: float = 0.0
    feeling_it_indices: Optional[np.ndarray] = None
    parent_segment_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return (f"EmotionWindow(id={self.window_id}, emotion={self.emotion}, "
                f"duration={self.duration:.1f}s, purity={self.emotion_purity:.2f})")


# Backwards compatibility: EmotionSegment is just an EmotionWindow with purity=1.0
EmotionSegment = EmotionWindow


class DataSlicer:
    """
    Slice and filter emotion-labeled physiological data with flexible windowing.
    
    Supports three windowing modes:
    1. Segment mode: One window per emotion segment (original behavior)
    2. Sliding mode: Fixed-size windows with configurable stride
    3. Hybrid mode: Multiple windows per segment, no boundary crossing
    
    Parameters
    ----------
    sampling_rate : int, default=100
        Sampling rate of physiological signals in Hz
    
    Examples
    --------
    >>> # Create slicer
    >>> slicer = DataSlicer(sampling_rate=100)
    >>> 
    >>> # Segment mode (76 samples)
    >>> windows = slicer.create_windows(data, 'session1', window_mode='segment')
    >>> 
    >>> # Training mode (1,000 samples)
    >>> windows = slicer.create_windows(
    ...     data, 'session1',
    ...     window_mode='hybrid',
    ...     window_size_s=30,
    ...     stride_s=5
    ... )
    """
    
    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
    
    # ========================================================================
    # UNIFIED WINDOWING INTERFACE
    # ========================================================================
    
    def create_windows(self,
                      data: pd.DataFrame,
                      session_id: str,
                      window_mode: str = 'segment',
                      window_size_s: float = 30.0,
                      stride_s: float = 5.0,
                      emotion_col: str = 'emotion',
                      feeling_col: Optional[str] = 'feeling_it',
                      signal_cols: List[str] = None,
                      min_purity: float = 0.5,
                      **kwargs) -> List[EmotionWindow]:
        """
        Create windows using specified windowing strategy.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw physiological data with emotion labels
        session_id : str
            Identifier for this session
        window_mode : str, default='segment'
            Windowing strategy:
            - 'segment': One window per emotion segment
            - 'sliding': Fixed windows with stride (may cross boundaries)
            - 'hybrid': Multiple windows per segment (no crossing)
        window_size_s : float, default=30.0
            Window duration in seconds (for sliding/hybrid modes)
        stride_s : float, default=5.0
            Step between consecutive windows in seconds (for sliding/hybrid)
        emotion_col : str, default='emotion'
            Column name containing emotion labels
        feeling_col : str or None, default='feeling_it'
            Column name for feeling_it pedal (None to ignore)
        signal_cols : list of str, optional
            Column names for physiological signals
            Default: ['heart', 'gsr', 'respiration']
        min_purity : float, default=0.5
            Minimum emotion purity (for sliding mode)
            Windows with purity < min_purity are excluded
        **kwargs
            Additional arguments passed to windowing functions
            
        Returns
        -------
        windows : list of EmotionWindow
            List of windows created according to specified mode
        
        Examples
        --------
        >>> # Segment mode (original behavior)
        >>> windows = slicer.create_windows(data, 'session1', window_mode='segment')
        >>> print(f"Created {len(windows)} segments")
        >>> 
        >>> # Sliding mode (maximum data)
        >>> windows = slicer.create_windows(
        ...     data, 'session1',
        ...     window_mode='sliding',
        ...     window_size_s=30,
        ...     stride_s=5,
        ...     min_purity=0.8  # Only use windows that are 80%+ one emotion
        ... )
        >>> 
        >>> # Hybrid mode (recommended for training)
        >>> windows = slicer.create_windows(
        ...     data, 'session1',
        ...     window_mode='hybrid',
        ...     window_size_s=30,
        ...     stride_s=5
        ... )
        """
        
        if signal_cols is None:
            signal_cols = ['heart', 'gsr', 'respiration']
        
        print(f"Creating windows in '{window_mode}' mode...")
        
        if window_mode == 'segment':
            windows = self._create_segment_windows(
                data, session_id, emotion_col, feeling_col, signal_cols, **kwargs
            )
            
        elif window_mode == 'sliding':
            windows = self._create_sliding_windows(
                data, session_id, window_size_s, stride_s,
                emotion_col, feeling_col, signal_cols, min_purity
            )
            
        elif window_mode == 'hybrid':
            windows = self._create_hybrid_windows(
                data, session_id, window_size_s, stride_s,
                emotion_col, feeling_col, signal_cols
            )
            
        else:
            raise ValueError(
                f"Unknown window_mode: {window_mode}. "
                f"Choose from: 'segment', 'sliding', 'hybrid'"
            )
        
        print(f"  → Created {len(windows)} windows")
        
        return windows
    
    # ========================================================================
    # SEGMENT MODE (Original Behavior)
    # ========================================================================
    
    def _create_segment_windows(self,
                                data: pd.DataFrame,
                                session_id: str,
                                emotion_col: str,
                                feeling_col: Optional[str],
                                signal_cols: List[str],
                                **kwargs) -> List[EmotionWindow]:
        """
        Create one window per emotion segment (original behavior).
        
        This is the current approach: detects emotion boundaries and creates
        one window for each contiguous block of the same emotion.
        
        Returns
        -------
        windows : list of EmotionWindow
            One window per segment, typically 50-100 windows per dataset
        """
        return self.session_to_segments(
            data, session_id, emotion_col, feeling_col, signal_cols
        )
    
    # ========================================================================
    # SLIDING MODE (Maximum Data, May Cross Boundaries)
    # ========================================================================
    
    def _create_sliding_windows(self,
                               data: pd.DataFrame,
                               session_id: str,
                               window_size_s: float,
                               stride_s: float,
                               emotion_col: str,
                               feeling_col: Optional[str],
                               signal_cols: List[str],
                               min_purity: float = 0.5) -> List[EmotionWindow]:
        """
        Create sliding windows over entire session.
        
        Windows slide across the entire session with fixed stride, potentially
        crossing emotion boundaries. Emotion label is determined by majority vote.
        
        Good for:
        - Maximum training data
        - Real-time simulation (stride=1s)
        - When you want dense sampling
        
        Parameters
        ----------
        min_purity : float, default=0.5
            Minimum proportion of window that must be single emotion
            Windows with lower purity are excluded
            
        Returns
        -------
        windows : list of EmotionWindow
            Sliding windows, typically 1,000-5,000 per dataset
        """
        window_samples = int(window_size_s * self.sampling_rate)
        stride_samples = int(stride_s * self.sampling_rate)
        
        windows = []
        excluded_count = 0
        
        for start_idx in range(0, len(data) - window_samples + 1, stride_samples):
            end_idx = start_idx + window_samples
            window_data = data.iloc[start_idx:end_idx]
            
            # Determine emotion label (majority vote)
            emotion_counts = window_data[emotion_col].value_counts()
            emotion = emotion_counts.index[0]
            
            # Calculate emotion purity
            purity = emotion_counts.iloc[0] / len(window_data)
            
            # Filter by purity
            if purity < min_purity:
                excluded_count += 1
                continue
            
            # Extract signals
            signals = {col: window_data[col].values for col in signal_cols}
            
            # Feeling_it statistics
            feeling_it_data = None
            feeling_it_ratio = 0.0
            if feeling_col and feeling_col in window_data.columns:
                feeling_it_data = window_data[feeling_col].values
                feeling_it_ratio = feeling_it_data.sum() / len(feeling_it_data)
            
            window = EmotionWindow(
                window_id=f"{session_id}_sliding_{start_idx:06d}",
                session_id=session_id,
                emotion=emotion,
                emotion_purity=purity,
                start_idx=start_idx,
                end_idx=end_idx,
                start_time=start_idx / self.sampling_rate,
                end_time=end_idx / self.sampling_rate,
                duration=window_size_s,
                signals=signals,
                feeling_it=feeling_it_ratio > 0,
                feeling_it_ratio=feeling_it_ratio,
                metadata={'window_mode': 'sliding', 'stride_s': stride_s}
            )
            
            windows.append(window)
        
        if excluded_count > 0:
            print(f"  → Excluded {excluded_count} windows (purity < {min_purity})")
        
        return windows
    
    # ========================================================================
    # HYBRID MODE (Multiple Windows per Segment, No Crossing)
    # ========================================================================
    
    def _create_hybrid_windows(self,
                              data: pd.DataFrame,
                              session_id: str,
                              window_size_s: float,
                              stride_s: float,
                              emotion_col: str,
                              feeling_col: Optional[str],
                              signal_cols: List[str]) -> List[EmotionWindow]:
        """
        Create multiple windows per segment without crossing boundaries.
        
        First detects emotion segments, then creates sliding windows within
        each segment. Windows never span multiple emotions (purity = 1.0).
        
        Good for:
        - Training (more data than segment mode)
        - Maintaining emotion purity
        - Avoiding label confusion at boundaries
        
        Returns
        -------
        windows : list of EmotionWindow
            Multiple windows per segment, typically 500-1,500 per dataset
        """
        # First, get emotion segments
        segments = self.session_to_segments(
            data, session_id, emotion_col, feeling_col, signal_cols
        )
        
        window_samples = int(window_size_s * self.sampling_rate)
        stride_samples = int(stride_s * self.sampling_rate)
        
        all_windows = []
        
        for segment in segments:
            segment_samples = len(segment.signals[signal_cols[0]])
            
            # Skip segments shorter than window size
            if segment_samples < window_samples:
                continue
            
            # Create windows within this segment
            for offset in range(0, segment_samples - window_samples + 1, stride_samples):
                start_idx_global = segment.start_idx + offset
                end_idx_global = start_idx_global + window_samples
                
                # Extract signals for this window
                window_signals = {
                    col: segment.signals[col][offset:offset + window_samples]
                    for col in signal_cols
                }
                
                # Feeling_it statistics for this window
                feeling_it_ratio = 0.0
                if segment.feeling_it_indices is not None:
                    # Which feeling_it indices fall in this window?
                    window_feeling_indices = segment.feeling_it_indices[
                        (segment.feeling_it_indices >= offset) &
                        (segment.feeling_it_indices < offset + window_samples)
                    ]
                    feeling_it_ratio = len(window_feeling_indices) / window_samples
                
                window = EmotionWindow(
                    window_id=f"{segment.window_id}_win{len(all_windows):03d}",
                    session_id=session_id,
                    emotion=segment.emotion,
                    emotion_purity=1.0,  # Always pure (within one segment)
                    start_idx=start_idx_global,
                    end_idx=end_idx_global,
                    start_time=start_idx_global / self.sampling_rate,
                    end_time=end_idx_global / self.sampling_rate,
                    duration=window_size_s,
                    signals=window_signals,
                    feeling_it=feeling_it_ratio > 0,
                    feeling_it_ratio=feeling_it_ratio,
                    parent_segment_id=segment.window_id,
                    metadata={
                        'window_mode': 'hybrid',
                        'stride_s': stride_s,
                        'parent_duration': segment.duration
                    }
                )
                
                all_windows.append(window)
        
        return all_windows
    
    # ========================================================================
    # ORIGINAL METHODS (Kept for Backwards Compatibility)
    # ========================================================================
    
    def session_to_segments(self,
                           data: pd.DataFrame,
                           session_id: str,
                           emotion_col: str = 'emotion',
                           feeling_col: Optional[str] = 'feeling_it',
                           signal_cols: List[str] = None) -> List[EmotionWindow]:
        """
        Convert session data to emotion segments (original method).
        
        Detects emotion boundaries and creates one segment per contiguous
        emotion block.
        
        This method is preserved for backwards compatibility and is called
        by window_mode='segment'.
        """
        if signal_cols is None:
            signal_cols = ['heart', 'gsr', 'respiration']
        
        print(f"Processing session '{session_id}' with signals: {signal_cols}")
        
        # Detect emotion boundaries
        emotion_changes = data[emotion_col] != data[emotion_col].shift()
        segment_starts = data.index[emotion_changes].tolist()
        
        segments = []
        
        for i, start_idx in enumerate(segment_starts):
            # Determine end index
            if i < len(segment_starts) - 1:
                end_idx = segment_starts[i + 1]
            else:
                end_idx = len(data)
            
            # Extract segment data
            segment_data = data.iloc[start_idx:end_idx]
            emotion = segment_data[emotion_col].iloc[0]
            
            # Extract signals
            signals = {col: segment_data[col].values for col in signal_cols}
            
            # Feeling_it statistics
            feeling_it_array = None
            feeling_it_ratio = 0.0
            feeling_it_indices = None
            
            if feeling_col and feeling_col in segment_data.columns:
                feeling_it_array = segment_data[feeling_col].values
                feeling_it_indices = np.where(feeling_it_array)[0]
                feeling_it_ratio = len(feeling_it_indices) / len(feeling_it_array)
            
            # Calculate times
            start_time = start_idx / self.sampling_rate
            end_time = end_idx / self.sampling_rate
            duration = end_time - start_time
            
            segment = EmotionWindow(
                window_id=f"{session_id}_seg{i:03d}",
                session_id=session_id,
                emotion=emotion,
                emotion_purity=1.0,
                start_idx=start_idx,
                end_idx=end_idx,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                signals=signals,
                feeling_it=len(feeling_it_indices) > 0 if feeling_it_indices is not None else False,
                feeling_it_ratio=feeling_it_ratio,
                feeling_it_indices=feeling_it_indices,
                metadata={'window_mode': 'segment'}
            )
            
            segments.append(segment)
        
        print(f"  → Created {len(segments)} segments from emotion boundaries")
        
        return segments
    
    # ========================================================================
    # FILTERING METHODS (Work on any list of windows)
    # ========================================================================
    
    def filter_by_emotions(self,
                          windows: List[EmotionWindow],
                          include_emotions: Optional[List[str]] = None,
                          exclude_emotions: Optional[List[str]] = None) -> List[EmotionWindow]:
        """Filter windows by emotion labels."""
        filtered = windows.copy()
        
        if include_emotions:
            filtered = [w for w in filtered if w.emotion in include_emotions]
            print(f"Filter: Including emotions {include_emotions}")
            print(f"  → Kept {len(filtered)}/{len(windows)} windows")
        
        if exclude_emotions:
            filtered = [w for w in filtered if w.emotion not in exclude_emotions]
            print(f"Filter: Excluding emotions {exclude_emotions}")
            print(f"  → Kept {len(filtered)}/{len(windows)} windows")
        
        return filtered
    
    def filter_by_feeling_it(self,
                            windows: List[EmotionWindow],
                            require_feeling_it: bool = True,
                            min_feeling_ratio: float = 0.0,
                            time_tolerance_s: float = 0.0) -> List[EmotionWindow]:
        """Filter windows by feeling_it pedal status."""
        if not require_feeling_it and min_feeling_ratio == 0:
            return windows
        
        filtered = []
        
        for window in windows:
            # Check feeling_it ratio
            if window.feeling_it_ratio < min_feeling_ratio:
                continue
            
            # Check if any feeling_it present (considering tolerance)
            if require_feeling_it and not window.feeling_it:
                continue
            
            filtered.append(window)
        
        print(f"Filter: feeling_it (require={require_feeling_it}, min_ratio={min_feeling_ratio})")
        print(f"  → Kept {len(filtered)}/{len(windows)} windows")
        
        return filtered
    
    def filter_by_quality(self,
                         windows: List[EmotionWindow],
                         min_duration_s: float = 5.0,
                         max_duration_s: Optional[float] = None,
                         check_signal_validity: bool = False,
                         max_flat_ratio: float = 0.9) -> List[EmotionWindow]:
        """Filter windows by quality criteria."""
        filtered = []
        
        for window in windows:
            # Duration filter
            if window.duration < min_duration_s:
                continue
            if max_duration_s and window.duration > max_duration_s:
                continue
            
            # Signal validity check
            if check_signal_validity:
                valid = True
                for signal_name, signal_data in window.signals.items():
                    # Check for flatness
                    if len(np.unique(signal_data)) / len(signal_data) < (1 - max_flat_ratio):
                        valid = False
                        break
                    
                    # Check for zeros
                    if np.all(signal_data == 0):
                        valid = False
                        break
                
                if not valid:
                    continue
            
            filtered.append(window)
        
        print(f"Quality filter: Kept {len(filtered)}/{len(windows)} windows "
              f"(min_duration={min_duration_s}s, check_validity={check_signal_validity})")
        
        return filtered
    
    def filter_by_purity(self,
                        windows: List[EmotionWindow],
                        min_purity: float = 0.8) -> List[EmotionWindow]:
        """
        Filter windows by emotion purity.
        
        Useful for sliding mode where windows may span multiple emotions.
        
        Parameters
        ----------
        min_purity : float, default=0.8
            Minimum proportion of window that must be single emotion
            
        Returns
        -------
        filtered : list of EmotionWindow
            Windows with purity >= min_purity
        """
        filtered = [w for w in windows if w.emotion_purity >= min_purity]
        
        print(f"Purity filter: Kept {len(filtered)}/{len(windows)} windows "
              f"(min_purity={min_purity})")
        
        return filtered
    
    # ========================================================================
    # SUMMARY AND UTILITIES
    # ========================================================================
    
    def print_summary(self, windows: List[EmotionWindow]):
        """Print summary statistics of windows."""
        if len(windows) == 0:
            print("\n" + "="*80)
            print("Empty DataFrame")
            print("Columns: []")
            print("Index: []")
            print("="*80)
            return
        
        # Gather statistics
        stats = {}
        for window in windows:
            emotion = window.emotion
            if emotion not in stats:
                stats[emotion] = {
                    'count': 0,
                    'total_duration': 0.0,
                    'feeling_count': 0,
                    'feeling_duration': 0.0
                }
            
            stats[emotion]['count'] += 1
            stats[emotion]['total_duration'] += window.duration
            if window.feeling_it:
                stats[emotion]['feeling_count'] += 1
                stats[emotion]['feeling_duration'] += window.duration * window.feeling_it_ratio
        
        # Create summary DataFrame
        rows = []
        for emotion, data in stats.items():
            rows.append({
                'emotion': emotion,
                'n_windows': data['count'],
                'total_duration_s': data['total_duration'],
                'avg_duration_s': data['total_duration'] / data['count'],
                'n_with_feeling': data['feeling_count'],
                'feeling_duration_s': data['feeling_duration'],
                'avg_feeling_ratio': data['feeling_duration'] / data['total_duration'] if data['total_duration'] > 0 else 0
            })
        
        df = pd.DataFrame(rows).sort_values('n_windows', ascending=False)
        
        # Add total row
        total_row = {
            'emotion': 'TOTAL',
            'n_windows': df['n_windows'].sum(),
            'total_duration_s': df['total_duration_s'].sum(),
            'avg_duration_s': df['total_duration_s'].sum() / df['n_windows'].sum(),
            'n_with_feeling': df['n_with_feeling'].sum(),
            'feeling_duration_s': df['feeling_duration_s'].sum(),
            'avg_feeling_ratio': df['feeling_duration_s'].sum() / df['total_duration_s'].sum()
        }
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        
        # Print
        print("\n" + "="*80)
        print("WINDOW SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
    
    def get_windowing_info(self, windows: List[EmotionWindow]) -> Dict[str, Any]:
        """
        Get information about windowing configuration used.
        
        Returns
        -------
        info : dict
            Dictionary with windowing statistics
        """
        if len(windows) == 0:
            return {}
        
        # Extract metadata from first window
        first = windows[0]
        mode = first.metadata.get('window_mode', 'unknown')
        
        info = {
            'window_mode': mode,
            'n_windows': len(windows),
            'window_size_s': first.duration,
            'total_duration_s': sum(w.duration for w in windows),
        }
        
        if 'stride_s' in first.metadata:
            info['stride_s'] = first.metadata['stride_s']
            info['overlap_ratio'] = 1 - (info['stride_s'] / info['window_size_s'])
        
        # Purity statistics
        purities = [w.emotion_purity for w in windows]
        info['mean_purity'] = np.mean(purities)
        info['min_purity'] = np.min(purities)
        
        return info


def slice_features_into_windows(features_df: pd.DataFrame,
                                window_size_s: float = 30,
                                stride_s: float = 5,
                                min_purity: float = 0.7) -> List[Dict]:
    """
    Slice continuous features into windows for training.

    Parameters
    ----------
    features_df : pd.DataFrame
        Continuous features with 'timestamp' and 'emotion' columns
    window_size_s : float
        Window size in seconds
    stride_s : float
        Stride between windows in seconds
    min_purity : float
        Minimum emotion purity (0.0-1.0)

    Returns
    -------
    list of dict
        Windows with aggregated features and metadata
    """

    windows = []

    max_time = features_df['timestamp'].max()
    window_id = 0

    for start_time in np.arange(0, max_time - window_size_s, stride_s):
        end_time = start_time + window_size_s

        # Get features in this window
        mask = (features_df['timestamp'] >= start_time) & (features_df['timestamp'] < end_time)
        window_features = features_df[mask]

        if len(window_features) < 2:
            continue

        # Determine emotion label (majority vote)
        emotion_counts = window_features['emotion'].value_counts()
        dominant_emotion = emotion_counts.index[0]
        emotion_purity = emotion_counts.iloc[0] / len(window_features)

        # Skip if purity too low
        if emotion_purity < min_purity:
            continue

        # Aggregate features (mean over window)
        feature_cols = [c for c in window_features.columns
                       if c not in ['timestamp', 'sample_idx', 'emotion', 'feeling_it']]

        window_dict = {
            'window_id': f'win{window_id:04d}',
            'start_time': start_time,
            'end_time': end_time,
            'duration': window_size_s,
            'emotion': dominant_emotion,
            'emotion_purity': emotion_purity,
        }

        # Mean of features over window
        for col in feature_cols:
            window_dict[col] = window_features[col].mean()

        # Add feeling_it if available
        if 'feeling_it' in window_features.columns:
            window_dict['feeling_it_ratio'] = window_features['feeling_it'].mean()

        windows.append(window_dict)
        window_id += 1

    return windows


if __name__ == "__main__":
    print("Data Slicer Module with Flexible Windowing")
    print("=" * 80)
    print("\nUsage:")
    print("  from modules.data_slicer import DataSlicer")
    print("  slicer = DataSlicer(sampling_rate=100)")
    print("  windows = slicer.create_windows(data, 'session1', window_mode='hybrid')")
