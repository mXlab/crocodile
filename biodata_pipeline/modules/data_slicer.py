"""
Crocodile Project - Module 2: Data Slicer

This module handles slicing and filtering of emotion-labeled physiological data.
Key features:
- Filter by emotion labels (include/exclude lists)
- Filter by feeling_it pedal (with time tolerance)
- Extract temporal windows (fixed size, overlapping)
- Quality control (minimum duration, signal validity)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings


@dataclass
class EmotionSegment:
    """
    Represents a segment of physiological data with emotion label.
    """
    segment_id: str
    session_id: str
    emotion: str
    start_idx: int              # Start index in original data
    end_idx: int                # End index in original data
    start_time: float           # Start time in seconds
    end_time: float             # End time in seconds
    duration: float             # Duration in seconds
    signals: Dict[str, np.ndarray]  # {'heart': array, 'gsr': array, 'respiration': array}
    feeling_it: bool            # True if any feeling_it==1 in segment
    feeling_it_ratio: float     # Proportion of segment with feeling_it==1
    feeling_it_indices: List[int]  # Indices where feeling_it==1
    metadata: Dict              # Additional info (participant, session_date, etc.)
    
    def __repr__(self):
        return (f"EmotionSegment(id={self.segment_id}, emotion={self.emotion}, "
                f"duration={self.duration:.1f}s, feeling_it={self.feeling_it_ratio:.2f})")


class DataSlicer:
    """
    Slice and filter emotion-labeled physiological data.
    """
    
    def __init__(self, sampling_rate: int = 100):
        """
        Parameters
        ----------
        sampling_rate : int
            Sampling frequency in Hz (default: 100)
        """
        self.sr = sampling_rate
        
    # ========================================================================
    # 1. SESSION TO SEGMENTS: Convert continuous data to emotion segments
    # ========================================================================
    
    def session_to_segments(self, 
                           data: pd.DataFrame,
                           session_id: str = 'session',
                           emotion_col: str = 'emotion',
                           feeling_col: str = 'feeling_it',
                           signal_cols: Optional[List[str]] = None,
                           metadata: Optional[Dict] = None) -> List[EmotionSegment]:
        """
        Convert a continuous session DataFrame into discrete emotion segments.
        Creates a segment whenever emotion label changes.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns: [signal_cols..., emotion_col, feeling_col]
        session_id : str
            Identifier for this session
        emotion_col : str
            Name of emotion label column
        feeling_col : str
            Name of feeling_it pedal column
        signal_cols : List[str], optional
            Names of signal columns. If None, auto-detect all numeric columns
            except emotion and feeling_it
        metadata : dict, optional
            Additional metadata to attach to segments
            
        Returns
        -------
        segments : List[EmotionSegment]
            List of emotion segments
        """
        
        if metadata is None:
            metadata = {}
        
        # Auto-detect signal columns if not provided
        if signal_cols is None:
            signal_cols = [col for col in data.columns 
                          if col not in [emotion_col, feeling_col] and 
                          pd.api.types.is_numeric_dtype(data[col])]
        
        print(f"Processing session '{session_id}' with signals: {signal_cols}")
        
        # Detect emotion boundaries (where emotion label changes)
        emotion_changes = data[emotion_col].ne(data[emotion_col].shift())
        change_indices = np.where(emotion_changes)[0].tolist()
        
        # Add start and end indices
        if 0 not in change_indices:
            change_indices.insert(0, 0)
        change_indices.append(len(data))
        
        segments = []
        
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]
            
            # Extract segment data
            segment_data = data.iloc[start_idx:end_idx]
            emotion = segment_data[emotion_col].iloc[0]
            
            # Extract signals
            signals = {}
            for col in signal_cols:
                signals[col] = segment_data[col].values
            
            # Analyze feeling_it
            feeling_it_values = segment_data[feeling_col].values
            feeling_it_indices = np.where(feeling_it_values == 1)[0].tolist()
            feeling_it_ratio = np.sum(feeling_it_values == 1) / len(feeling_it_values)
            
            # Create segment
            segment = EmotionSegment(
                segment_id=f"{session_id}_seg{i:03d}",
                session_id=session_id,
                emotion=emotion,
                start_idx=start_idx,
                end_idx=end_idx,
                start_time=start_idx / self.sr,
                end_time=end_idx / self.sr,
                duration=(end_idx - start_idx) / self.sr,
                signals=signals,
                feeling_it=len(feeling_it_indices) > 0,
                feeling_it_ratio=feeling_it_ratio,
                feeling_it_indices=feeling_it_indices,
                metadata=metadata.copy()
            )
            
            segments.append(segment)
        
        print(f"  → Created {len(segments)} segments from emotion boundaries")
        return segments
    
    # ========================================================================
    # 2. EMOTION FILTERING
    # ========================================================================
    
    def filter_by_emotions(self,
                          segments: List[EmotionSegment],
                          include_emotions: Optional[List[str]] = None,
                          exclude_emotions: Optional[List[str]] = None) -> List[EmotionSegment]:
        """
        Filter segments by emotion labels.
        
        Parameters
        ----------
        segments : List[EmotionSegment]
            Input segments
        include_emotions : List[str], optional
            Keep only these emotions (if provided)
        exclude_emotions : List[str], optional
            Remove these emotions (if provided)
            
        Returns
        -------
        filtered : List[EmotionSegment]
            Filtered segments
        """
        
        filtered = segments
        
        if include_emotions is not None:
            include_set = set(include_emotions)
            filtered = [seg for seg in filtered if seg.emotion in include_set]
            print(f"Filter: Including emotions {include_emotions}")
            print(f"  → Kept {len(filtered)}/{len(segments)} segments")
        
        if exclude_emotions is not None:
            exclude_set = set(exclude_emotions)
            filtered = [seg for seg in filtered if seg.emotion not in exclude_set]
            print(f"Filter: Excluding emotions {exclude_emotions}")
            print(f"  → Kept {len(filtered)}/{len(segments)} segments")
        
        return filtered
    
    # ========================================================================
    # 3. FEELING_IT FILTERING
    # ========================================================================
    
    def filter_by_feeling_it(self,
                            segments: List[EmotionSegment],
                            require_feeling_it: bool = True,
                            min_feeling_ratio: float = 0.0,
                            time_tolerance_s: float = 0.0) -> List[EmotionSegment]:
        """
        Filter segments based on feeling_it pedal.
        
        Parameters
        ----------
        segments : List[EmotionSegment]
            Input segments
        require_feeling_it : bool
            If True, keep only segments with feeling_it==1
        min_feeling_ratio : float
            Minimum proportion of segment that must have feeling_it==1 (0.0 to 1.0)
        time_tolerance_s : float
            Extend feeling_it zones by this many seconds before and after (default: 0.0)
            
        Returns
        -------
        filtered : List[EmotionSegment]
            Filtered segments
        """
        
        if not require_feeling_it and min_feeling_ratio == 0.0:
            print("Filter: No feeling_it filtering (all segments kept)")
            return segments
        
        filtered = []
        
        for seg in segments:
            # Check basic feeling_it requirement
            if require_feeling_it and not seg.feeling_it:
                continue
            
            # Check feeling_it ratio
            if seg.feeling_it_ratio < min_feeling_ratio:
                continue
            
            # Apply time tolerance if needed
            if time_tolerance_s > 0.0 and seg.feeling_it:
                seg_expanded = self._expand_segment_around_feeling_it(seg, time_tolerance_s)
                filtered.append(seg_expanded)
            else:
                filtered.append(seg)
        
        print(f"Filter: feeling_it (require={require_feeling_it}, "
              f"min_ratio={min_feeling_ratio:.2f}, tolerance={time_tolerance_s}s)")
        print(f"  → Kept {len(filtered)}/{len(segments)} segments")
        
        return filtered
    
    def _expand_segment_around_feeling_it(self,
                                         segment: EmotionSegment,
                                         time_tolerance_s: float) -> EmotionSegment:
        """
        Expand segment to include time_tolerance before and after feeling_it zones.
        """
        
        if len(segment.feeling_it_indices) == 0:
            return segment
        
        tolerance_samples = int(time_tolerance_s * self.sr)
        
        # Find first and last feeling_it indices
        first_feeling = segment.feeling_it_indices[0]
        last_feeling = segment.feeling_it_indices[-1]
        
        # Expand boundaries
        new_start_idx = max(0, first_feeling - tolerance_samples)
        new_end_idx = min(len(list(segment.signals.values())[0]), 
                         last_feeling + tolerance_samples + 1)
        
        # Create new segment with expanded boundaries
        new_signals = {}
        for signal_name, signal_data in segment.signals.items():
            new_signals[signal_name] = signal_data[new_start_idx:new_end_idx]
        
        # Recalculate feeling_it stats
        # Note: feeling_it_indices are relative to segment start, need to adjust
        adjusted_indices = [idx for idx in segment.feeling_it_indices 
                           if new_start_idx <= idx < new_end_idx]
        adjusted_indices = [idx - new_start_idx for idx in adjusted_indices]
        
        new_duration = (new_end_idx - new_start_idx) / self.sr
        new_feeling_ratio = len(adjusted_indices) / (new_end_idx - new_start_idx)
        
        return EmotionSegment(
            segment_id=segment.segment_id + "_expanded",
            session_id=segment.session_id,
            emotion=segment.emotion,
            start_idx=segment.start_idx + new_start_idx,
            end_idx=segment.start_idx + new_end_idx,
            start_time=segment.start_time + new_start_idx / self.sr,
            end_time=segment.start_time + new_end_idx / self.sr,
            duration=new_duration,
            signals=new_signals,
            feeling_it=len(adjusted_indices) > 0,
            feeling_it_ratio=new_feeling_ratio,
            feeling_it_indices=adjusted_indices,
            metadata=segment.metadata
        )
    
    def extract_feeling_zones(self, 
                             segments: List[EmotionSegment],
                             min_zone_duration_s: float = 1.0) -> List[EmotionSegment]:
        """
        Extract continuous zones where feeling_it==1.
        Splits segments at feeling_it boundaries.
        
        Parameters
        ----------
        segments : List[EmotionSegment]
            Input segments
        min_zone_duration_s : float
            Minimum duration of a feeling_it zone to keep
            
        Returns
        -------
        feeling_zones : List[EmotionSegment]
            Segments corresponding to continuous feeling_it==1 zones
        """
        
        feeling_zones = []
        
        for seg in segments:
            if not seg.feeling_it:
                continue
            
            # Find continuous runs of feeling_it==1
            feeling_array = np.zeros(len(list(seg.signals.values())[0]))
            feeling_array[seg.feeling_it_indices] = 1
            
            # Detect zone boundaries
            changes = np.diff(np.concatenate([[0], feeling_array, [0]]))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            
            # Create segment for each zone
            for zone_idx, (start, end) in enumerate(zip(starts, ends)):
                zone_duration = (end - start) / self.sr
                
                if zone_duration < min_zone_duration_s:
                    continue
                
                # Extract signals for this zone
                zone_signals = {}
                for signal_name, signal_data in seg.signals.items():
                    zone_signals[signal_name] = signal_data[start:end]
                
                zone = EmotionSegment(
                    segment_id=f"{seg.segment_id}_zone{zone_idx:02d}",
                    session_id=seg.session_id,
                    emotion=seg.emotion,
                    start_idx=seg.start_idx + start,
                    end_idx=seg.start_idx + end,
                    start_time=seg.start_time + start / self.sr,
                    end_time=seg.start_time + end / self.sr,
                    duration=zone_duration,
                    signals=zone_signals,
                    feeling_it=True,
                    feeling_it_ratio=1.0,
                    feeling_it_indices=list(range(end - start)),
                    metadata=seg.metadata
                )
                
                feeling_zones.append(zone)
        
        print(f"Extract feeling zones: Found {len(feeling_zones)} continuous zones "
              f"(min duration: {min_zone_duration_s}s)")
        
        return feeling_zones
    
    # ========================================================================
    # 4. TEMPORAL WINDOWING
    # ========================================================================
    
    def create_fixed_windows(self,
                            segments: List[EmotionSegment],
                            window_size_s: float = 30.0,
                            overlap_s: float = 0.0,
                            min_feeling_ratio: float = 0.0) -> List[EmotionSegment]:
        """
        Create fixed-size overlapping windows from segments.
        
        Parameters
        ----------
        segments : List[EmotionSegment]
            Input segments
        window_size_s : float
            Window size in seconds
        overlap_s : float
            Overlap between consecutive windows in seconds
        min_feeling_ratio : float
            Minimum proportion of window that must have feeling_it==1
            
        Returns
        -------
        windows : List[EmotionSegment]
            Fixed-size windows
        """
        
        window_samples = int(window_size_s * self.sr)
        step_samples = int((window_size_s - overlap_s) * self.sr)
        
        if step_samples <= 0:
            raise ValueError("Overlap must be less than window size")
        
        windows = []
        
        for seg in segments:
            segment_length = len(list(seg.signals.values())[0])
            
            # Skip if segment is shorter than window
            if segment_length < window_samples:
                continue
            
            # Extract windows
            start_idx = 0
            window_count = 0
            
            while start_idx + window_samples <= segment_length:
                end_idx = start_idx + window_samples
                
                # Extract window signals
                window_signals = {}
                for signal_name, signal_data in seg.signals.items():
                    window_signals[signal_name] = signal_data[start_idx:end_idx]
                
                # Calculate feeling_it for this window
                window_feeling_indices = [idx - start_idx for idx in seg.feeling_it_indices
                                         if start_idx <= idx < end_idx]
                window_feeling_ratio = len(window_feeling_indices) / window_samples
                
                # Check minimum feeling ratio
                if window_feeling_ratio < min_feeling_ratio:
                    start_idx += step_samples
                    continue
                
                # Create window segment
                window = EmotionSegment(
                    segment_id=f"{seg.segment_id}_win{window_count:03d}",
                    session_id=seg.session_id,
                    emotion=seg.emotion,
                    start_idx=seg.start_idx + start_idx,
                    end_idx=seg.start_idx + end_idx,
                    start_time=seg.start_time + start_idx / self.sr,
                    end_time=seg.start_time + end_idx / self.sr,
                    duration=window_size_s,
                    signals=window_signals,
                    feeling_it=len(window_feeling_indices) > 0,
                    feeling_it_ratio=window_feeling_ratio,
                    feeling_it_indices=window_feeling_indices,
                    metadata=seg.metadata
                )
                
                windows.append(window)
                window_count += 1
                start_idx += step_samples
        
        print(f"Create fixed windows: {len(windows)} windows "
              f"(size={window_size_s}s, overlap={overlap_s}s, min_feeling={min_feeling_ratio:.2f})")
        
        return windows
    
    # ========================================================================
    # 5. QUALITY CONTROL
    # ========================================================================
    
    def filter_by_quality(self,
                         segments: List[EmotionSegment],
                         min_duration_s: float = 10.0,
                         max_duration_s: Optional[float] = None,
                         check_signal_validity: bool = True,
                         max_flat_ratio: float = 0.5) -> List[EmotionSegment]:
        """
        Filter segments by quality criteria.
        
        Parameters
        ----------
        segments : List[EmotionSegment]
            Input segments
        min_duration_s : float
            Minimum segment duration
        max_duration_s : float, optional
            Maximum segment duration (if provided)
        check_signal_validity : bool
            Check for flat/invalid signals
        max_flat_ratio : float
            Maximum allowed proportion of flat signal (0.0 to 1.0)
            
        Returns
        -------
        filtered : List[EmotionSegment]
            Quality-filtered segments
        """
        
        filtered = []
        
        for seg in segments:
            # Duration check
            if seg.duration < min_duration_s:
                continue
            
            if max_duration_s is not None and seg.duration > max_duration_s:
                continue
            
            # Signal validity check
            if check_signal_validity:
                if not self._check_signal_validity(seg, max_flat_ratio):
                    continue
            
            filtered.append(seg)
        
        print(f"Quality filter: Kept {len(filtered)}/{len(segments)} segments "
              f"(min_duration={min_duration_s}s, check_validity={check_signal_validity})")
        
        return filtered
    
    def _check_signal_validity(self,
                               segment: EmotionSegment,
                               max_flat_ratio: float) -> bool:
        """
        Check if signals in segment are valid (not flat, not all zeros, etc.)
        """
        
        for signal_name, signal_data in segment.signals.items():
            # Check for all zeros
            if np.all(signal_data == 0):
                return False
            
            # Check for flat signal (no variation)
            if np.std(signal_data) < 1e-6:
                return False
            
            # Check for excessive flatness (too many consecutive identical values)
            diff = np.diff(signal_data)
            flat_samples = np.sum(np.abs(diff) < 1e-6)
            flat_ratio = flat_samples / len(diff)
            
            if flat_ratio > max_flat_ratio:
                return False
        
        return True
    
    # ========================================================================
    # 6. SUMMARY & STATISTICS
    # ========================================================================
    
    def get_summary(self, segments: List[EmotionSegment]) -> pd.DataFrame:
        """
        Get summary statistics for segments.
        
        Returns
        -------
        summary : pd.DataFrame
            Summary table with emotion counts, durations, feeling_it stats
        """
        
        if len(segments) == 0:
            return pd.DataFrame()
        
        summary_data = []
        
        # Group by emotion
        emotions = set(seg.emotion for seg in segments)
        
        for emotion in sorted(emotions):
            emotion_segs = [seg for seg in segments if seg.emotion == emotion]
            
            total_duration = sum(seg.duration for seg in emotion_segs)
            avg_duration = np.mean([seg.duration for seg in emotion_segs])
            
            feeling_segs = [seg for seg in emotion_segs if seg.feeling_it]
            feeling_duration = sum(seg.duration for seg in feeling_segs)
            avg_feeling_ratio = np.mean([seg.feeling_it_ratio for seg in emotion_segs])
            
            summary_data.append({
                'emotion': emotion,
                'n_segments': len(emotion_segs),
                'total_duration_s': total_duration,
                'avg_duration_s': avg_duration,
                'n_with_feeling': len(feeling_segs),
                'feeling_duration_s': feeling_duration,
                'avg_feeling_ratio': avg_feeling_ratio
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Add totals
        totals = {
            'emotion': 'TOTAL',
            'n_segments': len(segments),
            'total_duration_s': sum(seg.duration for seg in segments),
            'avg_duration_s': np.mean([seg.duration for seg in segments]),
            'n_with_feeling': sum(1 for seg in segments if seg.feeling_it),
            'feeling_duration_s': sum(seg.duration for seg in segments if seg.feeling_it),
            'avg_feeling_ratio': np.mean([seg.feeling_it_ratio for seg in segments])
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([totals])], ignore_index=True)
        
        return summary_df
    
    def print_summary(self, segments: List[EmotionSegment]) -> None:
        """Print formatted summary of segments."""
        
        summary = self.get_summary(segments)
        
        print("\n" + "="*80)
        print("SEGMENT SUMMARY")
        print("="*80)
        print(summary.to_string(index=False))
        print("="*80)


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Example usage and testing of DataSlicer module.
    """
    
    print("="*80)
    print("DATA SLICER MODULE - EXAMPLE USAGE")
    print("="*80)
    
    # Example 1: Load a session and convert to segments
    print("\n" + "="*80)
    print("EXAMPLE 1: Convert session to segments")
    print("="*80)
    
    # Load sample data
    data = pd.read_csv('/mnt/user-data/uploads/sample_emotion_biodata.csv')
    
    slicer = DataSlicer(sampling_rate=100)
    
    # Convert to segments
    segments = slicer.session_to_segments(
        data,
        session_id='sample_session',
        emotion_col='emotion',
        feeling_col='feeling_it',
        signal_cols=['heart', 'gsr', 'respiration']
    )
    
    print(f"\nCreated {len(segments)} segments:")
    for seg in segments[:5]:  # Show first 5
        print(f"  {seg}")
    
    slicer.print_summary(segments)
    
    # Example 2: Filter by emotions
    print("\n" + "="*80)
    print("EXAMPLE 2: Filter by emotions")
    print("="*80)
    
    # Keep only specific emotions (exclude 'nul' baseline)
    filtered = slicer.filter_by_emotions(
        segments,
        exclude_emotions=['nul']
    )
    
    slicer.print_summary(filtered)
    
    # Example 3: Extract feeling_it zones
    print("\n" + "="*80)
    print("EXAMPLE 3: Extract feeling_it zones")
    print("="*80)
    
    # Note: This sample has no feeling_it==1, so this will return empty
    feeling_zones = slicer.extract_feeling_zones(
        segments,
        min_zone_duration_s=1.0
    )
    
    print(f"Found {len(feeling_zones)} feeling_it zones")
    
    # Example 4: Create fixed windows
    print("\n" + "="*80)
    print("EXAMPLE 4: Create fixed windows")
    print("="*80)
    
    windows = slicer.create_fixed_windows(
        segments,
        window_size_s=5.0,  # Small windows for this short sample
        overlap_s=2.5,       # 50% overlap
        min_feeling_ratio=0.0  # No feeling_it requirement for demo
    )
    
    print(f"\nCreated {len(windows)} windows:")
    for win in windows[:5]:
        print(f"  {win}")
    
    slicer.print_summary(windows)
    
    # Example 5: Quality filtering
    print("\n" + "="*80)
    print("EXAMPLE 5: Quality filtering")
    print("="*80)
    
    quality_segments = slicer.filter_by_quality(
        segments,
        min_duration_s=2.0,
        check_signal_validity=True,
        max_flat_ratio=0.5
    )
    
    print(f"After quality filter: {len(quality_segments)}/{len(segments)} segments")
    
    print("\n" + "="*80)
    print("MODULE 2 TESTING COMPLETE")
    print("="*80)
