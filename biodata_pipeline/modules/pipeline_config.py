"""
Configuration Management for Biodata Pipeline

Supports YAML config files for easy experimentation.

Usage:
    # Use a config file
    python scripts/slice_and_evaluate.py --config configs/best_2emo.yaml
    
    # Override config values
    python scripts/slice_and_evaluate.py --config configs/base.yaml --window-size 25
    
    # Generate config from command line
    python scripts/generate_config.py --include fea sad --window-size 30 --output configs/fea_sad.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


class PipelineConfig:
    """Pipeline configuration with defaults."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Parameters
        ----------
        config_dict : dict, optional
            Configuration dictionary (from YAML or command line)
        """
        
        # Set defaults
        self.config = {
            # Input
            'features_file': 'data/processed/continuous_features.csv',
            
            # Emotion filtering
            'include_emotions': None,  # List of emotions to include
            'exclude_emotions': None,  # List of emotions to exclude
            
            # Windowing parameters
            'window_size_s': 30.0,
            'stride_s': 5.0,
            'min_purity': 0.7,
            'super_segment_size_s': None,  # None = no super-segments
            
            # Cross-validation
            'n_folds': 5,
            'classifier': 'random_forest',
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
            
            # Output
            'output_file': None,  # Auto-generate if None
            'save_model': False,
            'model_output': 'models/trained_model.pkl',
            
            # Flags
            'evaluate': True,
            'verbose': True,
        }
        
        # Update with provided config
        if config_dict:
            self.config.update(config_dict)
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def to_dict(self):
        return self.config.copy()
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def to_yaml(self, yaml_path: Path):
        """Save configuration to YAML file."""
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'PipelineConfig':
        """Create configuration from command-line arguments."""
        
        config_dict = {}
        
        # Map args to config keys
        arg_mapping = {
            'features': 'features_file',
            'include': 'include_emotions',
            'exclude': 'exclude_emotions',
            'window_size': 'window_size_s',
            'stride': 'stride_s',
            'min_purity': 'min_purity',
            'super_segment_size': 'super_segment_size_s',
            'n_folds': 'n_folds',
            'output': 'output_file',
            'no_evaluate': 'evaluate',
        }
        
        for arg_name, config_key in arg_mapping.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    # Special handling for no_evaluate (inverted)
                    if arg_name == 'no_evaluate':
                        config_dict[config_key] = not value
                    else:
                        config_dict[config_key] = value
        
        return cls(config_dict)
    
    def merge_with_args(self, args: argparse.Namespace):
        """Merge command-line args into config (args override config)."""
        
        arg_config = self.from_args(args)
        
        # Override only non-None values from args
        for key, value in arg_config.to_dict().items():
            if value is not None:
                self.config[key] = value
    
    def validate(self) -> bool:
        """Validate configuration."""
        
        errors = []
        
        # Check mutually exclusive
        if self.config['include_emotions'] and self.config['exclude_emotions']:
            errors.append("Cannot specify both include_emotions and exclude_emotions")
        
        # Check ranges
        if self.config['window_size_s'] <= 0:
            errors.append("window_size_s must be positive")
        
        if self.config['stride_s'] <= 0:
            errors.append("stride_s must be positive")
        
        if not (0 <= self.config['min_purity'] <= 1):
            errors.append("min_purity must be between 0 and 1")
        
        if self.config['n_folds'] < 2:
            errors.append("n_folds must be at least 2")
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def summary(self) -> str:
        """Return human-readable configuration summary."""
        
        lines = []
        lines.append("Configuration:")
        lines.append(f"  Input: {self.config['features_file']}")
        
        if self.config['include_emotions']:
            lines.append(f"  Emotions: {', '.join(self.config['include_emotions'])}")
        elif self.config['exclude_emotions']:
            lines.append(f"  Exclude: {', '.join(self.config['exclude_emotions'])}")
        else:
            lines.append(f"  Emotions: All")
        
        lines.append(f"  Window: {self.config['window_size_s']}s, stride {self.config['stride_s']}s")
        lines.append(f"  Min purity: {self.config['min_purity']*100:.0f}%")
        
        if self.config['super_segment_size_s']:
            lines.append(f"  Super-segments: {self.config['super_segment_size_s']}s")
        else:
            lines.append(f"  Super-segments: Disabled")
        
        lines.append(f"  Cross-validation: {self.config['n_folds']}-fold")
        lines.append(f"  Classifier: {self.config['classifier']} (n={self.config['n_estimators']})")
        
        return "\n".join(lines)


def load_config(config_path: Optional[Path] = None, 
               args: Optional[argparse.Namespace] = None) -> PipelineConfig:
    """
    Load configuration with priority: args > config file > defaults.
    
    Parameters
    ----------
    config_path : Path, optional
        Path to YAML config file
    args : Namespace, optional
        Command-line arguments
    
    Returns
    -------
    PipelineConfig
        Loaded configuration
    """
    
    if config_path and config_path.exists():
        # Load from YAML
        config = PipelineConfig.from_yaml(config_path)
        print(f"✓ Loaded config from: {config_path}")
        
        # Override with command-line args
        if args:
            config.merge_with_args(args)
            print(f"✓ Applied command-line overrides")
    
    elif args:
        # Create from args only
        config = PipelineConfig.from_args(args)
    
    else:
        # Use defaults
        config = PipelineConfig()
    
    return config


if __name__ == "__main__":
    # Test configuration system
    
    print("="*80)
    print("CONFIGURATION SYSTEM TEST")
    print("="*80)
    
    # Create test config
    config = PipelineConfig({
        'include_emotions': ['fea', 'sad'],
        'window_size_s': 30,
        'stride_s': 2,
        'super_segment_size_s': 45
    })
    
    print("\nTest config:")
    print(config.summary())
    
    # Save to YAML
    test_path = Path('configs/test_config.yaml')
    test_path.parent.mkdir(exist_ok=True)
    config.to_yaml(test_path)
    print(f"\n✓ Saved to: {test_path}")
    
    # Load back
    loaded_config = PipelineConfig.from_yaml(test_path)
    print(f"\n✓ Loaded back from YAML")
    print(loaded_config.summary())
