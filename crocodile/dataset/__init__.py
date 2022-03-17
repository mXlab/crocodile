from .laurence import LaurenceDataset
from .latent import LatentDataset
from .biodata import Biodata
from pathlib import Path
from typing import Tuple


def load_biodata(path: Path, sampling_rate: int = 1000, usecols: Tuple[int,...] = (1,2)):
    return Biodata(path, sampling_rate, usecols)