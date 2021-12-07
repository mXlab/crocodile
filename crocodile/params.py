from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from simple_parsing.helpers.serialization import register_decoding_fn
from pathlib import Path
from typing import Optional
from crocodile.dataset import LaurenceDataset
from crocodile.utils.optim import OptimizerArgs, OptimizerType
from crocodile.utils.loss import LossParams
import os


@dataclass
class ComputeLatentParams(Serializable):
    generator_path: Path
    epoch: Optional[int] = None
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    batch_size: int = 64
    optimizer: OptimizerArgs = OptimizerArgs()
    loss: LossParams = LossParams()
    num_epochs: int = 100
    log_dir: Path = Path("./results/latent")
    name: str = "test_1"
    num_test_samples: int = 10
    slurm_job_id: Optional[str] = os.environ.get('SLURM_JOB_ID')
    debug: bool = False

    def __post_init__(self):
        self.save_dir = self.log_dir / self.name
        if self.optimizer.lr is None:
            if self.optimizer.optimizer == OptimizerType.SGD:
                self.optimizer.lr = 20
            elif self.optimizer.optimizer == OptimizerType.ADAM:
                self.optimizer.lr = 2e-2


register_decoding_fn(Path, Path)