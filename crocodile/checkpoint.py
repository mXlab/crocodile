from typing import Any, Dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import mlflow


class MLFlowCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        epoch = trainer.current_epoch
        step = trainer.global_step
        filename = f"epoch={epoch}-step={step}"

        mlflow.pytorch.log_model(pl_module, filename)
