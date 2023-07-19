from pathlib import Path
from peewee import (
    Model,
    CharField,
    SqliteDatabase,
    Field,
    ForeignKeyField,
    IntegerField,
    FloatField,
    UUIDField,
)
from enum import Enum
from datetime import datetime

from simple_parsing import Serializable
import torch
from typing import Optional
from torch import nn

db = SqliteDatabase(None)  # Un-initialized database.


class BaseModel(Model):
    class Meta:
        database = db

class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"


class DateTimeField(Field):
    field_type = "datetime"

    def db_value(self, value: datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S-00:00")

    def python_value(self, value: str):
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S-00:00")


class StatusField(Field):
    field_type = "status"

    def db_value(self, value):
        return value.value

    def python_value(self, value):
        return Status(value)


class ExperimentTable(BaseModel):
    experiment_id = UUIDField()
    name = CharField()
    created_at = DateTimeField()
    path = CharField()
    status = StatusField(default=Status.PENDING)
    updated_at = DateTimeField(null=True)

    class Meta:
        table_name = "experiment"


class Model(BaseModel):
    name = CharField()
    type = CharField()
    created_at = DateTimeField()
    uploaded_at = DateTimeField(null=True)
    path = CharField()
    experiment = ForeignKeyField(ExperimentTable, backref="models")
    iteration = IntegerField()
    fid = FloatField(null=True)

    class Meta:
        table_name = "model"

    @classmethod
    def save(cls, model: nn.Module, name: str, model_type: str, path: Path, experiment: ExperimentTable, iteration: int, fid: Optional[float] = None):
        path = path / f"{name}-{iteration:06}.pth"
        torch.save(model.state_dict(), path)

        model_entry = cls.create(
            name=name,
            type=model_type,
            created_at=datetime.now(),
            path=path,
            experiment=experiment,
            iteration=iteration,
            fid=fid,
        )
        model_entry.save()

    @classmethod
    def load(cls, model_id: int):
        model = cls.get_by_id(model_id)
        return torch.load(model.path)

class Experiment:
    def __init__(self, experiment: ExperimentTable):
        self.experiment = experiment
        self.models_path.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def create(name: str, exp_id: str, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        experiment = ExperimentTable.create(
            experiment_id=exp_id, name=name, created_at=datetime.now(), path=path
        )
        experiment.save()
        return Experiment(experiment)

    @property
    def models_path(self) -> Path:
        return self.experiment.path / "models"

    def start(self):
        self.experiment.status = Status.RUNNING  # TODO: Fix type error
        self.experiment.updated_at = datetime.now()  # TODO: Fix type error
        self.experiment.save()

    def save_model(
        self,
        name: str,
        model_type: str,
        model: nn.Module,
        iteration: int,
        fid: Optional[float] = None,
    ):
        Model.save(model, name, model_type, self.models_path, self.experiment, iteration, fid)
        self.experiment.updated_at = datetime.now()  # TODO: Fix type error
        self.experiment.save()

    def get_root_dir(self):
        return self.experiment.path

    def end(self):
        self.experiment.status = Status.FINISHED
        self.experiment.updated_at = datetime.now()  # TODO: Fix type error
        self.experiment.save()

class Database:
    def __init__(self, db_name: str):
        db.init(db_name)
        db.connect()
        db.create_tables([ExperimentTable, Model])

    @staticmethod
    def create_experiment(
        name: str, exp_id: str, path: Path, config: Serializable
    ):
        experiment = Experiment.create(name, exp_id, path)
        config.save(path / "config.yaml")
        return experiment
    
    @staticmethod
    def load_model(model_id: int):
        return Model.load(model_id)


