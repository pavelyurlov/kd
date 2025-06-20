from pydantic import BaseModel


class ModelConfig(BaseModel):
    teacher: str
    student: str


class DataConfig(BaseModel):
    name: str
    language: str = "ru"
    sonnet_easy: bool = True
    dummy: bool = False
    filter: int = 1


class TrainConfig(BaseModel):
    epochs: int = 4
    inference_batch: int = 256
    lr: float = 3e-6
    accum_steps: int = 4


class Config(BaseModel):
    models: ModelConfig
    data: DataConfig
    train: TrainConfig
