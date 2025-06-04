from pydantic import BaseModel


class ModelConfig(BaseModel):
    teacher: str
    student: str

class DataConfig(BaseModel):
    name: str
    dummy: bool = False
    filter: int = 1

class TrainConfig(BaseModel):
    epochs: int = 1
    inference_batch: int = 512

class Config(BaseModel):
    models: ModelConfig
    data: DataConfig
    train: TrainConfig
