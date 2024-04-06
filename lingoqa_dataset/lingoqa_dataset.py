import os
from abc import ABCMeta
from pathlib import Path

from torch.utils.data import Dataset


class DatasetInfo(ABCMeta):
    gdown_object_id: str
    zip_filename: str = "images.zip"
    parquet_filename: str


class SceneryDataset(DatasetInfo):
    gdown_object_id = "1GiwWGfrM8pO27CYLu_9Uwtdcz0JoqHr7"
    parquet_filename = "train.parquet"


class ActionDataset(DatasetInfo):
    gdown_object_id = "1QQqBrR3uGDC05Zc11zMeui6Zzl7RvFZg"
    parquet_filename = "train.parquet"


class EvaluationDataset(DatasetInfo):
    gdown_object_id = "1oA7W8-Ej_uJEuUxZIjPP5K8hQGGzYsPq"
    parquet_filename = "val.parquet"


class LingoQADataset(Dataset):
    lingoqa_dataset_root_dir: Path

    def __init__(self) -> None:
        self.lingoqa_dataset_root_dir = Path(
            os.environ.get("LINGOQA_DATASET_ROOT_DIR", "/tmp")
        )


if __name__ == "__main__":
    dataset = LingoQADataset()
