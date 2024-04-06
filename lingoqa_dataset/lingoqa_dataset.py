import os
from abc import ABC
from enum import Enum
from pathlib import Path

import gdown
from torch.utils.data import Dataset


class DatasetInfo(ABC):
    gdown_object_id: str
    zip_filename: str = "images.zip"
    parquet_filename: str
    save_directory: str


class SceneryDatasetInfo(DatasetInfo):
    gdown_object_id = "1GiwWGfrM8pO27CYLu_9Uwtdcz0JoqHr7"
    parquet_filename = "train.parquet"
    save_directory = "scenery"


class ActionDatasetInfo(DatasetInfo):
    gdown_object_id = "1QQqBrR3uGDC05Zc11zMeui6Zzl7RvFZg"
    parquet_filename = "train.parquet"
    save_directory = "action"


class EvaluationDatasetInfo(DatasetInfo):
    gdown_object_id = "1oA7W8-Ej_uJEuUxZIjPP5K8hQGGzYsPq"
    parquet_filename = "val.parquet"
    save_directory = "evaluation"


class DatasetType(Enum):
    SCENARY = 0
    ACTION = 1
    EVALUATION = 2


class LingoQADataset(Dataset):
    lingoqa_dataset_root_dir: Path
    dataset_info: DatasetInfo

    def __init__(self, type: DatasetType) -> None:
        self.lingoqa_dataset_root_dir = Path(
            os.environ.get("LINGOQA_DATASET_ROOT_DIR", "/tmp/lingoqa_dataset")
        )
        if type == DatasetType.SCENARY:
            self.dataset_info = SceneryDatasetInfo()
        elif type == DatasetType.ACTION:
            self.dataset_info = ActionDatasetInfo()
        elif type == DatasetType.EVALUATION:
            self.dataset_info = EvaluationDatasetInfo()
        else:
            raise Exception(
                "Dataset type should be scenary/action/evaluation. \
                    Please check type of the dataset."
            )
        self.download()

    def dataset_downloaded(self) -> bool:
        return Path.exists(
            self.lingoqa_dataset_root_dir.joinpath(self.dataset_info.zip_filename)
        ) and Path.exists(
            self.lingoqa_dataset_root_dir.joinpath(self.dataset_info.parquet_filename)
        )

    def download(self):
        if not self.dataset_downloaded():
            os.makedirs(self.lingoqa_dataset_root_dir, exist_ok=True)
            gdown.download_folder(
                id=self.dataset_info.gdown_object_id,
                quiet=False,
                output=str(self.lingoqa_dataset_root_dir),
            )


if __name__ == "__main__":
    dataset = LingoQADataset(DatasetType.EVALUATION)
