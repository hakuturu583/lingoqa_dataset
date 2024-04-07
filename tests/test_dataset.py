import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from lingoqa_dataset.lingoqa_dataset import DatasetType, LingoQADataset


def test_read_images_and_qa_pairs() -> None:
    dataset = LingoQADataset(
        DatasetType.EVALUATION, transforms=transforms.Resize((256, 512))
    )
    dataloader = DataLoader(dataset=dataset, batch_size=3, shuffle=True)
    assert len(dataloader) == 334
    for data, question, answer in dataloader:
        pass
