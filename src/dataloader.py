from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose


class SiameseDataset(Dataset):

    def __init__(self, path: str, transforms: Compose, train=True):
        self.path = path
        self.train = train

        self.dataset = datasets.ImageFolder(path, transform=transforms)

    def __getitem__(self, index):
        # TODO: Hace parejas, debería hacer permutaciones. Revisar utils

        pos1 = index // len(self.dataset)
        pos2 = index % len(self.dataset)

        i0 = self.dataset[pos1]
        i1 = self.dataset[pos2]

        return (i0[0], i1[0]), i0[1] == i1[1]

    def __len__(self):
        return len(self.dataset) ** 2
