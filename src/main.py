import os

import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torchvision.transforms import transforms

from network import SiameseNN


def main():
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')

    parser.add_argument('dataset', help='Dataset')

    args = parser.parse_args()

    print('Device: {}'.format(torch.cuda.get_device_name(0)))

    train_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(1., 1., 1.))
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(1., 1., 1.))
    ])

    trainset = datasets.ImageFolder(args.dataset, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=40, shuffle=True, num_workers=4)
    print(trainset)

    dataloaders = {'train': trainloader, 'val': trainloader}
    dataset_sizes = {'train': len(trainset), 'val': len(trainloader)}
    num_classes = len(trainset.classes)

    model = SiameseNN()


if __name__ == '__main__':
    main()
