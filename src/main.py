import argparse
import sys

import torch
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from dataloader import SiameseDataset
from network import SiameseNN

EPOCHS = 25


def main():
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')

    parser.add_argument('dataset', help='Dataset')
    parser.add_argument('--cpu', help='Use cpu', action='store_true')

    args = parser.parse_args()

    if not args.cpu:
        print('Device: {}'.format(torch.cuda.get_device_name(0)))

    train_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = SiameseDataset(args.dataset, train_transform)
    train_loader = DataLoader(train_set, batch_size=40, shuffle=True, num_workers=4)
    print(train_set)

    dataloaders = {'train': train_loader, 'val': train_loader}
    dataset_sizes = {'train': len(train_set), 'val': len(train_loader)}

    model = SiameseNN()
    if not args.cpu:
        model = model.cuda()
    optimizer = Adam(model.parameters())
    scheduler = StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    for epoch in tqdm(range(EPOCHS), desc="Training...", file=sys.stdout):
        for i, data in tqdm(enumerate(train_loader, 0), desc='Step...', file=sys.stdout):
            input, labels = data
            print(labels)


if __name__ == '__main__':
    main()
