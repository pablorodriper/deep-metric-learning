import argparse
import sys

from PIL import Image
from tqdm import tqdm

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from dataloader import SiameseDataset
from network import SiameseNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--epochs', type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()

    cuda = torch.cuda.is_available()
    if cuda:
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

    train_set = SiameseDataset(args.dataset_dir, train_transform)
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(train_set, batch_size=40, shuffle=True, **kwargs)
    print(train_set)

    dataloaders = {'train': train_loader, 'val': train_loader}
    dataset_sizes = {'train': len(train_set), 'val': len(train_loader)}

    model = SiameseNet()
    if cuda:
        model = model.cuda()
    optimizer = Adam(model.parameters())
    scheduler = StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    for epoch in tqdm(range(1, args.epochs+1), desc="Training", file=sys.stdout):
        for batch_idx, data in tqdm(enumerate(train_loader), desc='Epoch {}/{}'.format(epoch, args.epochs), total=len(train_loader), file=sys.stdout):
            sample, target = data
            print(target)


if __name__ == '__main__':
    main()
