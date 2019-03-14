import argparse
import sys

from PIL import Image
from tqdm import tqdm, trange

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from dataloader import SiameseDataset
from network import SiameseNet
from loss import ContrastiveLoss


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
    train_loader = DataLoader(train_set, batch_size=40, shuffle=True, num_workers=4)
    print(train_set)

    model = SiameseNet()
    if cuda:
        model = model.cuda()

    criterion = ContrastiveLoss(margin=1.)
    optimizer = Adam(model.parameters())
    scheduler = StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    for epoch in trange(1, args.epochs+1, desc="Training", file=sys.stdout):
        scheduler.step()
        with tqdm(enumerate(train_loader), desc='Epoch {}/{}'.format(epoch, args.epochs), total=len(train_loader), file=sys.stdout) as pbar:
            for batch_idx, data in pbar:
                (sample1, sample2), target = data
                if cuda:
                    sample1 = sample1.cuda()
                    sample2 = sample2.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output1, output2 = model(sample1, sample2)

                loss = criterion(output1, output2, target)
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())


if __name__ == '__main__':
    main()
