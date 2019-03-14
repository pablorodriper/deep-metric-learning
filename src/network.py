from torch import nn
from torchvision import models


class SiameseNet(nn.Module):

    def __init__(self, num_dimensions: int = 16):
        super().__init__()

        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(2048, num_dimensions)

    def forward(self, *data):
        res = []
        for i in range(2):
            res.append(self.model.forward(data[i]))
        return res
