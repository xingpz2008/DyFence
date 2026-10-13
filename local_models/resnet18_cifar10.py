import torchvision.models as models
import torch.nn as nn

class ResNet18_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights=None)
        self.base.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.base(x)