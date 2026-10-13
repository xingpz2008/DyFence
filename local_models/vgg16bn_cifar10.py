import torchvision.models as models
import torch.nn as nn

class VGG16_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.vgg16_bn(weights=None)
        
        self.base.avgpool = nn.Identity()
        
        self.base.classifier = nn.Sequential(
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.base(x)
