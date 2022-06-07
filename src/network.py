from torch import nn
from torchvision.models import resnet50

model_cfg = {
    3: [(3, 3), (2, 2), (1, 1)],
    5: [(5, 5), (2, 2), (2, 2)],
    7: [(7, 7), (2, 2), (3, 3)],
    9: [(9, 9), (2, 2), (4, 4)]
}


class Resnet50(nn.Module):
    def __init__(self, kernel_size):
        super(Resnet50, self).__init__()
        cfg = model_cfg[kernel_size]
        self.net = resnet50(num_classes=1623)
        self.net.conv1 = nn.Conv2d(1, self.net.bn1.num_features,
                                   kernel_size=cfg[0], stride=cfg[1],
                                   padding=cfg[2], bias=False)

    def forward(self, x):
        return self.net(x)
