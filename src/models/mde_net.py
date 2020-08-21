import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet101
from src.data import IMG_WIDTH


# ============================= Feature Pyramid Network ================================= #
# based on source: https://github.com/wolverinn/Depth-Estimation-PyTorch

class print_size(nn.Module):
    def forward(self, x):
        print(x.size())
        return x


class flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size()[0], -1)


def predict(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU()
    )


def predict_range(inp, out):
    return nn.Sequential(nn.Linear(inp, out),
                         nn.ReLU(),
                         nn.Linear(out, 2)
                         )


class FPNNet(nn.Module):
    def __init__(self, num_channels=3, pretrained=True):
        super(FPNNet, self).__init__()

        resnet = resnet101(pretrained=pretrained)

        self.img_channels = num_channels
        self.inplanes = 64

        self.conv1 = nn.Conv2d(self.img_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.layer0 = nn.Sequential(self.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)  # 256
        self.layer2 = nn.Sequential(resnet.layer2)  # 512
        self.layer3 = nn.Sequential(resnet.layer3)  # 1024

        # Top layer
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Upconvolve to get bigger image (for 256x256 case)
        self.upconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        # Depth prediction
        self.predict1 = predict(256, 64)
        self.predict2 = predict(64, 1)
        self.predict_range = predict_range(128 * 128 * 128, 128)

        self.print = print_size()
        self.flatten = flatten()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Bilinear upsample supports arbitrary output sizes.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        _, _, H, W = x.size()  # batchsize N, channel, height, width

        # Bottom-up
        c1 = self.layer0(x)
        c2 = self.layer1(c1)  # 256 channels
        c3 = self.layer2(c2)  # 512 channels
        c4 = self.layer3(c3)  # 1024 channels

        # Top-down
        p5 = self.toplayer(c4)  # 256 channels
        p4 = self._upsample_add(p5, self.latlayer1(c4))  # 256 channels
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))  # 256 channels
        p3 = self.smooth2(p3)  # 256 channels
        p2 = self._upsample_add(p3, self.latlayer3(c2))  # 256 channels - 128
        p2 = self.smooth3(p2)  # 256 channels
        if IMG_WIDTH == 256:
            p2 = self.upconv(p2)
        # flat = self.flatten(p2)
        # range = self.predict_range(flat)
        return self.predict2(self.predict1(p2))  # , range
