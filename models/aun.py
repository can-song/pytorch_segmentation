import math
import torch
import torch.nn.functional as F
from torch import nn
# from models import resnet
from torchvision import models
from base import BaseModel
from utils.helpers import initialize_weights, set_trainable
from itertools import chain
from .modules import PPM, DAUM, AUM, GaussianSmoother


class AUNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='resnet152', pretrained=True, use_aux=True, freeze_bn=False,
                 freeze_backbone=False):
        super().__init__()
        # TODO: Use synch batchnorm
        norm_layer = nn.BatchNorm2d
        # model = getattr(resnet, backbone)(pretrained, norm_layer=norm_layer)
        model = getattr(models, backbone)(pretrained)

        self.initial = nn.Sequential(*list(model.children())[:4])
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        out_size1 = model.layer1[-1].bn3.num_features
        out_size2 = model.layer2[-1].bn3.num_features
        out_size3 = model.layer3[-1].bn3.num_features
        out_size4 = model.layer4[-1].bn3.num_features

        bin_sizes = [1, 2, 3, 6]
        self.ppm4 = nn.Sequential(
            PPM(out_size4, bin_sizes, norm_layer=norm_layer),
            nn.Conv2d(out_size4 // len(bin_sizes), num_classes, kernel_size=1)
        )
        self.ppm3 = nn.Sequential(
            PPM(out_size3, bin_sizes, norm_layer=norm_layer),
            nn.Conv2d(out_size3//len(bin_sizes), num_classes, kernel_size=1)
        )
        self.ppm2 = nn.Sequential(
            PPM(out_size2, bin_sizes, norm_layer=norm_layer),
            nn.Conv2d(out_size2//len(bin_sizes), num_classes, kernel_size=1)
        )

        self.daum = DAUM(in_channels, (9, 9))
        self.daum1 = AUM(in_channels, (5, 5))
        self.daum2 = AUM(in_channels, (5, 5))
        self.daum3 = AUM(in_channels, (5, 5))
        # self.daum4 = DAUM(512, (5, 5))

        self.smoother = GaussianSmoother()

        initialize_weights(self.ppm4, self.ppm3, self.ppm2,
                           self.daum1, self.daum2, self.daum3)
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        # input_size = (x.size()[2], x.size()[3])
        input_size = (x.size(2), x.size(3))
        fmaps = x
        fmaps_2 = self.smoother(fmaps)
        fmaps_4 = self.smoother(fmaps_2)
        fmaps_8 = self.smoother(fmaps_4)
        fmaps_16 = self.smoother(fmaps_8)
        x = self.initial(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        output2 = self.ppm2(x2)
        x3 = self.layer3(x2)
        output3 = self.ppm3(x3)
        x4 = self.layer4(x3)
        output4 = self.ppm4(x4)

        output = self.daum3(output4, F.interpolate(fmaps_16, size=x3.shape[-2:], mode="bilinear", align_corners=True))
        output = torch.cat((output.reshape(-1, 1), output3.reshape(-1, 1)), dim=1).max(dim=1)[0].reshape(*output.size())
        output = self.daum3(output, F.interpolate(fmaps_8, size=x2.shape[-2:], mode="bilinear", align_corners=True))
        output = torch.cat((output.reshape(-1, 1), output2.reshape(-1, 1)), dim=1).max(dim=1)[0].reshape(*output.size())
        output = self.daum3(output, F.interpolate(fmaps_4, size=x1.shape[-2:], mode="bilinear", align_corners=True))

        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        if self.training:
            return output, output2, output3, output4
        else:
            return output
        # return output

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(),
                     self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.ppm2.parameters(), self.ppm3.parameters(), self.ppm4.parameters(),
                     self.daum1.parameters(), self.daum2.parameters(), self.daum3.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()