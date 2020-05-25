import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import initialize_weights


class PPM(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer=nn.BatchNorm2d):
        super().__init__()
        out_channels = in_channels // len(bin_sizes)
        self.bin_sizes = bin_sizes
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        initialize_weights(self)

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        # prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        for bin_size, stage in zip(self.bin_sizes, self.stages):
            x = F.adaptive_avg_pool2d(features, output_size=(h//bin_size, w//bin_size))
            x =  stage(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            pyramids.append(x)
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class ASSP(nn.Module):
    def __init__(self, in_channels, output_stride, norm_layer=nn.BatchNorm2d):
        super().__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = self._make_branch(in_channels, 256, 1, dilation=dilations[0], norm_layer=norm_layer)
        self.aspp2 = self._make_branch(in_channels, 256, 3, dilation=dilations[1], norm_layer=norm_layer)
        self.aspp3 = self._make_branch(in_channels, 256, 3, dilation=dilations[2], norm_layer=norm_layer)
        self.aspp4 = self._make_branch(in_channels, 256, 3, dilation=dilations[3], norm_layer=norm_layer)

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(256 * 5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def _make_branch(in_channels, out_channles, kernel_size, dilation, norm_layer):
        padding = 0 if kernel_size == 1 else dilation
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
            norm_layer(out_channles),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x


class AUM(nn.Module):
    def __init__(self, high_channels, kernel_size, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.bn = norm_layer(high_channels)
        self.kernel_size = kernel_size
        self.bias = nn.Parameter(torch.zeros(kernel_size[0] * kernel_size[1]))

    def forward(self, low_fmaps, high_fmaps):
        high_fmaps = self.bn(high_fmaps)
        kernel_size = self.kernel_size
        bias = self.bias

        batch_size, low_channels, low_height, low_width = low_fmaps.size()
        _, high_channels, high_height, high_width = high_fmaps.size()
        kernel_height, kernel_width = kernel_size
        padding = kernel_height // 2, kernel_width // 2

        unfold_low_fmaps = F.unfold(low_fmaps, kernel_size, padding=padding)
        upsample_unfold_low_fmaps = F.upsample_nearest(
            unfold_low_fmaps.view(batch_size, low_channels * kernel_height * kernel_width,
                                  low_height, low_width), size=(high_height, high_width))
        reshape_upsample_unfold_low_fmaps = upsample_unfold_low_fmaps.reshape(batch_size, low_channels,
                                                                              kernel_height * kernel_width, -1)

        downsample_high_fmaps = F.adaptive_avg_pool2d(high_fmaps, (low_height, low_width))
        unfold_downsample_high_fmaps = F.unfold(downsample_high_fmaps, kernel_size, padding=padding)
        upsample_unfold_downsample_high_fmaps = F.upsample_nearest(
            unfold_downsample_high_fmaps.view(batch_size, high_channels * kernel_height * kernel_width,
                                              low_height, low_width), size=(high_height, high_width))

        reshape_upsample_unfold_downsample_high_fmaps = upsample_unfold_downsample_high_fmaps.reshape(batch_size,
                                                high_channels * kernel_height * kernel_width, -1)

        unfold_high_fmaps = F.unfold(high_fmaps, kernel_size, padding=padding)

        diff = unfold_high_fmaps - reshape_upsample_unfold_downsample_high_fmaps
        reshape_diff = diff.view(batch_size, high_channels, kernel_height * kernel_width, -1)
        pow_reshape_diff = reshape_diff.square()
        pow_reshape_diff = pow_reshape_diff.sum(dim=1) + bias.view(-1, 1)
        softmax_pow_reshape_diff = F.softmax(pow_reshape_diff, dim=1)

        output = torch.einsum('bkp,blkp->blp', softmax_pow_reshape_diff, reshape_upsample_unfold_low_fmaps)
        reshape_output = output.reshape(batch_size, low_channels, high_height, high_width)

        return reshape_output


class DAUM(nn.Module):
    def __init__(self, high_channels, kernel_size, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            kernel_size = kernel_size
        else:
            raise TypeError(f"Invalidate type of 'kernel_size': {type(kernel_size)}, it should be 'int' or 'tuple'")
        self.aum1 = AUM(high_channels, (kernel_size[0], 1), norm_layer)
        self.aum2 = AUM(high_channels, (1, kernel_size[1]), norm_layer)

    def forward(self, low_fmaps, high_fmaps):
        mid_fmaps = F.interpolate(high_fmaps, size=(high_fmaps.size(2), low_fmaps.size(3)), mode="bilinear", align_corners=True)
        x = self.aum1(low_fmaps, mid_fmaps)
        x = self.aum2(x, high_fmaps)
        return x


class GaussianSmoother(nn.Module):
    def __init__(self, in_channels=3, kernel_size=5, downsample=True):
        super().__init__()
        kernel = cv2.getGaussianKernel(kernel_size, -1)
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel.view(1, -1) * kernel.view(-1, 1)
        kernel = kernel.expand((in_channels, 1, kernel_size, kernel_size))
        self.in_channels = in_channels
        self.padding = (kernel_size//2, kernel_size-kernel_size//2-1)
        self.downsample = downsample
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        x = F.conv2d(x, self.kernel, padding=self.padding, groups=self.in_channels)
        if self.downsample:
            x = x[:, :, ::2, ::2]
        return x

