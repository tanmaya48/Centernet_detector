import torch
import torch.nn as nn
import torchvision
import torch.functional as F

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)
    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                            diffY // 2, diffY - diffY//2))
        else:
            x = x1
        x = self.conv(x)
        return x


class centernet(nn.Module):
    def __init__(self, n_classes=1):
        super(centernet, self).__init__()
        # create backbone.
        kernel_size = 3  # Pooling kernel size
        padding = 1  # Padding for height and width
        stride = 1  # Stride
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        basemodel = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V2")
        basemodel = nn.Sequential(*list(basemodel.children())[:-1])
        self.base_model = basemodel
        num_ch = 1280
        self.up1 = up(num_ch, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)
        self.outr = nn.Conv2d(256, 2, 1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.base_model(x)
        # Add positional info        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        # outr = self.outr(x)
        outc = self.outc(x)
        outr = self.outr(x)
        # return outc, outr
        if self.training:
            return outc, outr
        #
        outc = torch.sigmoid(outc)
        out_pool = self.maxpool(outc)
        mask = (outc == out_pool)
        outc= outc*mask
        return outc,outr


