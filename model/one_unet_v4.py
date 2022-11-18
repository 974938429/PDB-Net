'''v4:第三阶段与原图进行堆叠，并且与前两阶段特征层也进行堆叠'''



import sys
sys.path.append('../')
sys.path.append('./')
from model.unet_parts_paralleling_new import *


class UNetStage(nn.Module):
    def __init__(self, n_channels=3, bilinear=False):
        super(UNetStage, self).__init__()
        factor = 2 if bilinear else 1
        _factor = 1 if bilinear else 2
        # print('factor is : ',_factor)
        self.n_channels = n_channels
        self.n_classes = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        stage_x1 = self.up1(x5, x4)
        stage_x2 = self.up2(stage_x1, x3)
        stage_x3 = self.up3(stage_x2, x2)
        stage_x4 = self.up4(stage_x3, x1)
        logits = self.outc(stage_x4)
        return [logits, stage_x1, stage_x2, stage_x3]

class Cat_Unet(nn.Module):
    def __init__(self, n_channels=5, bilinear=False):
        super(Cat_Unet, self).__init__()
        factor = 2 if bilinear else 1
        _factor = 1 if bilinear else 2

        print('factor is : ', _factor)
        self.n_channels = n_channels
        self.n_classes = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.maxpool = MaxPool()
        self.down1 = Down_no_pool(128, 128)
        self.down2 = Down_no_pool(256, 256)
        self.down3 = Down_no_pool(512, 512)

        self.down4 = Down_no_pool(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)

        self.fuse2 = FuseStageOut(in_channels=128 * _factor + 64, out_channles=128)
        self.fuse3 = FuseStageOut(in_channels=256 * _factor + 128, out_channles=256)
        self.fuse4 = FuseStageOut(in_channels=512 * _factor + 256, out_channles=512)

    def forward(self, x_1, x_2, x_3, stage3, stage2, stage1, _stage3, _stage2, _stage1):

        x = torch.cat([x_1, x_2, x_3], dim=1)

        x1 = self.inc(x)

        # fuse stage
        x2 = self.maxpool(x1)
        x2 = self.fuse2(x2, stage1, _stage1)
        x2 = self.down1(x2)

        x3 = self.maxpool(x2)
        x3 = self.fuse3(x3, stage2, _stage2)
        x3 = self.down2(x3)

        x4 = self.maxpool(x3)
        x4 = self.fuse4(x4, stage3, _stage3)
        x4 = self.down3(x4)

        x5 = self.maxpool(x4)
        x5 = self.down4(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        final = self.outc(x)
        return [final,x]