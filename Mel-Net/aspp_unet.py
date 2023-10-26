from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

import torch.nn.init as init
'''
ASPP:空洞空间金字塔，解决了使用单一空洞卷积带来的信息连续性损失和远距离信息相关性差的问题
'''
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class bian(nn.Module):
    def __init__(self):
        super(bian, self).__init__()
        self.lala = nn.Sequential(
            nn.MaxPool2d((4, 1))
        )

    def forward(self, x):
        x = self.lala(x)
        return x


from torch import nn
import torch
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channel):
        depth = in_channel
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear',align_corners=True)

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        cat = torch.cat([image_features, atrous_block1, atrous_block6,
                         atrous_block12, atrous_block18], dim=1)
        net = self.conv_1x1_output(cat)
        return net


# aspp = ASPP(256)
# out = torch.rand(2, 256, 13, 13)
# print(aspp(out).shape)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            # nn.Conv2d(F_g, F_int, kernel_size=2, padding=0),
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)

        # g11 = F.pad(g1, (0, 1, 1, 0),value=0)
        # print('g11======',g11.shape)
        x1 = self.W_x(x)

        # concat + relu
        psi = self.relu(g1 + x1)
        # #
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi
# class uup(nn.Module):
#     """ up path
#         conv_transpose => double_conv
#     """
#
#     def __init__(self, in_ch, out_ch, Transpose=False):
#         super(uup, self).__init__()
#
#         #  would be a nice idea if the upsampling could be learned too,
#         #  but my machine do not have enough memory to handle all those weights
#         if Transpose:
#             self.uup = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
#         else:
#             # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.uup = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#                                      nn.Conv2d(in_ch, in_ch // 2, kernel_size=2, padding=0),
#                                      nn.ReLU(inplace=True))
#         self.conv = double_conv(in_ch, out_ch)
#         self.uup.apply(self.init_weights)
#         self.xiao = bian()
#
#     def forward(self, x):
#         """
#             conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
#         """
#
#         x = self.uup(x)
#         x = F.pad(x, (1, 0, 1, 0), value=0)
#         # x2 = self.xiao(x2)
#         # x2 = self.xiao(x2)
#
#         # diffY = x2.size()[2] - x1.size()[2]
#         # diffX = x2.size()[3] - x1.size()[3]
#
#         print('xuup=',x.shape)
#
#         # x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
#         # print('xi变化后==')
#         # print(x1.shape)
#
#         # x = torch.cat([x2, x1], dim=1)
#
#         # x = self.conv(x)
#         return x
#
#     @staticmethod
#     def init_weights(m):
#         if type(m) == nn.Conv2d:
#             init.kaiming_normal_(m.weight)
#             init.constant_(m.bias, 0)


class Unet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(Unet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=32)
        self.Conv2 = conv_block(ch_in=32, ch_out=64)
        self.xiao = bian()
        self.Conv3 = conv_block(ch_in=64, ch_out=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=256)
        self.Conv5 = conv_block(ch_in=256, ch_out=512)
        self.Conv6 = conv_block(ch_in=512, ch_out=1024)

        self.SPP1 =ASPP(1024)
        self.SPP2=ASPP(512)

        self.Up6 = up_conv(ch_in=1024, ch_out=512)
        self.Att6 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv6 = conv_block(ch_in=1024, ch_out=512)

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Att5 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Att4 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Att3 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Att3 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)


        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)


        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)


        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)


        x5 = self.Maxpool(x4)

        x5 = self.Conv5(x5)

        # x5=self.SPP2(x5)

        x6 = self.Maxpool(x5)

        x6 = self.Conv6(x6)


        # decoding + concat path
        d6=self.SPP1(x6)

        d6 = self.Up6(d6)

        # d6 = F.pad(d6, (0, 0, 1, 0), value=0)
        # print('d55======', d6.shape)

        x5 = self.Att6(g=d6, x=x5)
        d6 = torch.cat((x5, d6), dim=1)
        d6 = self.Up_conv6(d6)


        d5 = self.Up5(d6)
        # d5 = F.pad(d5, (1, 0, 0, 0), value=0)

        # d5 = F.pad(d5, (0, 0, 1, 0), value=0)#在200*256时候需要d5这个pad，参数是0010

        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)


        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

#d3和d2不进行拼接，直接upp
        d3 = self.Up3(d4)
        # x2 = self.Att3(g=d3, x=x2)
        # d3 = torch.cat((x2, d3), dim=1)
        # d3 = self.Up_conv3(d3)




        d2 = self.Up2(d3)
        # x1 = self.Att2(g=d2, x=x1)
        # d2 = torch.cat((x1, d2), dim=1)
        # d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        # d1 = self.sigmoid(d1)




        return d1


if __name__ == "__main__":
    data = torch.rand(4, 1, 200, 256).cpu()
    model = Unet().cpu()
    model.train()
    y = model(data)
    print(y.shape)
