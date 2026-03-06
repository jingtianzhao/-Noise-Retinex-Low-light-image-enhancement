import torch.nn as nn
from ..block import Conv
import torch 
import torch.nn.functional as F
class ASFF3(nn.Module):
    """ASFF3 module for YOLO AFPN head https://arxiv.org/abs/2306.15988"""
    # c1 输入通道数量的元组
    # c2 输出通道数量
    def __init__(self, c1, c2, scale_factor1):
        super(ASFF3, self).__init__()
        c1_l, c1_m, c1_h = c1[0], c1[1], c1[2]
        self.level = 0
        self.dim = c1_l, c1_m, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8
        # 对前两个通道进行上采样，并统一通道数量
        self.stride_level_1 = Conv(c1_m, self.inter_dim,1,1)
        self.stride_level_2 = ASFFDownsample(c1_h, self.inter_dim,scale_factor1)
        
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 3, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 3, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 3, 1)

        self.weights_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, c2, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1, x_level_2 = x[0], x[1], x[2]
        level_0_resized =  x_level_0
        level_1_resized = self.stride_level_1(x_level_1)
        level_2_resized = self.stride_level_2(x_level_2)
        # print(level_0_resized.shape,level_1_resized.shape,level_2_resized.shape)
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        # print(level_0_weight_v.shape,level_1_weight_v.shape,level_2_weight_v.shape)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        w = self.weights_levels(levels_weight_v)
        w = F.softmax(w, dim=1)

        fused_out_reduced = level_0_resized * w[:, :1] + level_1_resized * w[:, 1:2] + level_2_resized * w[:, 2:]
        return self.conv(fused_out_reduced)
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class ASFFDownsample(nn.Module):    
    def __init__(self, c1, c2, scale_factor=2):
        super(ASFFDownsample, self).__init__()
        self.quene=nn.Sequential()
        if scale_factor == 2:
            self.quene.add_module("down",Conv(c1, c2, 3, 2, 1))
        elif scale_factor == 4:
            self.quene.add_module("down1",Conv(c1, c2, 3, 2, 1))
            self.quene.add_module("down2",Conv(c2, c2, 3, 2, 1))
        elif scale_factor == 8:
            self.quene.add_module("down1",Conv(c1, c2, 3, 2, 1))
            self.quene.add_module("down2",Conv(c2, c2, 3, 2, 1))
            self.quene.add_module("down3",Conv(c2, c2, 3, 2, 1))
    def forward(self, x):
        return self.quene(x)
class ASFFUpsample(nn.Module):
    """Applies convolution followed by upsampling."""

    def __init__(self, c1, c2, scale_factor=2):
        super(ASFFUpsample, self).__init__()

        if scale_factor == 2:
            self.cv = nn.ConvTranspose2d(c1, c2, 2, 2, 0, bias=True)  # 如果下采样率为2，就用Stride为2的2×2卷积来实现2次下采样
        elif scale_factor == 4:
            self.cv = nn.ConvTranspose2d(c1, c2, 4, 4, 0, bias=True)  # 如果下采样率为4，就用Stride为4的4×4卷积来实现4次下采样
        elif scale_factor == 8:
            self.cv = nn.ConvTranspose2d(c1, c2, 8, 8, 0, bias=True)  # 如果下采样率为8，就用Stride为8的8×8卷积来实现8次下采样
        elif scale_factor == 1:
            self.cv = nn.Conv2d(c1, c2, 1, 1, 0, bias=True)  # 如果下采样率为1，就用1×1卷积来调整通道数
    def forward(self, x):
        return self.cv(x)
