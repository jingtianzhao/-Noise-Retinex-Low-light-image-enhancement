import torch
from ultralytics.nn.modules.conv import Conv,Conv
import torch.nn as nn
import kornia
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class moratt(torch.nn.Module):
    #初始化必须有两个输入参数，一个输入，一个输出
    def dwt(self,x):
        x1=x[:,:,0::2,:]/2
        x2=x[:,:,1::2,:]/2
        x1_1=x1[:,:,:,0::2] #偶数行-偶数列
        x2_1=x2[:,:,:,0::2] #奇数行-偶数列
        x1_2=x1[:,:,:,1::2] #偶数行-奇数列
        x2_2=x2[:,:,:,1::2] #奇数行-奇数列
        # print(x1_1.size(),x2_1.size(),x1_2.size(),x2_2.size())
        A=x1_1+x2_1+x1_2+x2_2
        B=-x1_1-x2_1+x1_2+x2_2 #行差分
        C=-x1_1+x2_1-x1_2+x2_2 #列差分
        D=x1_1-x2_1-x1_2+x2_2  #对角线差分
        return A,B,C,D
    def morp(self,tensor,size):
        open_res=kornia.morphology.opening(tensor,size)
        return tensor-open_res
    def __init__(self,channel1,channel2,s1,s2,s3):
        super().__init__()
        #通道注意力
        self.gap=torch.nn.AdaptiveAvgPool2d(1) #全局平均池化
        self.gmp=torch.nn.AdaptiveMaxPool2d(1) #全局最大池化
        self.conv1=Conv(2,1,1,1,0)
        # 第三分支
        self.conv=Conv(channel1,channel1,1,1,0)  
        #空间注意力
        self.convA=Conv(channel1,1,1,1,0)
        self.m1=Conv_withoutBN(channel1,channel1,1,1)
        self.size1=torch.ones(s1,1)
        self.size2=torch.ones(s2,s2)
        self.size3=torch.ones(1,s3)
        # self.input=None
        # self.output=None
        # self.A=[]
        # self.B=[]
        # self.C=[]
        # self.D=[]
    def forward(self, x):
        # print(x.shape)
        # self.input=x
        device=x.device
        A,B,C,D=self.dwt(x)
        b,c,h,w=x.size()
        #原始分支通道
        v=self.conv(x)
        v=v.view(b,1,c,-1) #b 1 c h*w
        #通道注意力机制
        #使用softmax是在使用矩阵乘法前，归一化所有权重
        x_gap=self.gap(A).softmax(1).view(b,1,1,c)
        x_gmp=self.gmp(A).softmax(1).view(b,1,1,c)
        #使用矩阵乘法
        v_gap=torch.matmul(x_gap,v).view(b,1,h,w)
        v_gmp=torch.matmul(x_gmp,v).view(b,1,h,w)
        ## 修改内容
        # A=A.view(b,1,c,-1)
        # h1m
        # v_gap=torch.matmul(x_gap,v).view(b,1,h,w)
        # v_gmp=torch.matmul(x_gmp,v).view(b,1,h,w)
        v_cat=torch.cat([v_gap,v_gmp],1)     # b 2 h w
        #这个还需要乘以空间注意力机制
        A_weight=self.conv1(v_cat).sigmoid() # b 1 h w
        #1.在使用矩阵乘法时，需要提前在相对应的通道引入激活函数，矩阵乘法-softmax
        #高频使用空间注意力机制
        # x_spatial=self.convA(x).view(b,1,-1,1).softmax(2)
        x_spatial=self.convA(x)
        # self.A=x_spatial
        x_spatial=self.morp(x_spatial,self.size1.to(device))
        # self.B=x_spatial
        x_spatial=self.morp(x_spatial,self.size2.to(device))
        # self.C=x_spatial
        x_spatial=self.morp(x_spatial,self.size3.to(device)).view(b,1,-1,1).softmax(2)
        # self.D=x_spatial.view(b,1,h,w)
        #通道注意力进行
        #修改尺寸
        #

        A2=torch.matmul(v,x_spatial).view(b,c,1,1)
        res=self.m1(A2)*A_weight
        # self.output=res
        return res+x #
    
class Conv_withoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))