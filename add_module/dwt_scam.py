import torch
from ultralytics.nn.modules.conv import Conv,Conv
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class dwt_scam(torch.nn.Module):
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
    def idwt(self,A,B,C,D):
        x1_1=(A-B-C+D)/4
        x2_1=(A-B+C-D)/4
        x1_2=(A+B-C-D)/4
        x2_2=(A+B+C+D)/4
        B,C,H,W=x1_1.size()
        x1=torch.zeros(B,C,H,W*2).to(A.device)
        x2=torch.zeros(B,C,H,W*2).to(A.device)
        x1[:,:,:,0::2]=x1_1
        x1[:,:,:,1::2]=x1_2
        x2[:,:,:,0::2]=x2_1
        x2[:,:,:,1::2]=x2_2
        x=torch.zeros(B,C,H*2,W*2).to(A.device)
        x[:,:,0::2,:]=x1*2
        x[:,:,1::2,:]=x2*2
        return x
    def ffm(self,A1,A2):
        # A2=A2.repeat(1,A1.size(1),1,1) #模板扩展到和深层特征一样的通道数
        mask1=torch.abs(A1)-torch.abs(A2)
        mask2=torch.abs(A2)-torch.abs(A1)
        mask1[mask1>0]=1
        mask2[mask2>0]=1
        res=mask1*A1+mask2*A2
        return res
    #初始化必须有两个输入参数，一个输入，一个输出
    def __init__(self,channel1,channel2):
        super().__init__()
        #通道注意力
        self.gap=torch.nn.AdaptiveAvgPool2d(1) #全局平均池化
        self.gmp=torch.nn.AdaptiveMaxPool2d(1) #全局最大池化
        self.conv1=Conv(2,1,1,1,0)
        # 第三分支
        self.conv=Conv(channel1,channel1,1,1,0)  
        #空间注意力
        self.convA=Conv(channel1,1,1,1,0)
        self.convB=Conv(channel1,1,1,1,0)
        self.convC=Conv(channel1,1,1,1,0)
        self.convD=Conv(channel1,1,1,1,0)
        self.m1=Conv_withoutBN(channel1,channel1,1,1)
        self.m2=Conv_withoutBN(channel1,channel1,1,1)
        self.m3=Conv_withoutBN(channel1,channel1,1,1)
        self.m4=Conv_withoutBN(channel1,channel1,1,1)
        self.input=None
        self.output=None
        self.A=[]
        self.B=[]
        self.C=[]
        self.D=[]
    def forward(self, x):
        # print(x.shape)
        self.input=x
        A,B,C,D=self.dwt(x)
        b,c,h,w=A.size()
        #原始分支通道
        v=self.conv(x)
        A1,B1,C1,D1=self.dwt(v)
        #
        v=A1.view(b,1,c,-1)
        #低频率使用通道注意力机制
        #使用softmax是在使用矩阵乘法前，归一化所有权重
        A_gap=self.gap(A).softmax(1).view(b,1,1,c)
        A_gmp=self.gmp(A).softmax(1).view(b,1,1,c)
        #使用矩阵乘法
        v_gap=torch.matmul(A_gap,v).view(b,1,h,w)
        v_gmp=torch.matmul(A_gmp,v).view(b,1,h,w)
        v_cat=torch.cat([v_gap,v_gmp],1)     # b 2 h w
        #这个还需要乘以空间注意力机制
        A_weight=self.conv1(v_cat).sigmoid() # b 1 h w
        #
        #1.在使用矩阵乘法时，需要提前在相对应的通道引入激活函数，矩阵乘法-softmax
        #高频使用空间注意力机制
        A_spatial=self.convA(A).view(b,1,-1,1).softmax(2)   #CBS b 1 h*w 1
        B_spatial=self.convB((B+A)/2).view(b,1,-1,1).softmax(2) #CBS
        C_spatial=self.convC((C+A)/2).view(b,1,-1,1).softmax(2) #CBS
        D_spatial=self.convD((D+A)/2).view(b,1,-1,1).softmax(2) #CBS
        #修改尺寸
        A1=A1.view(b,1,c,-1) #b 1 c h*w

        B1=B1.view(b,1,c,-1) #b 1 c h*w

        C1=C1.view(b,1,c,-1) #b 1 c h*w

        D1=D1.view(b,1,c,-1) #b 1 c h*w

        #
        A2=torch.matmul(A1,A_spatial).view(b,c,1,1)
        fin_a=self.m1(A2)*A_weight
        #B
        B2=torch.matmul(B1,B_spatial).view(b,c,1,1) #空间注意力
        b_gap=torch.matmul(A_gap,B1).view(b,1,h,w) #通道注意力
        fin_b=self.m2(B2)*(b_gap).sigmoid()
        #C
        C2=torch.matmul(C1,C_spatial).view(b,c,1,1)
        c_gap=torch.matmul(A_gap,C1).view(b,1,h,w)
        fin_c=self.m3(C2)*c_gap.sigmoid()
        #D
        D2=torch.matmul(D1,D_spatial).view(b,c,1,1)
        d_gap=torch.matmul(A_gap,D1).view(b,1,h,w)
        fin_d=self.m4(D2)*d_gap.sigmoid()
        self.A=A_spatial.view(b,1,h,w)
        self.B=B_spatial.view(b,1,h,w)
        self.C=C_spatial.view(b,1,h,w)
        self.D=D_spatial.view(b,1,h,w)
        res=1-self.idwt(fin_a,fin_b,fin_c,fin_d)
        self.output=res
        return res+x #引入残差
    
class Conv_withoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))