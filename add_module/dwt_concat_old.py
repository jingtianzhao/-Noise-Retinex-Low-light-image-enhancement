from typing import List
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
# from ..conv import Concat
class dwt_concat(nn.Module):
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
    #channel1=x2的通道数、channel2=x2的通道数
    def __init__(self,dimension=1,channel1=128,channel2=256,mode=0):
        super().__init__()
        self.mode=mode
        if mode ==0:
            self.conv=nn.Conv2d(channel1,channel2,kernel_size=1,stride=1,padding=0,bias=False)
        else:
            self.conv=nn.Conv2d(channel2,channel1,kernel_size=1,stride=1,padding=0,bias=False)
        self.d=dimension
    #类型不一致报错，halftensor和float32tensor类型不一致报错，主要原因是AMP训练完后，测试输入为float32
    def forward(self,x: nn.ModuleList):
        # x_1表示的是深层特征，包含着更多的语义信息；x_2表示的是低层特征，包含着更多的高频信息
        x_1, x_2 = x
        # print(x_1.shape,x_2.shape)
        ##按照最大通道原则进行特征图对齐
        if self.mode==0:
            new_x_2 = self.conv(x_2)  # 转换为1通道，作为模板
            new_x_1 = x_1
            A1,B1,C1,D1=self.dwt(new_x_1)   #进行下采样
            A2,B2,C2,D2=self.dwt(new_x_2) #分解成4个模板
            # print(A1.shape,A2.shape)
            fin_A=(A1+A2)/2 #按照模板+结合深层特征的每个通道进行分别融合
            fin_B=self.ffm(B1,B2) #按照模板+结合深层特征的每个通道进行分别融合
            fin_C=self.ffm(C1,C2) #按照模板+结合深层特征的每个通道进行分别融合
            fin_D=self.ffm(D1,D2) #按照模板+结合深层特征的每个通道进行分别融合
            x_1_res=self.idwt(fin_A,fin_B,fin_C,fin_D)
            res=torch.cat([x_1_res,x_2],self.d)
        else:
            new_x_1 = self.conv(x_1)  # 转换为1通道，作为模板
            new_x_2 = x_2
            A1,B1,C1,D1=self.dwt(new_x_1)   #进行下采样
            A2,B2,C2,D2=self.dwt(new_x_2) #分解成4个模板
            # print(A1.shape,A2.shape)
            fin_A=(A1+A2)/2 #按照模板+结合深层特征的每个通道进行分别融合
            fin_B=self.ffm(B1,B2) #按照模板+结合深层特征的每个通道进行分别融合
            fin_C=self.ffm(C1,C2) #按照模板+结合深层特征的每个通道进行分别融合
            fin_D=self.ffm(D1,D2) #按照模板+结合深层特征的每个通道进行分别融合
            x_2_res=self.idwt(fin_A,fin_B,fin_C,fin_D)
            res=torch.cat([x_1,x_2_res],self.d)
        # print(res.shape)
        return res
