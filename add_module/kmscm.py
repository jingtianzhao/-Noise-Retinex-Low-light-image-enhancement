import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2 as cv
from sklearn import preprocessing
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
from skfuzzy import cmeans
from skimage.morphology import reconstruction
#这个方法实现的conv实现126转8通道，然后每一类使用kmeans聚类，形态学重构后求均值获得空间注意力掩膜，然后使用sigmoid归一化乘以原始特征图，引入残差最后使用上采样恢复原始尺寸
class kmscm(nn.Module):
    _shared=None
    _num=0
    @staticmethod
    def _consume():
        if kmscm._shared==None:
            return None
        x=kmscm._shared
        kmscm._shared=None
        return x
    @staticmethod
    def _put(array):
        if kmscm._shared !=None:
            kmscm._shared=array
    def __init__(self,in_channel,out_channel,k1,k2,size):
        super().__init__()
        self.k1=k1 #通道聚类数量
        self.k2=k2 #空间聚类数量
        self.conv=nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.label=[]
        self.sigmoid=nn.Sigmoid()
        # k-means
        self.cluster_center=nn.Parameter(torch.rand(k2,k1,dtype=torch.float32),requires_grad=True)
        # fuzzy c-means
        # self.cluster_center=nn.Parameter(torch.rand(size1,k2,dtype=torch.float32),requires_grad=True)
        self.kmeans1 = KMeans(n_clusters=self.k1, tol=0.1)
        self.structure=size
        #需不需要降采样
        #卷积层使用的是nn.conv2d
        #可视化
        self.input=[]
        self.eroded=[]
        self.temp=[]
        self.dilate=[]
        self.rec=[]
        self.att_map=[]
        self.output=[]
    def forward(self,x:torch.tensor):
        self.input=x.detach().cpu().numpy()
        x=self.conv(x)
        B,C,H,W=x.size()
        #随机选择一个batch 
        #kmeans
        #通道聚类,背景聚类
        if kmscm._shared == None and kmscm._num!=-1:
            r=random.randint(0,B-1)
            xx=x[r,:,:,:].reshape(C,-1).detach().cpu().numpy()
            label_c=self.kmeans1.fit(np.nan_to_num(preprocessing.minmax_scale(xx,axis=1), nan=0.0, posinf=0.0, neginf=0.0)).labels_
            v2_c=[]
            #存储类别
            {v2_c.append(np.where(label_c==kk)) for kk in range(self.k1)}
            #通道独立聚类
            fin_attention = np.zeros((B, 1, H, W))
            for b in range(B):
                # v22_c 已在 GPU，上 CPU 仅在 fit 这一步
                feat = np.zeros((H*W,self.k1))
                #new 
                for kk in range(self.k1):
                    # 获取第 kk 个簇的像素值，展平成 (C, H*W)
                    result = x[b, v2_c[kk], :, :].reshape(-1, H*W).detach().cpu().numpy()
                    # 计算每个簇的均值，沿着每个像素位置（axis=1）计算均值
                    feat[:, kk] = np.mean(result, axis=0)  # 计算每个簇的均值并填充进 feat
                    #feat=preprocessing.minmax_scale(feat,axis=1).reshape(H*W,-1) #reshape有问题
                feat=preprocessing.minmax_scale(feat,axis=1)
                #use k-means
                kmeans2 = KMeans(n_clusters=self.k2,init=self.cluster_center.detach().cpu().numpy(), tol=0.1)
                label_s = kmeans2.fit(np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)).labels_
                self.cluster_center.data = torch.from_numpy(kmeans2.cluster_centers_).type_as(x).to(x.device).detach()
                # 需要区分前景和背景
                pixel_num=[]
                {pixel_num.append(np.sum(label_s==i)) for _,i in enumerate(np.unique(label_s))}
                sort_num=np.argsort(pixel_num)
                if sort_num.size>1:
                    min_val=sort_num[1]
                    if min_val !=0:
                        label_s[label_s==min_val]=-1
                        label_s[label_s==0]=min_val
                        label_s[label_s==-1]=0
                    mes=[]
                    tp1=np.mean(np.where(label_s==0,label_s,0))
                    for k in range(self.k2-1):
                        tp2=np.where(label_s==k+1,label_s,0)
                        mes.append(np.mean(tp1 - tp2) ** 2)
                    ind=np.argsort(mes)
                    label_s[label_s==(ind[0]+1)]=0
                    label_s[label_s==(ind[1]+1)]=1
                    label_s[label_s==(ind[2]+1)]=1
                
                #图像反转
                label_s=np.abs(label_s-1)
                label_s=label_s.astype(np.float16)
                new=np.where(label_s==1,label_s,0.25)
                lab2d = new.reshape(H, W).astype(np.uint8)
                # print(np.sum(new[new==1]))
                self.label=lab2d
                fin_map=self.label
                #赋值
                fin_attention[b,0] = np.abs(fin_map)
            self.att_map=fin_attention
            kmscm._put(fin_attention)
            tp3=torch.from_numpy(fin_attention).type_as(x).to(x.device)
        elif kmscm._num==-1:
            tp3=torch.zeros(x.size()).to(x.device)
            kmscm._num=0
        else:
            fin_attention=kmscm._consume()
            tp3=torch.from_numpy(fin_attention).type_as(x).to(x.device)
            tp3=F.interpolate(tp3,x.size())
        
        res=self.sigmoid(tp3)*x
        xx_attention=x+res
        self.output=xx_attention.detach().cpu().numpy()
        return xx_attention
        #return x