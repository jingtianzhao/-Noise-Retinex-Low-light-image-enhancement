import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2 as cv
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.morphology import reconstruction
#双通道聚类
class kmscm(nn.Module):
    def __init__(self,in_channel,out_channel,k1,k2):
        super().__init__()
        self.k1=k1 #通道聚类数量
        self.k2=k2 #空间聚类数量
        self.kmeans1 = KMeans(n_clusters=self.k1, tol=0.1)
        self.kmeans2 = KMeans(n_clusters=self.k2, tol=0.1)
        self.conv1=nn.Conv2d(in_channel,k1,kernel_size=1,stride=1)
        #需不需要降采样
        #卷积层使用的是nn.conv2d
        self.down=nn.Conv2d(in_channel,in_channel//2,kernel_size=3,stride=2,padding=1)
        self.up=nn.Sequential(
            nn.Conv2d(in_channel//2,out_channel,kernel_size=1,stride=1),
            nn.ReLU()
        )
        self.att_map=[]   #可视化
    def forward(self,x:torch.tensor):
        B,C,H,W=x.size()
        xx=self.down(x)
        _,c,h,w=xx.size()
        r=random.randint(0,B-1)
        #随机选择一个batch 
        x1=xx[r,:,:,:].view(c,-1).permute(1,0).detach().cpu().numpy()
        x2=xx[r,:,:,:].view(c,-1).detach().cpu().numpy()
        #kmeans
        #通道聚类,背景聚类
        label_c=self.kmeans1.fit(np.nan_to_num(preprocessing.minmax_scale(x1.reshape(c,-1),axis=1), nan=0.0, posinf=0.0, neginf=0.0)).labels_
        v2_c=[]
        v22_c=[]
        {v2_c.append(np.where(label_c==kk)) for kk in range(self.k1)}
        for kk in range(len(v2_c)):
            if v2_c[kk][0].size > 0:  # 簇里有样本
                v22_c.append(np.max(x2[v2_c[kk]], 0))
            else:  # 空簇，给一个默认值
                v22_c.append(np.zeros(x2.shape[1]))
        #通道分别聚类
        fin_attention=torch.tensor(np.zeros((B,1,h,w)))
        for c in range(B):
            label_s=[]
            {label_s.append(self.kmeans2.fit(np.nan_to_num(preprocessing.minmax_scale(v22_c[kk].reshape(-1,1),axis=0), nan=0.0, posinf=0.0, neginf=0.0)).labels_) for kk in range(len(v22_c))}
            #采用形态学方法重构图像
            eroded=[]
            rec=[]
            fin=[]
            for i in range(self.k1):
                eroded.append(cv.dilate(label_s[i].reshape(h,w).astype(np.uint8), cv.getStructuringElement(cv.MORPH_RECT, (4, 4))))
                rec.append(reconstruction(eroded[i], label_s[i].reshape(h,w).astype(np.uint8),  method='erosion'))
                fin.append((label_s[i].reshape(h,w).astype(np.uint8)-rec[i]).astype(np.uint8))
            fin_attention[c,0,:,:]=torch.tensor(np.mean(fin,0)).cuda()
        # plt.subplot(1,3,1)
        # plt.imshow(np.mean(rec,0), cmap='gray')
        # plt.subplot(1,3,2)
        # plt.imshow(np.mean(label_s,0).reshape(h,w), cmap='gray')
        # plt.subplot(1,3,3)
        # plt.imshow(np.mean(fin,0), cmap='gray')
        # plt.show()
        #Visual_feature.visual_all_channel_map(label_s,h,w)
        #重新使用pytorch
        #unsqueeze(),向前增加一个维度
        self.att_map = fin_attention.detach().cpu().numpy()
        device=xx.device
        fin_attention=fin_attention.to(dtype=x.dtype,device=device)
        xx=xx.to(dtype=x.dtype,device=device)
        res=fin_attention*xx
        xx_attention=F.interpolate(self.up(res+xx),size=(x.shape[2], x.shape[3]),mode='bilinear',align_corners=False)+x
        #return xx_attention
        return x