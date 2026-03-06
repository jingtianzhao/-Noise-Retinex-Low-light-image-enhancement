import torch.nn as nn
import torch
class cr2c(nn.module):
    def __init__(self,dim=1,channel1=1,channel2=1):
        super.__init__()
        self.w1=nn.Parameter(torch.ones(2,dtype=torch.float32),requires_grad=True)
        self.w2=nn.Parameter(torch.ones(channel1,dtype=torch.float32),requires_grad=True)
        self.w3=nn.Parameter(torch.ones(channel2,dtype=torch.float32),requires_grad=True)
        self.d=dim
        self.epsilon=0.0001
    def forward(self,x):
        B1,C1,H1,W1=x[0].size()
        B2,C2,H2,W2=x[1].size()
        w1=self.w1[:2]   #
        w2=self.w2[:C1]  #
        w3=self.w3[:C2]  #
        weight1=w1/(torch.sum(w1,dim=0)+self.epsilon)
        weight2=w2/(torch.sum(w2,dim=0)+self.epsilon)
        weight3=w3/(torch.sum(w3,dim=0)+self.epsilon)
        x1=(weight1[0]*weight2*x[0].view(B1,H1,W1,C1)).view(B1,C1,H1,W1)
        x2=(weight1[1]*weight3*x[1].view(B2,H2,W2,C2)).view(B2,C2,H2,W2)
        x_res=[x1,x2]
        res=torch.cat(x_res,self.d)
        return res
