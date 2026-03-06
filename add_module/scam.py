import torch
from ultralytics.nn.modules.conv import Conv
class SCAM(torch.nn.Module):
    def __init__(self,channel):
        super.__init__()
        self.avg_pool=torch.nn.AdaptiveAvgPool2d(1) #全局平均池化
        self.max_pool=torch.nn.AdaptiveMaxPool2d(1) #全局最大池化
        self.conv=Conv(channel,1,1,1,0)
        self.conv1=Conv(2,1,1,1,0)
        self.softmax=torch.nn.Softmax()
    def forward(self,x):
        avg_out=self.conv1(self.softmax(torch.cat([self.avg_pool(x),self.max_pool(x)],1)))
        #最大池化
        max_out=self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out=avg_out+max_out
        return x*self.sigmoid(out)