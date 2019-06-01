import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch.nn.functional as F
from data import Traindataset

EPOCH=100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv=torch.nn.Conv2d(3,12,kernel_size=3,padding=1)
        self.act=torch.nn.ReLU()
        self.conv2=torch.nn.Conv2d(12,12,kernel_size=3,padding=1)
        self.max=torch.nn.MaxPool2d(2,2)
        self.conv3=torch.nn.Conv2d(12,24,kernel_size=3,padding=1)
        self.fc1=torch.nn.Linear(32*32*24,100)
        self.fc2=torch.nn.Linear(100,1)
        self.sig=torch.nn.Sigmoid()
    def forward(self,x):
        convout=self.conv(x)
        convout2=self.conv2(F.relu(convout))
        convout2=self.max(F.relu(convout2))
        convout3=self.conv3(convout2)
        fcout1=self.fc1(F.relu(convout3).view(convout.shape[0],-1))
        output=self.fc2(F.relu(fcout1))
        output=self.sig(output)
        return output

class FC(nn.Module):
    def __init__(self):
        super(FC,self).__init__()
        self.fc1=torch.nn.Linear(3*64*64,1)

        self.sig=torch.nn.Sigmoid()
    def forward(self,x):
        fcout1=self.fc1(x.view(x.shape[0],-1))
        output=self.sig(fcout1)
        return output


model=CNN().to(device)

train=Traindataset()
trainloader=DataLoader(dataset=train,batch_size=5,shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

criterion=torch.nn.BCELoss()
losses=0
for _ in range(EPOCH):
    for (img,label) in trainloader:
      img=img.to(device)
      label=label.to(device)
      y_pred=model(img)
      loss=criterion(y_pred.reshape(y_pred.shape[0]),label)
      loss.backward()
      losses+=loss.item()
      model.zero_grad()
      optimizer.step()
    print(losses)
    losses=0


