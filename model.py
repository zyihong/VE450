import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from data import Traindataset
from tqdm import tqdm
EPOCH = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
LOAD_FROM_CHECKPOINT=True



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.act = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(12, 12, kernel_size=3, padding=1)
        # self.max=torch.nn.MaxPool2d(2,2)
        # self.conv3=torch.nn.Conv2d(12,24,kernel_size=3,padding=1)
        self.fc1 = torch.nn.Linear(64 * 64 * 12, 100)
        self.fc2 = torch.nn.Linear(100, 5)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        convout = self.conv(x)
        convout2 = self.conv2(F.relu(convout))
        # convout2=self.max(F.relu(convout2))
        # convout3=self.conv3(convout2)
        fcout1 = self.fc1(self.act(convout2).view(convout2.shape[0], -1))
        output = self.fc2(self.act(fcout1))
        output = self.sig(output)
        return output


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        # self.conv = torch.nn.Conv2d(3, 12, kernel_size=3, padding=1)
        # self.max = torch.nn.MaxPool2d(2, 2)
        # self.fc1 = torch.nn.Linear(12 * 32 * 32, 10)
        # self.fc2 = torch.nn.Linear(10, 1)
        self.fc1 = torch.nn.Linear(3 * 64 * 64, 1)
        self.sig = torch.nn.Sigmoid()

        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        fcout1 = self.fc1(x.view(x.shape[0], -1))
        output = self.sig(fcout1)
        # convout = self.conv(x)
        # max = self.max(F.relu(convout))
        # fcout1 = self.fc1(max.view(max.shape[0], -1))
        # r = F.relu(fcout1)
        # fc2 = self.fc2(r)
        #
        # output = self.sig(fc2)
        return output



model=CNN().to(device)
if LOAD_FROM_CHECKPOINT:
    model.load_state_dict(torch.load("./model.ckpt"))



train=Traindataset("./resize")
test=Traindataset("./resizetest")
trainloader=DataLoader(dataset=train, batch_size=10, shuffle=True)
testloader = DataLoader(dataset=test, batch_size=1, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0000008)#,weight_decay=0)

criterion=torch.nn.CrossEntropyLoss()
losses=0
for _ in range(EPOCH):
    for (img,label) in trainloader:
      img = img.to(device)
      label = label.to(device)
      optimizer.zero_grad()
      y_pred = model(img)
      #print('label size: ', label.size())
      #print('ypred size: ', y_pred.size())
      # yy = y_pred.reshape(y_pred.shape[0])
      #print('yypred size: ', yy.size())
      loss = criterion(y_pred, label)

      loss.backward()
      losses += loss.item()
      # model.zero_grad()
      optimizer.step()


    total = 0
    correct = 0
    with torch.no_grad():
        for (img, label) in testloader:

          img = img.to(device)
          label = label.to(device)
          y_pred = model(img)
          # yy = y_pred.reshape(y_pred.shape[0])
          predicted = torch.argmax(y_pred,dim=1)
          total += label.size(0)
          correct += (predicted == label).sum().item()


    print('loss: ', losses, 'acc: ', (100 * correct / total))

    losses = 0

torch.save(model.state_dict(),"./model.ckpt")
