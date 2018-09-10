# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import RandomCrop, Resize, Compose, ToTensor
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import time

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.imagesize = (224,224,3)
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, padding=2)     #out 10,224,224
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2)    #out 20,224,224
        self.mp1 = nn.MaxPool2d(2)                                   #out 20,112,112

        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, padding=1)  # out 20,112,112
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3, padding=1)  # out 20,112,112
        self.mp2 = nn.MaxPool2d(2)  # out 20,56,56

        self.fc1 = nn.Linear(20 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 4)
        self.aaa = 5

    def forward(self, x):
        in_size = x.size(0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.mp1(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = F.relu(self.mp2(self.conv4(x)))

        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)             # batch, 4

        return F.log_softmax(x, dim=1)      #dim = classes dimension


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('runing device : {}'.format(device))



#model = Net()
#model.to(device)
model = torch.load('./bestmodel33.pb')
model.eval()



def test():
    global bestacc
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        #data, target = Variable(data, volatile=True), Variable(target)
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    teloss.append(test_loss)
    teacc.append(100. * correct / len(test_loader.dataset))
    if teacc[-1] > bestacc:
        bestacc = teacc[-1]
        torch.save(Net, 'bestmodel.pb')
        print('best model is updated')
    model.train()


