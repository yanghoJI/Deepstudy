# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, RandomRotation, RandomHorizontalFlip
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import time


# Training settings
batch_size = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('runing device : {}'.format(device))

transferF = Compose([RandomRotation([.5,5]),RandomHorizontalFlip(), ToTensor()])
train_dataset = ImageFolder(root='../dataset/train/',transform=transferF)
test_dataset = ImageFolder(root='../dataset/val/',transform=transferF)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x)


model = Net()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
trloss, teloss, teacc = [], [], []

def plotdata(trl, tel, tea):
    xlist = range(len(trl))
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(xlist, trl, 'r-', label='train loss')
    plt.plot(xlist, tel, 'b-', label='validation loss')
    plt.ylabel('loss value')
    plt.title('loss graph')
    plt.legend(loc=1)

    ax2 = plt.subplot(2, 1, 2)
    plt.plot(xlist, tea, 'b-', label='validation acc')
    #plt.ylim(0, 100)
    plt.yticks(range(0,101,10))
    plt.grid(True)
    plt.ylabel('acc(%)')
    plt.title('acc graph')
    plt.legend(loc=1)

    plt.tight_layout()

    plt.savefig('TrainGraph.png', dpi=300)
    plt.close()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = Variable(data), Variable(target)
        #The Variable API has been deprecated: Variables are no longer necessary
        t1 = time.time()
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('\n##############################')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            trloss.append(loss.item())
            test()
            plotdata(trloss, teloss, teacc)
            print('runtime : {:.3f} sec'.format(time.time() - t1))




def test():
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
    model.train()



for epoch in range(1, 100):
    train(epoch)