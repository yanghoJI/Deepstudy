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
import cv2
import random as r
import os


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.imagesize = (224,224,3)
        self.labeldict = {'0' : '오징어짬뽕', '1' : '비빔면', '2' : '짜왕', '3' : '무파마'}

        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, padding=2)     #out 10,224,224
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2)    #out 20,224,224
        self.mp1 = nn.MaxPool2d(2)                                   #out 20,112,112

        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, padding=1)  # out 20,112,112
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3, padding=1)  # out 20,112,112
        self.mp2 = nn.MaxPool2d(2)  # out 20,56,56

        self.fc1 = nn.Linear(20 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 4)


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

    def predict(self, imgarr, device):
        inimage = cv2.resize(imgarr, (self.imagesize[0], self.imagesize[1]))
        inimage = np.expand_dims(np.transpose(inimage, (2, 0, 1)), axis=0) / 255
        inimage = torch.tensor(inimage.astype('float32'))
        inimage = inimage.to(device)
        netout = self.forward(torch.tensor(inimage))
        label = str(netout.argmax().item())
        return self.labeldict[label]



def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


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

    plt.savefig('TrainGraphWithxavier#2.png', dpi=300)
    plt.close()


def test(model_, loader, device):
    model_.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        #data, target = Variable(data, volatile=True), Variable(target)
        data = data.to(device)
        target = target.to(device)
        output = model_(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))
    test_acc = 100. * correct / len(loader.dataset)
    model_.train()
    return (test_loss, test_acc)




# device check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('runing device : {}'.format(device))


# Training settings
batch_size = 64
bestacc = 0
train_mode = True


# noodle Dataset build
transferF = Compose([Resize([256, 256]), RandomCrop([224, 224]), ToTensor()])
transferFte = Compose([Resize([224, 224]), ToTensor()])
train_dataset = ImageFolder(root='../dataset/train/',transform=transferF)
test_dataset = ImageFolder(root='../dataset/val/',transform=transferFte)
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# model build
model = Net()
model.apply(weight_init)
model.to(device)

if train_mode == True:
    # training model
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    trloss, teloss, teacc = [], [], []
    bestacc = 0
    for epoch in range(1, 50):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # data, target = Variable(data), Variable(target)
            # The Variable API has been deprecated: Variables are no longer necessary
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
                teloss_, teacc_ = test(model, test_loader, device)
                teloss.append(teloss_)
                teacc.append(teacc_)
                if teacc_ > bestacc:
                    bestacc = teacc_
                    torch.save(model, 'bestmodel33.pb')
                    print('best model is updated')

                plotdata(trloss, teloss, teacc)
                print('runtime : {:.3f} sec'.format(time.time() - t1))

else:
    # test model

    model = torch.load('./bestmodel33.pb')
    dirlist = os.listdir('../dataset/val/')
    dirpath = os.path.join('../dataset/val/', r.sample(dirlist, 1)[0])
    filelist = os.listdir(dirpath)
    filepath = os.path.join(dirpath, r.sample(filelist, 1)[0])
    #img = cv2.imread('../dataset/val/0/45.png')
    img = cv2.imread(filepath)
    sol = model.predict(img, device)
    print('prediction : {}'.format(sol))
    cv2.imshow('result', img)
    cv2.waitKey(-1)
    #teloss_, teacc_ = test(model, test_loader, device)
    #print('test loss : {}\ttest acc : {:.2f} %'.format(teloss_, teacc_))


