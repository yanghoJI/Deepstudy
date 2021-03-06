from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import RandomCrop, Resize, Compose, ToTensor
from torchvision import datasets, transforms
import matplotlib as plt

# Training settings
batch_size = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('runing device : {}'.format(device))

trans = Compose([Resize([512,512]), ToTensor()])
train_dataset = datasets.ImageFolder(root='../dataset/train/', transform = trans)
val_dataset = datasets.ImageFolder(root='../dataset/val/', transform = trans)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride= 2, padding =2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128*30*30, 2048)
        self.fc2 = nn.Linear(2048, 4)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x))) #(64,128,128)
        x = F.relu(self.mp(self.conv2(x)))  #(128, 62, 62)
        x = F.relu(self.mp(self.conv3(x)))  #(128, 30, 30)
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x)

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
    #plt.xlim(0, 100)
    plt.yticks(range(0,101,10))
    plt.grid(True)
    plt.ylabel('acc(%)')
    plt.title('acc graph')
    plt.legend(loc=1)

    plt.tight_layout()

    plt.savefig('batchNorWithxavier.png', dpi=300)
    plt.close()

model = Net()
model.apply(weight_init)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005)

trloss, teloss, teacc = [], [], []

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            trloss.append(loss.item())
            acc =  test()
            plotdata(trloss, teloss, teacc)
    return acc

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    teloss.append(test_loss)
    teacc.append(100. * correct / len(val_loader.dataset))
    return correct


def save_checkpoint(state, filename='model_adam.pth.tar'):
    torch.save(state, filename)


acc_ = 0
for epoch in range(1, 201):
    accurancy = train(epoch)
    if acc_ < accurancy:
        acc_ = accurancy
        ep = epoch
save_checkpoint(ep)
print('model saved at %d epoch'%ep)
