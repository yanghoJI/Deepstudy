# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

# device check
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('runing device : {}'.format(device))

# Training settings
batch_size = 64


train_set = dset.ImageFolder(root='../dataset/train',transform = transforms.Compose([
                                transforms.RandomResizedCrop(400),
                                transforms.ToTensor()
                               ]))

validation_set = dset.ImageFolder(root='../dataset/val',transform = transforms.Compose([
                                transforms.RandomResizedCrop(400),
                                transforms.ToTensor()]))

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                          batch_size=batch_size,
                                          shuffle=False)




class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=2)
        self.branch5x5_1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(256, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 3)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(24, 20, kernel_size=5)

        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(3668928, 10000)
        self.fc = nn.Linear(10000, 4)
#        self.fc = nn.Linear(1000, 4)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incept2(x)
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
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
    plt.yticks(range(0,101,10))
    plt.grid(True)
    plt.ylabel('acc(%)')
    plt.title('acc graph')
    plt.legend(loc=1)

    plt.tight_layout()

    plt.savefig('TrainGraph.png', dpi=300)
    plt.close()


model = Net()
model.apply(weight_init)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005)
trloss, teloss, teacc = [], [], []
best_acc = 0

if train_mode == True:
    # training model
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    trloss, teloss, teacc = [], [], []
    bestacc = 0
    for epoch in range(1, 30):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # data, target = Variable(data), Variable(target)
            # The Variable API has been deprecated: Variables are no longer necessary
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
                    torch.save(model, 'bestmodel.pb')
                    print('best model is updated')

                plotdata(trloss, teloss, teacc)

else:
    # test model

    model = torch.load('./bestmodel.pb')
    dirlist = os.listdir('../dataset/val/')
    teloss_, teacc_ = test(model, test_loader, device)
    print('test loss : {}\ttest acc : {:.2f} %'.format(teloss_, teacc_))
    while True:
        dirpath = os.path.join('../dataset/val/', r.sample(dirlist, 1)[0])
        filelist = os.listdir(dirpath)
        filepath = os.path.join(dirpath, r.sample(filelist, 1)[0])
        #img = cv2.imread('../dataset/val/0/45.png')
        '''
        with open(filepath, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = transferFte(img)
            img = img.numpy()
            img = np.expand_dims(img, axis=0)
            img = torch.tensor(img.astype('float32'))
            img = img.to(device)
            #img.to(device)
        out = model(img)
        print(out)
        label = str(out.argmax().item())
        sol = model.labeldict[label]
        '''
        img = cv2.imread(filepath)
        cv2.imshow('result', img)
        sol = model.predict(img, device)
        print('prediction : {}'.format(sol))



        cv2.waitKey(-1)