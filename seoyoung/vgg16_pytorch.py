from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import time
import os
import copy

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")


# device check
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('runing device : {}'.format(device))


# Training settings
batch_size = 64


train_set = dset.ImageFolder(root='../dataset/train',transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                               ]))

validation_set = dset.ImageFolder(root='../dataset/val',transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()]))


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                          batch_size=batch_size,
                                          shuffle=False)


# Load the pretrained model from pytorch
vgg16 = models.vgg16(pretrained=True)
vgg16.load_state_dict(torch.load("/home/artint/.torch/models/vgg16-397923af.pth"))
print(vgg16.classifier[6].out_features) # 1000




# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False


# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 4)]) # Add our layer with 4 outputs
features.extend([nn.LogSoftmax(1)])
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier



# If you want to train the model for more than 2 epochs, set this to True after the first run
resume_training = False

if resume_training:
    print("Loading pretrained model..")
    vgg16.load_state_dict(torch.load('/home/artint/.torch/models/vgg16-397923af.pth'))
    print("Loaded!")

vgg16.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

def train(vgg, optimizer, num_epochs=10):
    vgg.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = vgg(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                num_epochs, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(vgg):
    vgg.eval()
    test_loss = 0
    correct = 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = vgg(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(validation_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))
    return correct



'''
def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0

    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0


    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        vgg.train(True)

        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, batch_size / 2), end='', flush=True)

            # Use half training dataset
            if i >= batch_size / 2:
                break

            inputs, labels = data[0], data[1]

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        print()
        # * 2 as we only used half of the dataset
        avg_loss = loss_train * 2 / len(train_set)
        avg_acc = acc_train * 2 / len(train_set)

        vgg.train(False)
        vgg.eval()

        for i, data in enumerate(validation_loader):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, batch_size), end='', flush=True)

            inputs, labels = data[0], data[1]

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss_val += loss.item()
            acc_val += torch.sum(preds == labels)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss_val = loss_val / len(validation_set)
        avg_acc_val = acc_val / len(validation_set)

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    vgg.load_state_dict(best_model_wts)
    return vgg
'''
for epoch in range(5):
    train(vgg16, optimizer_ft)
test(vgg16)