# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

import sys
import cv2

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
        inimage = cv2.cvtColor(imgarr, cv2.COLOR_BGR2RGB)
        inimage = cv2.resize(inimage, (self.imagesize[0], self.imagesize[1]))
        inimage = np.transpose(inimage, (2, 0, 1)) / 255
        inimage = np.expand_dims(inimage, axis=0)
        inimage = torch.tensor(inimage.astype('float32'))
        inimage = inimage.to(device)
        netout = self.forward(torch.tensor(inimage))
        label = str(netout.argmax().item())
        return self.labeldict[label]



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('runing device : {}'.format(device))


model = Net()
model.to(device)
model = torch.load('./bestmodel#243.pb')
model.eval()

# set camera
video_capture = cv2.VideoCapture(0)

cropxmin, cropxmax, cropymin, cropymax = 100, 580, 0, 480

while True:
    # Get frame
    ret, frame = video_capture.read()
    rec_c = frame[cropymin:cropymax, cropxmin:cropxmax, :]
    cv2.imwrite('img1.png', rec_c)
    img = cv2.imread('./img1.png')
    #rec_c = cv2.cvtColor(rec_c, cv2.COLOR)
    print(rec_c.shape)
    sol = model.predict(img, device)
    print('prediction : {}'.format(sol))

    cv2.imshow('result', img)

    key = cv2.waitKey(50)
    if key == ord('q'):
        break

