# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:37:45 2018

@author: akash
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import matplotlib.pyplot as plt
import torch.utils.data as D
import torch.optim as optim
from torch.autograd import Variable
from modules import *
from torchvision import datasets,transforms
import argparse

def timeSince(since):
    now = time.time()
    s = now - since
    #m = math.floor(s / 60)
    #s -= m * 60
    return s

parser = argparse.ArgumentParser(description='MNIST Binarized weights')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',help='input batch size , default =64')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',help='input batch size for testing default=64')
parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed,default=1)')
parser.add_argument('--eps', type=float, default=1e-5, metavar='LR',help='learning rate,default=1e-5')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',help='for printing  training data is log interval')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    #train_loader
train_loader = D.DataLoader(datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),batch_size=args.batch_size, shuffle=True)
    
    #test_loaer
test_loader = D.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),batch_size=args.test_batch_size, shuffle=True)


################################################################
#MODEL
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            BinaryConv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16, momentum=args.momentum, eps=args.eps),
            nn.MaxPool2d(2),
            BinaryTanh())
        self.layer2 = nn.Sequential(
            BinaryConv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32, momentum=args.momentum, eps=args.eps),
            nn.MaxPool2d(2),
            BinaryTanh())
        self.fc = BinaryLinear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        
model = Model()
########################################################################
if args.cuda:
    #torch.cuda.set_device(3)
    model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        #if epoch%40==0:
            #optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
accur=[]
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    a=100.*correct / len(test_loader.dataset)
    accur.append(a)  
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

start = time.time()
time_graph=[]
e=[]
for epoch in range(1, args.epochs + 1):
    e.append(epoch)
    train(epoch)   
    seco=timeSince(start)
    time_graph.append(seco)
    test()

print(time_graph)
plt.title('Training for MNIST with epoch', fontsize=20)
plt.ylabel('time (s)')
plt.plot(e,time_graph,'bo')
plt.show()
plt.title('Accuracy With epoch', fontsize=20)
plt.plot(e,accur,'bo')
plt.show()


    
