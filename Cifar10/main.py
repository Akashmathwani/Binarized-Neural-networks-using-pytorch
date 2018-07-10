# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:32:22 2018

@author: akash
"""

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

parser = argparse.ArgumentParser(description='CIFAR Binarized weights')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',help='input batch size , default =64')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',help='input batch size for testing default=64')
parser.add_argument('--epochs', type=int, default=400, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed,default=1)')
parser.add_argument('--eps', type=float, default=1e-5, metavar='LR',help='learning rate,default=1e-5')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',help='for printing  training data is log interval')
parser.add_argument('--best_acc', type=float, default=0, metavar='N',help='Record of best accuracy')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    #train_loader
train_loader = D.DataLoader(datasets.CIFAR10('./data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ])),batch_size=args.batch_size, shuffle=True)
    
    #test_loaer
test_loader = D.DataLoader(datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ])),batch_size=args.test_batch_size, shuffle=True)


################################################################

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 128 , kernel_size=3, padding=1,),
            nn.BatchNorm2d(128,eps=1e-4, momentum=0.1),
            nn.Hardtanh(),

            BinConv2d(128 , 128 , kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128,eps=1e-4, momentum=0.1),
            nn.Hardtanh(),


            BinConv2d(128 , 256 , kernel_size=3, padding=1),
            nn.BatchNorm2d(256,eps=1e-4, momentum=0.1),
            nn.Hardtanh(),


            BinConv2d(256, 256 , kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256,eps=1e-4, momentum=0.1 ),
            nn.Hardtanh(),


            BinConv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512,eps=1e-4, momentum=0.1),
            nn.Hardtanh(),


            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(512,eps=1e-4, momentum=0.1),
            nn.Hardtanh()

        )
        self.classifier = nn.Sequential(
            BinaryLinear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(),
            #nn.Dropout(0.5),
            BinaryLinear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(),
            #nn.Dropout(0.5),
            BinaryLinear(1024, 10),
            nn.BatchNorm1d(10),
 
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x
model=Model()
########################################################################
if args.cuda:
    #torch.cuda.set_device(3)
    model.cuda()


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum)

def adjust_learning_rate(optimizer, epoch):
    update_list = [55, 100, 150,200,400,600]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


def train(epoch):
    #global best_acc
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        
        for name, param in list(model.named_parameters()):
            if param.requires_grad:
                if hasattr(param,'org'):
                    print('###########',param.data)
                if not hasattr(param,'org'):
                    print('@@@@@@@@@@@@@@@@@@@@@@@@',param.data)
              
        """
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        """
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))
        
                

        if batch_idx % args.log_interval == 0:
            tlos.append(loss.data[0])
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        """
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
                                    
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        """
        
    acc = 100. * correct / len(test_loader)
    if acc > args.best_acc:
        args.best_acc = acc
        #save_state(model, best_acc)
    test_loss /= len(test_loader) 
       
    accur.append( 100.*correct/total)
    print(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Best Accuracy:: ',args.best_acc)

start = time.time()
time_graph=[]
e=[]
accur=[]
tlos=[]

for epoch in range(1, args.epochs + 1):
    adjust_learning_rate(optimizer, epoch)
    e.append(epoch)
    train(epoch)   
    seco=timeSince(start)
    time_graph.append(seco)
    test()

print(time_graph)
plt.title('Training for CIFAR10 with epoch', fontsize=20)
plt.ylabel('time (s)')
plt.plot(e,time_graph)
plt.show()
plt.title('Accuracy With epoch', fontsize=20)
plt.plot(e,accur)
plt.show()
plt.title('Test loss With epoch', fontsize=20)
plt.plot(tlos)
plt.show()



    
