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
train_loader = D.DataLoader(datasets.CIFAR10('./data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ])),batch_size=args.batch_size, shuffle=True)
    
    #test_loaer
test_loader = D.DataLoader(datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ])),batch_size=args.test_batch_size, shuffle=True)


################################################################
#MODEL
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
"""

class Mod(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def model():
    return Model(BasicBlock, [2,2,2,2])

"""
"""
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            BinaryConv2d(3, 16, kernel_size=3, padding=2),
            nn.BatchNorm2d(16, momentum=args.momentum, eps=args.eps),
            #nn.Dropout(0.2),
            nn.MaxPool2d(2)
            )
        
        self.dr=nn.Dropout(0.5)
        self.layer2 = nn.Sequential(
            BinaryConv2d(16,32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32, momentum=args.momentum, eps=args.eps),
            nn.MaxPool2d(2),
            #nn.Dropout(0.2),
            )
        self.dr=nn.Dropout(0.5)
        self.fc1 = BinaryLinear(2592, 1024)
        self.fc2 = BinaryLinear(1024, 10)
        #self.fc3 = BinaryLinear(1024, 10)
        
    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        #out = self.fc3(out)
        
        return out
"""    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.xnor = nn.Sequential(
                nn.BinConv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),
                BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                BinConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.BinConv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x

model=Model()
########################################################################
if args.cuda:
    #torch.cuda.set_device(3)
    model.cuda()


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum)


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
    #global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


"""
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
        
"""
"""
start = time.time()
time_graph=[]
e=[]
"""
for epoch in range(1, args.epochs + 1):
    #e.append(epoch)
    train(epoch)   
    #seco=timeSince(start)
    #time_graph.append(seco)
    test()
"""
print(time_graph)
plt.title('Training for MNIST with epoch', fontsize=20)
plt.ylabel('time (s)')
plt.plot(e,time_graph,'bo')
plt.show()
plt.title('Accuracy With epoch', fontsize=20)
plt.plot(e,accur,'bo')
plt.show()
"""

    
