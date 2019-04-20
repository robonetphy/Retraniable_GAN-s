# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:25:48 2019

@author: DELL
"""
    
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import pickle
import time 

batchSize = 64 
imageSize = 64 
numberOfIteration=1
epoch_no=0
G_losses = []
D_losses = []


transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 



dataset = dset.CIFAR10(root = './data', download = True, transform = transform) 
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class G(nn.Module):
    
    def __init__(self):
        super(G,self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
                nn.Tanh()
                )
        
    def forward(self,input):
        output =self.main(input)
        return output
    
    
netG = G()
netG.apply(weights_init)


class D(nn.Module):
    
    def __init__(self):
        super(D,self).__init__()
        self.main =  nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias= False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias= False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias= False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1, bias= False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1, 4, 1, 0, bias= False),
                nn.Sigmoid()
                 )
    def forward(self,input):
         output = self.main(input)
         return output.view(-1)


netD = D()
netD.apply(weights_init)


criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(),lr = 0.0002 , betas = (0.5, 0.999 ))
optimizerD = optim.Adam(netD.parameters(),lr = 0.0002 , betas = (0.5, 0.999 ))

def weights_save(state, filename):
    torch.save(state,filename)


def weights_load():
   if os.path.isfile('./weight/G_weight.pth') and os.path.isfile('./weight/D_weight.pth'): 
        weightsD = torch.load('./weight/D_weight.pth')
        netD.load_state_dict(weightsD['state_dict'])
        optimizerD. load_state_dict(weightsD['optimizer'])
        netD.train()
        weightsG = torch.load('./weight/G_weight.pth')
        netG.load_state_dict(weightsG['state_dict'])
        optimizerG.load_state_dict(weightsG['optimizer'])
        netG.train()
        global epoch_no
        epoch_no =weightsD['epoch']

def loss_load():
     if os.path.isfile('./loss/G_list.loss') and os.path.isfile('./loss/D_list.loss'):
        with open("./loss/G_list.loss", "rb") as fp:
            global G_losses
            G_losses  = pickle.load(fp)
        with open("./loss/D_list.loss", "rb") as fp:
            global D_losses
            D_losses  = pickle.load(fp)
            
def loss_store():
    with open("./loss/G_list.loss", "wb") as fp: 
          pickle.dump(G_losses, fp)
    with open("./loss/D_list.loss", "wb") as fp: 
          pickle.dump(D_losses, fp)

weights_load()
loss_load()


for epoch in range(numberOfIteration):
    for i, data in enumerate(dataloader, 0):
        
       netD.zero_grad()
      
       real, _ = data
       input = Variable(real)
       target = Variable(torch.ones(input.size()[0]))
       output = netD(input)
       errorD_real=criterion(output, target)
       
       noise = Variable( torch.randn(input.size()[0], 100, 1, 1))
       fake = netG(noise)
       target = Variable(torch.zeros(input.size()[0]))
       output =netD(fake.detach())
       errorD_fake=criterion(output, target)
       
       errorD = errorD_real + errorD_fake
       errorD.backward()
       optimizerD.step()
       
       netG.zero_grad()
       target = Variable(torch.ones(input.size()[0]))
       output =netD(fake)
       errorG = criterion(output, target)
       errorG.backward()
       optimizerG.step()
       
       epoch_no=epoch_no+1
       G_losses.append(errorG.data[0])
       D_losses.append(errorD.data[0])

       
       print('[%d][%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch_no,epoch,numberOfIteration, i, len(dataloader), errorD.data[0], errorG.data[0])) 
       if i % 100 == 0: # Every 100 steps:
            vutils.save_image(real, '%s/real_samples.png' %  "./output", normalize = True) 
            fake = netG(noise) 
            vutils.save_image(fake.data, '%s/fake_samples_%.f.png' % ("./output",time.time()), normalize = True) 
            weights_save({
                'epoch': epoch_no + 1,
                'state_dict': netG.state_dict(),
                'optimizer' : optimizerG.state_dict(),
                }, './weight/G_weight.pth')
            weights_save({
                'epoch': epoch_no + 1,
                'state_dict': netD.state_dict(),
                'optimizer' : optimizerD.state_dict(),
                }, './weight/D_weight.pth')
            loss_store()

weights_save({
                'epoch': epoch_no + 1,
                'state_dict': netG.state_dict(),
                'optimizer' : optimizerG.state_dict(),
                }, './weight/G_weight.pth')
weights_save({
    'epoch': epoch_no + 1,
    'state_dict': netD.state_dict(),
    'optimizer' : optimizerD.state_dict(),
    }, './weight/D_weight.pth')
loss_store()
















