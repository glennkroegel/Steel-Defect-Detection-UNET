'''
created_by: Glenn Kroegel
date: 2 August 2019

'''

import pandas as pd
import numpy as np
from collections import defaultdict, Counter

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm
import shutil

from dataset import SteelDataset
from utils import count_parameters, accuracy
from callbacks import Hook
from config import NUM_EPOCHS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

status_properties = ['loss', 'accuracy']

#https://discuss.pytorch.org/t/how-to-resume-training/8583

#############################################################################################################################

insize = (1, 3, 256, 1600)
def get_hooks(m):
    md = {int(k):v for k,v in m._modules.items()}
    hooks = {k: Hook(layer) for k, layer in md.items()}
    x = torch.randn(insize).requires_grad_(False)
    m.eval()(x)
    out_szs = {k:h.output.shape for k,h in hooks.items()}
    inp_szs = {k:h.input[0].shape for k,h in hooks.items()}
    return hooks, inp_szs, out_szs

#############################################################################################################################

class SamePad(nn.Module):
    def __init__(self, in_dim, stride, ks):
        super(SamePad, self).__init__()
        h = in_dim
        w = in_dim
        self.h = h
        self.w = w
        self.stride = stride
        self.out_h = int(np.ceil(float(h) / float(stride)))
        self.out_w = int(np.ceil(float(w) / float(stride)))
        # self.pad = nn.ZeroPad2d(padding=(self.out_w//2, self.out_w//2, self.out_h//2, self.out_h//2))
        self.p = (ks-1)//2
        self.pad = nn.ZeroPad2d(self.p)

    def forward(self, x):
        x = self.pad(x)
        return x

class Dense(nn.Module):
    def __init__(self, in_size, out_size, bias=False):
        super(Dense, self).__init__()
        self.fc = nn.Linear(in_size, out_size, bias=bias)
        self.bn = nn.BatchNorm1d(out_size)
        self.drop = nn.Dropout(0.1)
        self.act = nn.LeakyReLU()
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x):
        x = x.view(-1, self.in_size)
        x = self.bn(self.drop(self.act(self.fc(x))))
        return x

class Conv(nn.Module):
    def __init__(self, in_c, out_c, ks=3, stride=1, padding=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=stride, bias=bias, padding=padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout(0.1)
        self.act = nn.LeakyReLU()
        self.in_size = in_c
        self.out_size = out_c

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class ResBlock(nn.Module):
    def __init__(self, n):
        super(ResBlock, self).__init__()
        self.c1 = Conv(n, n)
        self.c2 = Conv(n, n)

    def forward(self, x):
        return x + self.c2(self.c1(x))

class ConvResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvResBlock, self).__init__()
        self.conv = Conv(in_c=in_c, out_c=out_c, stride=2)
        self.res_block = ResBlock(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.res_block(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        j = 64
        self.encode = nn.Sequential(ConvResBlock(3, j),
        ConvResBlock(j, j), 
        ConvResBlock(j, j), 
        ConvResBlock(j, 2*j),
        ConvResBlock(2*j, 128),
        ConvResBlock(128, 256)) # Conv(256, 512, stride=2)

        # self.encode = nn.Sequential(ConvResBlock(3, 8),
        # ConvResBlock(8, 16), 
        # ConvResBlock(16, 32), 
        # ConvResBlock(32, 64),
        # ConvResBlock(64, 128),
        # ConvResBlock(128, 256)) # Conv(256, 512, stride=2)

    def forward(self, x):
        bs = x.size(0)
        x = self.encode(x)
        return x

class BottleNeck(nn.Module):
    def __init__(self, n):
        super(BottleNeck, self).__init__()
        # self.c = Conv(n, 2*n, stride=2)
        # self.c1 = Conv(2*n, 4*n)
        # self.c2 = Conv(4*n, 2*n)
        a = 1
        self.c = nn.MaxPool2d(2)#Conv(n, a*n, stride=2)
        self.c1 = Conv(a*n, 2*a*n)
        self.c2 = Conv(2*a*n, a*n)

    def forward(self, x):
        x = self.c(x)
        x = self.c2(self.c1(x))
        return x

class ExpansionBlock(nn.Module):
    def __init__(self, up_in_c, bypass_c, hook):
        super(ExpansionBlock, self).__init__()
        self.act = nn.LeakyReLU()
        self.hook = hook
        c_in = up_in_c + bypass_c
        c_out = c_in//2
        self.c1 = Conv(c_in, c_out)
        self.c2 = Conv(c_out, c_out)

    def forward(self, x):
        bypass = self.hook.output
        bypass_sz = bypass.size()
        x = F.interpolate(x, size=bypass_sz[-2:], mode='nearest')
        if len(bypass_sz) != 4:
            x = torch.cat([bypass.unsqueeze(0), x], dim=1)
        else:
            x = torch.cat([bypass, x], dim=1)
        x = self.act(x)
        x = self.c2(self.c1(x))
        return x

class FinalLayers(nn.Module):
    def __init__(self, up_in_c, bypass_c, hook):
        super(FinalLayers, self).__init__()
        self.act = nn.LeakyReLU()
        self.hook = hook
        c_in = up_in_c
        c_mid = c_in//8
        self.reduce1 = nn.Sequential(Conv(c_in, c_in//2),
                                    Conv(c_in//2, c_in//4), 
                                    Conv(c_in//4, c_mid))
        c_out = 4
        # self.reduce2 = nn.Sequential(Conv(c_mid+3, c_mid), Conv(c_mid, c_out))
        self.reduce2 = Conv(c_mid, c_out) #nn.Sequential(Conv(c_mid+3, c_mid), Conv(c_mid, c_out))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bypass = self.hook.input[0]
        bypass_sz = bypass.size()
        x = F.interpolate(x, size=bypass_sz[-2:], mode='nearest')
        x = self.reduce1(x)
        # x = torch.cat([bypass, x], dim=1)
        # x = self.act(x)
        x = self.reduce2(x)
        x = self.sigmoid(x)
        return x

#############################################################################################################################

class Contraction(nn.Module):
    def __init__(self):
        super(Contraction, self).__init__()
        self.enc1 = ResNet()

    def forward(self, x):
        bs = x.size(0)
        x = self.enc1(x)
        return x

class Expansion(nn.Module):
    def __init__(self):
        super(Expansion, self).__init__()
        self.dec1 = ResNet()

    def forward(self, x):
        bs = x.size(0)
        x = self.dec1(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        contraction = Contraction().eval()
        hooks, inp_szs, enc_szs = get_hooks(contraction.enc1.encode)
        idxs = list(enc_szs.keys())
        x_sz = enc_szs[len(enc_szs) - 1]
        bottleneck = BottleNeck(x_sz[1]).eval()
        x = torch.randn(x_sz).requires_grad_(False)
        x = bottleneck(x)
        layers = [contraction, bottleneck]
        for idx in list(reversed(idxs)):
            up_in_c = x.size(1)
            bypass_in_c = enc_szs[idx][1]
            exp_block = ExpansionBlock(up_in_c, bypass_in_c, hooks[idx]).eval()
            layers.append(exp_block)
            x = exp_block(x)
        up_in_c = x.size(1)
        bypass_in_c = enc_szs[idx][1]
        final_lyrs = FinalLayers(up_in_c, bypass_in_c, hooks[0]).eval()
        x = final_lyrs(x)
        layers.append(final_lyrs)
        [print(count_parameters(x)) for x in layers]
        layers = [m.train() for m in layers]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x 

###########################################################################

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Learner():
    '''Training loop'''
    def __init__(self, epochs=NUM_EPOCHS):
        self.model = Net().to(device)
        self.criterion = nn.L1Loss().to(device)#nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-3, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=8, eta_min=5e-4)
        self.epochs = epochs

        self.train_loader = torch.load('train_loader.pt')
        self.cv_loader = torch.load('cv_loader.pt')

        self.train_loss = []
        self.cv_loss = []

        self.best_loss = 1e50
        print('Model Parameters: ', count_parameters(self.model))
        # print(summary(self.model, (4, 3, 256, 1600)))

    def train(self, train_loader, model, criterion, optimizer, epoch):
        model.train()
        props = {k:0 for k in status_properties}
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print(i)
            x, targets = data
            logits = model(x.to(device))
            loss = criterion(logits, targets.to(device))
            props['loss'] += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            clip_grad_norm_(self.model.parameters(), 0.25)
            L = len(train_loader)
        props = {k:v/L for k,v in props.items()}
        return props

    def step(self):
        '''Actual training loop.'''
        for epoch in tqdm(range(self.epochs)):
            train_props = self.train(self.train_loader, self.model, self.criterion, self.optimizer, epoch)
            self.scheduler.step(epoch)
            lr = self.scheduler.get_lr()[0]
            print(lr)
            self.train_loss.append(train_props['loss'])
            # cross validation
            props = {k:0 for k in status_properties}
            with torch.no_grad():
                for _, data in enumerate(self.cv_loader):
                    self.model.eval()
                    x, targets = data
                    logits = self.model(x.to(device))
                    val_loss = self.criterion(logits, targets.to(device))
                    props['loss'] += val_loss.item()
                L = len(self.cv_loader)
                self.cv_loss.append(props['loss'])
                props = {k:v/L for k,v in props.items()}
                if epoch % 1 == 0:
                    self.status(epoch, train_props, props)
                if props['loss'] < self.best_loss:
                    print('dumping model...')
                    path = 'model' + '.pt'
                    torch.save(self.model, path)
                    self.best_loss = props['loss']
                    is_best = True
                save_checkpoint(
                    {'epoch': epoch + 1,
                    'lr': lr, 
                    'state_dict': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict(), 
                    'best_loss': self.best_loss}, is_best=is_best)
                is_best=False

    def status(self, epoch, train_props, cv_props):
        s0 = 'epoch {0}/{1}\n'.format(epoch, self.epochs)
        s1, s2 = '',''
        for k,v in train_props.items():
            s1 = s1 + 'train_'+ k + ': ' + str(v) + ' '
        for k,v in cv_props.items():
            s2 = s2 + 'cv_'+ k + ': ' + str(v) + ' '
        print(s0 + s1 + s2)

if __name__ == "__main__":
    try:
        mdl = Learner()
        mdl.step()
    except KeyboardInterrupt:
        pd.to_pickle(mdl.train_loss, 'train_loss.pkl')
        pd.to_pickle(mdl.cv_loss, 'cv_loss.pkl')
        print('Stopping')
