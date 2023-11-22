# This code is modified from https://github.com/floodsung/LearningToCompare_FSL/blob/master/miniimagenet/miniimagenet_train_few_shot.py and https://github.com/snap-stanford/comet/blob/master/TM/methods/relationnet.py

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

from methods.meta_template import MetaTemplate
from utils.data_utils import one_hot


class ProtoNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(ProtoNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

        self.relation_module = RelationNetwork( self.feat_dim , 8, self.loss_type ) #relation net features are not pooled, so self.feat_dim is [dim, w, h] 

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()
        z_support = z_support.view(self.n_way, self.n_support, -1) 
        z_support = torch.sum(z_support, 1).squeeze(1)
        
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        z_support_ext = z_support.unsqueeze(0).repeat(self.n_query * self.n_way,1,1,1,1)
        z_query_ext = z_query.unsqueeze(0).repeat(self.n_way,1,1,1,1)

        extend_final_feat_dim = self.feat_dim.copy()
        extend_final_feat_dim[0] *= 2

        extend_pairs = torch.cat((z_support_ext,z_query_ext),2).view(-1, *extend_final_feat_dim)
        scores =  self.relation_module(extend_pairs ).view(-1,self.n_way)

        return scores

    
    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


class RelationConvBlock(nn.Module):

    def __init__(self, indim, outdim, padding = 0):
        super(RelationConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.C      = nn.Conv2d(indim, outdim, 3, padding = padding )
        self.BN     = nn.BatchNorm2d(outdim, momentum=1, affine=True)
        self.relu   = nn.ReLU()
        self.pool   = nn.MaxPool2d(2)

        self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self,x):
        out = self.trunk(x)
        return out
    

class RelationNetwork(nn.Module):

    def __init__(self, in_layers, out_layers, hidden_size, padding = 0):

        self.conv1 = RelationConvBlock(in_layers[0], out_layers[0], padding = padding)
        self.conv2 = RelationConvBlock(in_layers[1], out_layers[1], padding = padding)
        
        self.fc1 = nn.Linear(out_layers[1]*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.layers = self.layers + [self.fc1, self.fc2, self.relu, self.sigmoid]
        self.layers = nn.Sequential(*self.layers)


        for m in self.modules():
            weights_init(m)

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0),-1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out


