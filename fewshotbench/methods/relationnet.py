# This code is modified from https://github.com/floodsung/LearningToCompare_FSL/blob/master/miniimagenet/miniimagenet_train_few_shot.py and https://github.com/snap-stanford/comet/blob/master/TM/methods/relationnet.py

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

from methods.meta_template import MetaTemplate

class RelationNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(RelationNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.relation_hidden_size = 128
        self.relation_module = RelationModule(self.feat_dim * 2, self.relation_hidden_size)

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way, self.n_support, -1)
        z_support = torch.sum(z_support, 1)  # Sum over the support set

        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        z_support_ext = z_support.unsqueeze(0).repeat(self.n_way * self.n_query, 1, 1)
        z_query_ext = z_query.unsqueeze(1).repeat(1, self.n_way, 1).view(self.n_way * self.n_query, self.n_way, -1)
        
        # Dimension of [batch_size, feature_dimension * 2]
        extend_pairs = torch.cat((z_support_ext, z_query_ext), 2).view(-1, self.feat_dim*2)
        # print('extend_pairs.shape',extend_pairs.shape)
        # print('self. feat_dim', self.feat_dim)
        scores = self.relation_module(extend_pairs).view(-1, self.n_way)
        return scores
    
    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'kernel_size'):
        print('Detected Conv in module name')
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        print('Detected BatchNorm in module name')
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        print('Detected Linear in module name')
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
    elif classname.find('RelationConvBlock') != -1:
        print('Detected RelationConvBlock in module name')
        conv_layer = m.C
        n = conv_layer.kernel_size[0] * conv_layer.kernel_size[1] * conv_layer.out_channels
        conv_layer.weight.data.normal_(0, math.sqrt(2. / n))
        if conv_layer.bias is not None:
            conv_layer.bias.data.zero_()

class RelationModule(nn.Module):

    def __init__(self, input_size, hidden_size=128):
        super(RelationModule, self).__init__()
    
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(0.5)

        self.layers = nn.Sequential(
            self.fc1, self.bn1, nn.ReLU(),
            self.dropout,
            self.fc2, self.bn1, nn.ReLU(),
            self.dropout,
            self.fc3, self.bn1, nn.ReLU(),
            self.dropout,
            self.fc4, nn.Sigmoid()
        )

        self.apply(weights_init)    
    
    def forward(self, x):
        out = self.layers(x)
        return out


