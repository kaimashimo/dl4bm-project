# This code is modified from https://github.com/jakesnell/prototypical-networks 

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from methods.meta_template import MetaTemplate


class RelationNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(RelationNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.relation_module = RelationModule(input_size = self.feat_dim*2)

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()
        z_query = z_query.contiguous()

        z_support = z_support.view(self.n_way, self.n_support, -1).sum(1)
        z_query = z_query.view(self.n_way * self.n_query, -1)

        # Precompute all concatenated pairs
        concatenated_pairs = torch.cat([torch.cat((z_support, query.repeat(self.n_way, 1)), dim=1) for query in z_query])
    
        # Pass the concatenated pairs through the Relation Module in batches
        relation_scores = self.relation_module(concatenated_pairs)
    
        # Reshape the scores
        relation_scores = relation_scores.view(self.n_way * self.n_query, self.n_way)

        return relation_scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )

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
        m.bias.data.zero_()
    elif classname.find('RelationConvBlock') != -1:
        print('Detected RelationConvBlock in module name')
        conv_layer = m.C
        n = conv_layer.kernel_size[0] * conv_layer.kernel_size[1] * conv_layer.out_channels
        conv_layer.weight.data.normal_(0, math.sqrt(2. / n))
        if conv_layer.bias is not None:
            conv_layer.bias.data.zero_()

class RelationModule(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(RelationModule, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(0.40))
        self.layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(0.40)) 
        self.layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

def cosine_similarity(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    # Normalize the rows of x and y
    x_normalized = F.normalize(x, p=2, dim=1)
    y_normalized = F.normalize(y, p=2, dim=1)

    # Calculate cosine similarity
    similarity = torch.mm(x_normalized, y_normalized.transpose(0, 1))

    return -similarity
