import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from methods.meta_template import MetaTemplate

class ReNAP(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(ReNAP, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.relation_module = RelationModule(self.feat_dim * 2, 128)
        self.apl_module = AdaptivePrototypeLearningModule(self.n_support, self.feat_dim)

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way, self.n_support, -1)
        z_support = self.apl_module(z_support)

        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        z_support_ext = z_support.unsqueeze(0).repeat(self.n_way * self.n_query, 1, 1)
        z_query_ext = z_query.unsqueeze(1).repeat(1, self.n_way, 1).view(self.n_way * self.n_query, self.n_way, -1)

        extend_pairs = torch.cat((z_support_ext, z_query_ext), 2).view(-1, self.feat_dim*2)
        scores = self.relation_module(extend_pairs).view(-1, self.n_way)
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query)

class AdaptivePrototypeLearningModule(nn.Module):
    def __init__(self, n_support, input_size):
        super(AdaptivePrototypeLearningModule, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1)  # Conv1d with 1 input channel and 1 output channel
        self.conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=1)  # Conv1d with 1 input channel and 1 output channel
        self.relu = nn.ReLU()
        self.n_support = n_support

    def forward(self, x):
        x = x.view(-1, 1, x.size(-1))  # Reshaping to [n_way * n_support, 1, feat_dim]
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = x.view(-1, self.n_support, x.size(-1))  # Reshape back to [n_way, n_support, feat_dim]
        return x.mean(dim=1)  # Averaging over the support set

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'kernel_size'):
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
    elif classname.find('RelationConvBlock') != -1:
        conv_layer = m.C
        n = conv_layer.kernel_size[0] * conv_layer.kernel_size[1] * conv_layer.out_channels
        conv_layer.weight.data.normal_(0, math.sqrt(2. / n))
        if conv_layer.bias is not None:
            conv_layer.bias.data.zero_()

class RelationModule(nn.Module):

    def __init__(self, input_size, hidden_size=128):
        super(RelationModule, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.layers = nn.Sequential(self.fc1, self.relu, self.fc2, self.sigmoid)

        for m in self.modules():
            weights_init(m)

    def forward(self, x):
        out = self.layers(x)
        return out
