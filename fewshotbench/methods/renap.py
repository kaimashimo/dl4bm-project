# This code is modified from https://github.com/jakesnell/prototypical-networks 

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from methods.meta_template import MetaTemplate


class ReNAP(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(ReNAP, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.adaptive_proto_module = AdaptivePrototypicalModule(self.feat_dim)
        self.relation_module = RelationModule(input_size = self.feat_dim*2)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # Else if mps is available
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()
        z_query = z_query.contiguous()

        z_support = z_support.view(self.n_way, self.n_support, -1)

        z_prototypes = torch.zeros(self.n_way, self.feat_dim).cuda()
        for i in range(self.n_way):
            z_proto_i = self.adaptive_proto_module(z_support[i])
            z_prototypes[i] = z_proto_i.mean(0)

        z_query = z_query.view(self.n_way * self.n_query, -1)

        # Precompute all concatenated pairs
        concatenated_pairs = torch.cat([torch.cat((z_proto.unsqueeze(0).repeat(self.n_way * self.n_query, 1), z_query), dim=1) for z_proto in z_prototypes])
    
        # Pass the concatenated pairs through the Relation Module in batches
        relation_scores = self.relation_module(concatenated_pairs)
    
        # Reshape the scores
        relation_scores = relation_scores.view(self.n_way * self.n_query, self.n_way)

        return relation_scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.to(self.device))

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )

class AdaptivePrototypicalModule(nn.Module):
    def __init__(self, feature_dim, hidden_size=512):
        super(AdaptivePrototypicalModule, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, feature_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RelationModule(nn.Module):
    def __init__(self, input_size, device='cuda'):
        super(RelationModule, self).__init__()
        self.device = device

        with open('experiments/relationnet/config', 'r') as file:
            lines = file.readlines()
            hidden_size = int(lines[0].strip())  # Reads first line as integer
            dropout = float(lines[1].strip())    # Reads second line as float
            n_layers = int(lines[2].strip())     # Reads third line as integer

        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                          nn.BatchNorm1d(hidden_size),
                          nn.ReLU(),
                          nn.Dropout(dropout)) for i in range(n_layers - 1)
        ])
        self.layer_final = nn.Linear(hidden_size, 1).to(device)
        self.sigmoid = nn.Sigmoid()

        # Move all layers to the specified device
        self.layers.to(device)

    def forward(self, x):
        x = x.to(self.device)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_final(x)
        x = self.sigmoid(x)
        return x

