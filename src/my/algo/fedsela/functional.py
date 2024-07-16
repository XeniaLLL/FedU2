from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.special import logsumexp
# Projector
class ProjectionMLPOrchestra(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super(ProjectionMLPOrchestra, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer1_bn = nn.BatchNorm1d(hidden_dim, affine=True)
        self.layer2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.layer2_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        x = F.relu(self.layer1_bn(self.layer1(x)))
        x = self.layer2_bn(self.layer2(x))
        return x


class SelaModel(nn.Module):
    def __init__(self, encoder, head_classifiers):
        super(SelaModel, self).__init__()
        self.encoder = encoder
        self.nhc = len(head_classifiers)
        self.head_classifiers_names =[k for k in head_classifiers.keys()]
        for k, v in head_classifiers.items():
            setattr(self, k, v)

    def forward(self, x):
        x = self.encoder(x)
        output_nhc = []
        for top_layer in self.head_classifiers_names:
            output_nhc.append(getattr(self, top_layer)(x))
        return output_nhc

def py_softmax(x, axis=None):
    """stable softmax"""
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))