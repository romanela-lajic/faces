import torch
from torch import nn

import math
import numpy as np
import  torch.nn.functional as F

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class MLLoss(nn.Module):
    def __init__(self, s=64.0):
        super(MLLoss, self).__init__()
        self.s = s
    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta.mul_(self.s)
        return cos_theta

'''

# from https://github.com/HuangYG123/CurricularFace/blob/master/head/metrics.py
class ElasticArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False,std=0.0125):
        super(ElasticArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        self.std=std


        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # nn.init.xavier_uniform_(self.kernel)
        nn.init.normal_(self.kernel, std=0.01)

        self.easy_margin = easy_margin


    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1 + 1e-5, 1 + 1e-5)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        index = torch.where(label != -1)[0]
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device).clamp(self.m-self.std, self.m+self.std) # Fast converge .clamp(self.m-self.std, self.m+self.std)
        with torch.no_grad():
            #distmat = cos_theta[index, label.view(-1)].detach().clone()
            _, idicate_cosie = torch.sort(target_logit, dim=0, descending=True)
            margin, _ = torch.sort(margin, dim=0)
        cos_m=torch.cos(margin)
        sin_m=torch.sin(margin)
        th=torch.cos(math.pi-margin)
        mm=torch.sin(math.pi-margin)*margin

        cos_theta_m = target_logit * cos_m - sin_theta * sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(target_logit > th, cos_theta_m, target_logit - mm)

        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output  # , origin_cos * self.s
'''
class ElasticArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50,std=0.0125, random=True):
        super(ElasticArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.std=std
        self.random=random

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device)#.clamp(self.m-self.std, self.m+self.std) # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if not self.random:
            with torch.no_grad():
                distmat = cos_theta[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)

        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta


class ElasticCosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35,std=0.0125, random=False):
        super(ElasticCosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.std=std
        self.random=random

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device)  # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if not self.random:
            with torch.no_grad():
                distmat = cos_theta[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        m_hot.scatter_(1, label[index, None], margin)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret

class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret

def loss_func(feat1, feat2):
    return  1- F.cosine_similarity(feat1, feat2).abs().mean()

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelHeader(nn.Module):
    """
    A header for multi-label classification that takes features from a backbone network
    and outputs logits for multiple binary classification tasks.
    
    Args:
        in_features (int): Number of input features (embedding size)
        out_features (int): Number of output classes (40 for CelebA)
        hidden_dim (int, optional): Dimension of hidden layer. Defaults to 1024.
        dropout (float, optional): Dropout probability. Defaults to 0.2.
        use_batchnorm (bool, optional): Whether to use batch normalization. Defaults to True.
    """
    def __init__(self, in_features, out_features, hidden_dim=1024, dropout=0.2, use_batchnorm=True):
        super(MultiLabelHeader, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        
        # Hidden layer
        self.fc1 = nn.Linear(in_features, hidden_dim)
        
        # BatchNorm layer
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity()
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
        
        # Output layer
        self.fc2 = nn.Linear(hidden_dim, out_features)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization and zero bias"""
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        """
        Forward pass of the multi-label header.
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, out_features)
        """
        # Hidden layer with ReLU activation and batch norm
        x = F.relu(self.bn1(self.fc1(x)))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Output layer (no activation here - BCEWithLogitsLoss expects raw logits)
        x = self.fc2(x)
        
        return x
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, hidden_dim={self.hidden_dim}'    
    




