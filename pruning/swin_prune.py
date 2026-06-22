import logging
import torch
import torchvision
from tqdm import tqdm
import timm
from pruning import vit_prune_utils
import torch.nn as nn
import torch.nn.functional as F
from pruning.vit_prune_utils import DistillLoss
from pruning.vit_prune_utils import attention_manager
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.data import Mixup
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from backbones import swin

class SWIN_prune():

    def __init__(self, args, train_loader, test_loader):
        self.logger = logging.getLogger("Agent")
        self.img_size = 112
        self.num_epochs = args.num_epoch
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = swin.SwinTransformer(img_size=112, num_classes=args.embedding_size).to('cuda')
        self.all_pruned=[]


    def load_model_parameters(self, file):
        self.model.load_state_dict(torch.load(file))

    def updateLN(self):
        for block in self.model.blocks:
            block.norm2.weight.grad.data.add_(0.0001 * torch.sign(block.norm2.weight.data))  # L1

    def non_zero_el(self):
        non_zero_params = 0
        for param in self.model.parameters():
            non_zero_params += torch.count_nonzero(param).item()  # .item() to get the scalar value
        return non_zero_params

    def set_zero(self):
        j=0
        for child in self.model.layers.children():
            for grandchild in child.children():
                for block in grandchild.children():
                    if isinstance(block, swin.SwinTransformerBlock):
                        if j < len(self.all_pruned):
                            bn = block.norm2
                            bn1 = block.norm1
                            mlp1 = block.mlp.fc1
                            with torch.no_grad():
                                pruned = self.all_pruned[j]
                                block.attn.qkv.weight[:, pruned] = 0

                                j = j + 1
                                pruned = self.all_pruned[j]
                                mlp1.weight[:, pruned] = 0
                                j = j + 1


    def compress(self, method, sparsity):
        if method=='l1':
            for child in self.model.layers.children():
                for grandchild in child.children():
                    for block in grandchild.children():
                        if isinstance(block, swin.SwinTransformerBlock):
                            bn = block.norm2
                            bn1 = block.norm1
                            mlp1 = block.mlp.fc1
                            stayed, pruned = vit_prune_utils.channel_selection(block.attn.qkv, sparsity=1.-sparsity, method=method)
                            with torch.no_grad():
                                block.attn.qkv.weight[:, pruned] = 0
                                block.attn.qkv.weight[:, pruned].requires_grad = False
                                self.all_pruned.append(pruned)
                                #block.attn.qkv.bias[pruned]=0
                                #block.attn.qkv.bias[pruned].requires_grad=False
                                temp=torch.ones_like(block.attn.qkv.weight)
                                temp[:,pruned]=0
                                temp = torch.ones_like(block.attn.qkv.bias)
                                temp[pruned] = 0
                            stayed, pruned = vit_prune_utils.channel_selection(mlp1, sparsity=1.-sparsity, method=method)
                            with torch.no_grad():
                                mlp1.weight[:, pruned] = 0
                                #mlp1.bias[pruned] = 0
                                self.all_pruned.append(pruned)
                                temp = torch.ones_like(mlp1.weight)
                                temp[:, pruned] = 0
                                temp = torch.ones_like(mlp1.bias)
                                temp[pruned] = 0

        if method=='slimming':
            self.all_pruned=[]
            for child in self.model.layers.children():
                for grandchild in child.children():
                    for block in grandchild.children():
                        if isinstance(block, swin.SwinTransformerBlock):

                            bn = block.norm2
                            bn1 = block.norm1
                            mlp1 = block.mlp.fc1
                            stayed, pruned = vit_prune_utils.channel_selection(bn1, sparsity=1.-sparsity, method=method)
                            with torch.no_grad():
                                block.attn.qkv.weight[:, pruned] = 0
                                block.attn.qkv.weight[:, pruned].requires_grad = False
                                self.all_pruned.append(pruned)
                                #block.attn.qkv.bias[pruned]=0
                                #block.attn.qkv.bias[pruned].requires_grad=False
                                temp=torch.ones_like(block.attn.qkv.weight)
                                temp[:,pruned]=0
                                temp = torch.ones_like(block.attn.qkv.bias)
                                temp[pruned] = 0
                            stayed, pruned = vit_prune_utils.channel_selection(bn, sparsity=1.-sparsity, method=method)
                            with torch.no_grad():
                                mlp1.weight[:, pruned] = 0
                                #mlp1.bias[pruned] = 0
                                self.all_pruned.append(pruned)
                                temp = torch.ones_like(mlp1.weight)
                                temp[:, pruned] = 0
                                temp = torch.ones_like(mlp1.bias)
                                temp[pruned] = 0


        if method=='random':
            self.all_pruned=[]
            for child in self.model.layers.children():
                for grandchild in child.children():
                    for block in grandchild.children():
                        if isinstance(block, swin.SwinTransformerBlock):
                            bn = block.norm2
                            bn1 = block.norm1
                            mlp1 = block.mlp.fc1
                            stayed, pruned = vit_prune_utils.channel_selection(block.attn.qkv, sparsity=1.-sparsity, method=method)
                            with torch.no_grad():
                                block.attn.qkv.weight[:, pruned] = 0
                                block.attn.qkv.weight[:, pruned].requires_grad = False
                                self.all_pruned.append(pruned)
                                #block.attn.qkv.bias[pruned]=0
                                #block.attn.qkv.bias[pruned].requires_grad=False
                                temp=torch.ones_like(block.attn.qkv.weight)
                                temp = torch.ones_like(block.attn.qkv.bias)
                                temp[pruned] = 0
                            stayed, pruned = vit_prune_utils.channel_selection(mlp1, sparsity=1.-sparsity, method=method)
                            with torch.no_grad():
                                mlp1.weight[:, pruned] = 0
                                #mlp1.bias[pruned] = 0
                                self.all_pruned.append(pruned)
                                temp = torch.ones_like(mlp1.weight)
                                temp[:, pruned] = 0
                                temp = torch.ones_like(mlp1.bias)
                                temp[pruned] = 0



