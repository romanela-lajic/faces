import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def channel_selection(module, sparsity, method, pruned=None):

    if method=='random':
        num_channel = module.weight.shape[0]
        num_pruned = int(math.ceil(num_channel * sparsity))
        num_stayed = num_channel - num_pruned
        pruneweight=torch.sum(torch.abs(module.weight), dim=0)
        numkeep = int(pruneweight.shape[0] * (1. - sparsity))
        _descend = torch.randperm(pruneweight.shape[0], device=pruneweight.device)[:numkeep]
        mask = torch.zeros_like(pruneweight).long()
        mask[_descend] = 1
        indices_stayed = torch.where(mask == 1)[0]
        indices_pruned = torch.where(mask != 1)[0]

    if method=='l1':
        num_channel = module.weight.shape[0]
        num_pruned = int(math.ceil(num_channel * sparsity))
        num_stayed = num_channel - num_pruned
        pruneweight=torch.sum(torch.abs(module.weight), dim=0)
        numkeep = int(pruneweight.shape[0] * (1. - sparsity))
        _ascend = torch.argsort(pruneweight)
        _descend = torch.flip(_ascend, (0,))[:numkeep]
        mask = torch.zeros_like(pruneweight).long()
        mask[_descend] = 1
        indices_stayed = torch.where(mask == 1)[0]
        indices_pruned = torch.where(mask != 1)[0]

    if method=='slimming':
        num_channel = module.weight.shape[0]
        num_pruned = int(math.ceil(num_channel * sparsity))
        num_stayed = num_channel - num_pruned
        pruneweight=module.weight.abs()
        numkeep = int(pruneweight.shape[0] * (1. - sparsity))

        _ascend = torch.argsort(pruneweight)
        _descend = torch.flip(_ascend, (0,))[:numkeep]
        mask = torch.zeros_like(pruneweight).long()
        mask[_descend] = 1
        indices_stayed = torch.where(mask == 1)[0]
        indices_pruned = torch.where(mask != 1)[0]
        
        
        # num_channel = module.weight.shape[0]
        # num_pruned = int(math.ceil(num_channel * sparsity)*iterative_step)
        # num_stayed = num_channel - num_pruned
        # pruneweight=module.weight.abs()

        # _ascend = torch.argsort(pruneweight)
        # _descend = torch.flip(_ascend, (0,))[:num_stayed]
        # mask = torch.zeros_like(pruneweight).long()
        # mask[_descend] = 1
        # indices_stayed = torch.where(mask == 1)[0]
        # indices_pruned = torch.where(mask != 1)[0]
    return indices_stayed, indices_pruned




class attention_manager(object):
    def __init__(self, model, target_layer='attention_target'):

        self.target_layer = target_layer
        self.attention = []
        self.handler = []

        self.model = model

        self.register_hook(self.model)

    def register_hook(self, model):
        def get_attention_features(_, inputs, outputs):
            self.attention.append(outputs)

        for name, layer in model._modules.items():
            # but recursively register hook on all it's module children
            if isinstance(layer, nn.Sequential):
                self.register_hook(layer)
            else:
                if name == self.target_layer:
                    handle = layer.register_forward_hook(get_attention_features)
                    self.handler.append(handle)

                else:
                    for name, layer2 in layer._modules.items():
                        if name == self.target_layer:
                            handle = layer2.register_forward_hook(get_attention_features)
                            self.handler.append(handle)

    def remove_hook(self):
        for handler in self.handler:
            handler.remove()

class DistillLoss(nn.Module):
    def __init__(self, distill_attn_param, device):
        super(DistillLoss, self).__init__()
        self.device = device
        self.distill_attn_param = distill_attn_param
        self.criterion = self.cosine_loss

    def cosine_loss(self, l, h):
        l = l.view(l.size(0), -1)
        h = h.view(h.size(0), -1)
        return torch.mean(1.0 - F.cosine_similarity(l, h))

    def forward(self, low_feature, high_feature):
        # calculate the attention distillation
        loss_sum = 0.
        for l, h in zip(low_feature, high_feature):
            l = l.reshape(l.size(0), -1)
            h = h.reshape(h.size(0), -1)

            d_loss = self.criterion(l, h)

            loss_sum += d_loss * self.distill_attn_param

        return loss_sum

def prune_attention_heads(model, num_pruned_heads):
    """
    Function to prune attention heads by zeroing out their weights and blocking gradients.
    """
    # Access the attention layer from the given transformer block

    pruned=[]
    for i, block in enumerate(model.blocks):
        attn_layer = block.attn
        bn1 = block.norm1.weight
        bn1 = bn1.reshape(6, -1)
        original_num_heads = attn_layer.num_heads
        num_keep=original_num_heads-num_pruned_heads
        importance_score = torch.linalg.norm(bn1, 1, dim=1)
        _ascend = torch.argsort(importance_score)
        _descend = torch.flip(_ascend, (0,))[:num_keep]
        mask = torch.zeros_like(importance_score).long()
        mask[_descend] = 1
        indices_stayed = torch.where(mask == 1)[0]
        heads_to_prune = torch.where(mask != 1)[0]
        pruned.append(heads_to_prune)
        # Get the original number of attention heads and head dimensions
        head_dim = attn_layer.head_dim
        embed_dim = attn_layer.qkv.weight.shape[1]  # This comes from qkv weight shape

        for head_to_prune in heads_to_prune:
            # Ensure the head index to prune is valid
            if head_to_prune >= original_num_heads or head_to_prune < 0:
                raise ValueError(f"Invalid attention head index: {head_to_prune}")

            # Zero out the QKV weights for the pruned head (Q, K, V)
            qkv_weight = attn_layer.qkv.weight.data
            qkv_bias = attn_layer.qkv.bias.data

            # Find the indices that correspond to the pruned attention head
            start_idx = head_to_prune * head_dim
            end_idx = (head_to_prune + 1) * head_dim

            # Zero the weights for the pruned head
            qkv_weight[start_idx:end_idx, :] = 0
            qkv_bias[start_idx:end_idx] = 0  # Zero the bias for the pruned head

            # Set the QKV weight and bias back into the model
            attn_layer.qkv.weight.data = qkv_weight
            attn_layer.qkv.bias.data = qkv_bias

            # Block the gradients for the pruned head's weights
            attn_layer.qkv.weight.requires_grad = True  # Allow gradient computation for QKV weight in general
            attn_layer.qkv.bias.requires_grad = True  # Allow gradient computation for QKV bias in general

            # Set the gradient of the pruned head's weights to None (effectively blocking it)
            attn_layer.qkv.weight.data[start_idx:end_idx, :] = attn_layer.qkv.weight.data[start_idx:end_idx,
                                                               :].cpu().detach()
            attn_layer.qkv.weight.data[start_idx:end_idx, :].requires_grad = False
            attn_layer.qkv.bias.data[start_idx:end_idx] = attn_layer.qkv.bias.data[start_idx:end_idx].cpu().detach()
            attn_layer.qkv.bias.data[start_idx:end_idx].requires_grad = False

    return model, pruned




