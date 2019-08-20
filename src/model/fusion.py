import torch
import torch.nn as nn
import torch.nn.functional as F
from block.models.networks.fusions.fusions import Tucker

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    if bias:
        lin.bias.data.zero_()
    return lin

class Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear_i = linear(dim,dim)
        self.linear_w = linear(dim,dim)
        self.tensor_linear = linear(dim,dim)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-12)
        
        self.tucker = Tucker((dim, dim), dim, mm_dim=300, shared=False)
        
    def forward(self, tensor_i, tensor_w, cmlm=False):
        
        if cmlm:
            s_axis_i = tensor_i.size(1)
            s_axis_w = tensor_w.size(1)
            r = abs(s_axis_i - s_axis_w)
            
            if s_axis_i > s_axis_w:
                exp_tensor = tensor_w.new_zeros(tensor_w.size(0), r, tensor_w.size(2))
                tensor_w = torch.cat([tensor_w, exp_tensor],dim=1)
            else:
                exp_tensor = tensor_i.new_zeros(tensor_i.size(0), r, tensor_i.size(2))
                tensor_i = torch.cat([tensor_i, exp_tensor],dim=1)
        
        # compute co-attention map
        attn = self.tucker([tensor_i, tensor_w])
        
        if cmlm:
            if s_axis_i > s_axis_w:
                tensor_w = tensor_w[:,:s_axis_w,:]
            else:
                tensor_i = tensor_i[:,:s_axis_i,:]
        
        # use co-attention to compute the attended value for tensor_i
        tensor_i = self.attn(tensor_i, attn) + self.linear_i(tensor_i)
        tensor_i = self.layer_norm(tensor_i)
        
        # use co-attention to compute the attended value for tensor_w
        tensor_w = self.attn(tensor_w, attn) + self.linear_w(tensor_w)
        tensor_w = self.layer_norm(tensor_w)
        
        return tensor_i, tensor_w
        
    def attn(self, tensor, attn):
        
        assert len(attn.size()) == 3
        
        # here we use the same linear layer for both modals
        tensor = self.tensor_linear(tensor)
        
        attn = torch.matmul(tensor, attn.permute(0,2,1)).sum(-1)
        attn = F.softmax(attn, -1).unsqueeze(-1)
        tensor = tensor * attn
        
        return tensor