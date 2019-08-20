# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import numpy as np
from src.utils import to_cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion import Fusion
from .multiheadattention import MultiHeadAttention
from .tansformerffn import TransformerFFN

N_MAX_POSITIONS_LANG = 256  # maximum input sequence length
N_IMG_FEATURE_VECTOR = 2048
N_IMG_SPATIAL_INFO = 6
LANG_ID = 0
IMG_ID = 1

DECODER_ONLY_PARAMS = [
    'layer_norm15.%i.weight', 'layer_norm15.%i.bias',
    'encoder_attn.%i.q_lin.weight', 'encoder_attn.%i.q_lin.bias',
    'encoder_attn.%i.k_lin.weight', 'encoder_attn.%i.k_lin.bias',
    'encoder_attn.%i.v_lin.weight', 'encoder_attn.%i.v_lin.bias',
    'encoder_attn.%i.out_lin.weight', 'encoder_attn.%i.out_lin.bias'
]

TRANSFORMER_LAYER_PARAMS = [
    'attentions.%i.q_lin.weight', 'attentions.%i.q_lin.bias',
    'attentions.%i.k_lin.weight', 'attentions.%i.k_lin.bias',
    'attentions.%i.v_lin.weight', 'attentions.%i.v_lin.bias',
    'attentions.%i.out_lin.weight', 'attentions.%i.out_lin.bias',
    'layer_norm1.%i.weight', 'layer_norm1.%i.bias',
    'ffns.%i.lin1.weight', 'ffns.%i.lin1.bias',
    'ffns.%i.lin2.weight', 'ffns.%i.lin2.bias',
    'layer_norm2.%i.weight', 'layer_norm2.%i.bias'
]


logger = getLogger()


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    # nn.init.normal_(m.weight, mean=0, std=1)
    # nn.init.xavier_uniform_(m.weight)
    # nn.init.constant_(m.bias, 0.)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def get_masks_word(inputlen, lengths):
    """
        inputlen = lengths.max.item()
        lengths = torch.size(32) -> length for each element(eg, for cap-img, it equals len(cap) + len(img)) in the lengths list 
        Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= inputlen
    bs = lengths.size(0)
    alen = torch.arange(inputlen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]
    
    # sanity check
    assert mask.size() == (bs, inputlen)
        
    attn_mask = mask

    return mask, attn_mask

def get_masks_image(x):
    """
        Generate hidden states mask for image
    """
    bs, bptt_img, n_features = x.size()
    mask = torch.from_numpy(np.ones([bs, bptt_img], dtype=np.float32))
    mask[torch.sum(x, dim=-1) == 0] = 0.

    assert mask.size() == (bs, bptt_img)
        
    attn_mask = mask

    return mask, attn_mask


class PredLayer_ce(nn.Module):
    """
        Prediction layer
    """
    def __init__(self, params):
        super().__init__()
        self.dim = params.emb_dim
        
    def forward(self, tensor, y, pred_mask, candidates):
        masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)

        scores = torch.mm(masked_tensor, candidates.t())
        loss = F.cross_entropy(scores, y, reduction='mean')

        return scores, loss


class PredLayer(nn.Module):
    """
        Prediction layer
    """
    def __init__(self, params):
        super().__init__()
        self.n_words = params.n_words
        self.pad_index = params.pad_index
        dim = params.emb_dim

        self.proj = Linear(dim, params.n_words, bias=True)

    def forward(self, x, y):
        """
            Compute the loss, and optionally the scores.
        """
        scores = self.proj(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores, y, reduction='mean')

        return scores, loss


class TransformerModel(nn.Module):

    ATTRIBUTES = ['encoder', 'with_output', 'eos_index', 'pad_index', 'n_langs', 'n_words', 'dim', 'n_layers', 'n_heads', 'hidden_dim', 'dropout', 'attention_dropout', 'asm', 'asm_cutoffs', 'asm_div_value']

    def __init__(self, params, dico, with_output):
        """
            Transformer model (encoder or decoder).
        """
        super().__init__()

        # output layer
        self.with_output = with_output
        self.params = params

        # dictionary / languages
        self.n_words = params.n_words # len(dico)
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.mask_index = params.mask_index
        self.dico = dico
        assert len(self.dico) == self.n_words
        
        # images
        self.img_pad_index = params.img_pad_index
        self.img_mask_index = params.img_mask_index
        
        # modal
        self.n_modality = params.n_modality
        self.modal2id = params.modal2id
        
        # model parameters
        self.dim = params.emb_dim       # 1024 by default
        self.hidden_dim = self.dim * 4  # 4096 by default
        self.n_heads = params.n_heads   # 8 by default
        self.n_layers = params.n_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'

        # embeddings
        self.position_embeddings = Embedding(N_MAX_POSITIONS_LANG, self.dim)
        if params.sinusoidal_embeddings:
            create_sinusoidal_embeddings(N_MAX_POSITIONS_LANG, self.dim, out=self.position_embeddings.weight)
        self.modality_embeddings = Embedding(self.n_modality, self.dim)
        self.word_embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb_w = nn.LayerNorm(self.dim, eps=1e-12)
        self.layer_norm_emb_i = nn.LayerNorm(self.dim, eps=1e-12)
        
        # image embeddings
        self.img_spatial_embeddings = Linear(N_IMG_SPATIAL_INFO, self.dim) # 6 -> dim
        self.img_feature_embeddings = Linear(N_IMG_FEATURE_VECTOR, self.dim) # 2048 -> dim
        
        # img_cls embedding
        self.img_embedding = Embedding(1, self.dim)
    
        # transformer layers
        self.attentions_w = nn.ModuleList()
        self.layer_norm1_w = nn.ModuleList()
        self.ffns_w = nn.ModuleList()
        self.layer_norm2_w = nn.ModuleList()
        
        self.attentions_i = nn.ModuleList()
        self.layer_norm1_i = nn.ModuleList()
        self.ffns_i = nn.ModuleList()
        self.layer_norm2_i = nn.ModuleList()    
        
        self.fusion = nn.ModuleList()

        for _ in range(self.n_layers):
            self.fusion.append(Fusion(self.dim))
            
            self.attentions_w.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.layer_norm1_w.append(nn.LayerNorm(self.dim, eps=1e-12))
            self.ffns_w.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, 
                                              dropout=self.dropout, gelu_activation=params.gelu_activation))
            self.layer_norm2_w.append(nn.LayerNorm(self.dim, eps=1e-12))
            
            self.attentions_i.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.layer_norm1_i.append(nn.LayerNorm(self.dim, eps=1e-12))
            self.ffns_i.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, 
                                              dropout=self.dropout, gelu_activation=params.gelu_activation))
            self.layer_norm2_i.append(nn.LayerNorm(self.dim, eps=1e-12))

        # output layer
        if self.with_output:
            self.pred_layer = PredLayer(params)
            if params.share_inout_emb:
                self.pred_layer.proj.weight = self.word_embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(self, x, lengths, ipm=False, cmlm=False, candidates=None, 
            spatials=None, src_enc=None, src_len=None, positions=None, langs=None, cache=None):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices/ LongTensor(flen, bs, features), containing region features
            `lengths` LongTensor(bs), containing the length of each sentence/ length of the region per pass
            `candidates` only when cmlm or ipm is True
            `ipm` Boolean, if True, input is ipm
            `cmlm` Boolean, if True, input is cap-img pair
            `positions` LongTensor(slen, bs), containing word positions
        """
        
        # check inputs
        if(cmlm is False):
            if ipm:
                inputlen_ipm, bs, _ = x.size()
                assert candidates is not None
                spatials = spatials.transpose(0, 1)
                x = x.transpose(0, 1)  # batch size as dimension 0
                
                # generate masks
                mask_i, attn_mask_i = get_masks_image(x)
                mask_i, attn_mask_i = to_cuda(mask_i, attn_mask_i)
                
                # assume img id is 1
                modality = x.new_ones([bs, inputlen_ipm]).long()
            else:
                inputlen_mlm, bs = x.size()
                # generate masks
                mask_w, attn_mask_w = get_masks_word(inputlen_mlm, lengths)
                x = x.transpose(0, 1)  # batch size as dimension 0           
                
                positions = x.new(inputlen_mlm).long()
                positions = torch.arange(inputlen_mlm, out=positions).unsqueeze(0)
                
                # assume cap id is zero
                modality = x.new_zeros(x.size()).long()
                
            modality = to_cuda(modality)[0]
        else:
            assert len(x) == 2
            
            inputlen_cap, bs = x[0].size()
            inputlen_img, _, _ = x[1].size()
            assert spatials[0] is None # sanity check for word spatial, must be None
            spatials_img = spatials[1].transpose(0, 1) # bs, n_features, 6 
            
            assert lengths[0].size()[0] == lengths[1].size()[0] == bs
            assert lengths[0].max().item() <= inputlen_cap and lengths[1].max().item() <= inputlen_img
            
            x_cap = x[0].transpose(0, 1)  # batch size as dimension 0   
            x_img = x[1].transpose(0, 1)  # batch size as dimension 0
            
            mask_w, attn_mask_w = get_masks_word(inputlen_cap, lengths[0])
            mask_i, attn_mask_i = get_masks_image(x_img)
            mask_w, attn_mask_w, mask_i, attn_mask_i = to_cuda(mask_w, attn_mask_w, mask_i, attn_mask_i)
            
            positions = positions.transpose(0, 1)
            
#             print(f'x_cap: {x_cap.shape}')
#             print(f'x_img: {x_img.shape}')
#             print(f'mask_i: {mask_i.shape}')
#             print(f'mask_w: {mask_w.shape}')
#             print(f'positions: {positions.shape}')

            modality_cap = x_cap.new_zeros(bs, inputlen_cap).long()
            modality_img = x_img.new_ones(bs, inputlen_img).long()
            
            modality_cap, modality_img = to_cuda(modality_cap, modality_img)
            
            
        # do not recompute cached elements 
        if cache is not None:
            if(cmlm is False):
                if ipm:
                    _inputlen_ipm = inputlen_ipm - cache['inputlen_ipm']
                    spatials = spatials[:, -_inputlen_ipm:, :]
                    x = x[:, -_inputlen_ipm:, :]
                    mask_i = mask_w[:, -_inputlen_ipm:]
                    attn_mask_i = attn_mask_w[:, -_inputlen_ipm:]
                else:
                    _inputlen_mlm = inputlen_mlm - cache['inputlen_mlm']
                    x = x[:, -_inputlen_mlm:]
                    positions = positions[:, -_inputlen_mlm:]
                    mask_w = mask_w[:, -_inputlen_mlm:]
                    attn_mask_w = attn_mask_w[:, -_inputlen_mlm:]
            
        # embedding/ position embeddings
        if(cmlm is False):
            if ipm:
                # image embedding
                tensor_i = self.img_feature_embeddings(x)
                tensor_i = tensor_i + self.img_spatial_embeddings(spatials)
                tensor_i = tensor_i + self.modality_embeddings(modality)
                tensor_i = self.layer_norm_emb_i(tensor_i)
                tensor_i = F.dropout(tensor_i, p=self.dropout, training=self.training)
                tensor_i *= mask_i.unsqueeze(-1).to(tensor_i.dtype)

                # turn image candidates into language space
                candidates = self.img_feature_embeddings(candidates)
                
                with torch.no_grad():
                    tensor_w = tensor_i.new_zeros(tensor_i.size())
                    tensor_w = to_cuda(tensor_w)[0]
            else:
                # word embeddings
                tensor_w = self.word_embeddings(x)
                tensor_w = tensor_w + self.position_embeddings(positions).expand_as(tensor_w)                
                tensor_w = tensor_w + self.modality_embeddings(modality)
                tensor_w = self.layer_norm_emb_w(tensor_w)
                tensor_w = F.dropout(tensor_w, p=self.dropout, training=self.training)
                tensor_w *= mask_w.unsqueeze(-1).to(tensor_w.dtype)
                
                with torch.no_grad():
                    tensor_i = tensor_w.new_zeros(tensor_w.size())
                    tensor_i = to_cuda(tensor_i)[0]
        else:
            tensor_i = self.img_feature_embeddings(x_img[:,1:])
            assert x_img[:,0].sum().long() == 0
            tensor_i_ave = self.img_embedding(torch.sum(x_img[:,0], dim=-1).long().unsqueeze(-1))  
            tensor_i = torch.cat((tensor_i_ave, tensor_i), 1)
            tensor_i = tensor_i + self.img_spatial_embeddings(spatials_img)
            tensor_i = tensor_i + self.modality_embeddings(modality_img)
            tensor_i = self.layer_norm_emb_i(tensor_i)
            tensor_i = F.dropout(tensor_i, p=self.dropout, training=self.training)
            tensor_i *= mask_i.unsqueeze(-1).to(tensor_i.dtype)
            
            # tranform raw img fetures into same img embedding space
            candidates = self.img_feature_embeddings(candidates)
            
            tensor_w = self.word_embeddings(x_cap)
            tensor_w = tensor_w + self.position_embeddings(positions).expand_as(tensor_w)
            tensor_w = tensor_w + self.modality_embeddings(modality_cap)
            tensor_w = self.layer_norm_emb_w(tensor_w)
            tensor_w = F.dropout(tensor_w, p=self.dropout, training=self.training)
            tensor_w *= mask_w.unsqueeze(-1).to(tensor_w.dtype)

            
        # transformer layers
        for i in range(self.n_layers):
            
            tensor_i, tensor_w = self.fusion[i](tensor_i, tensor_w, cmlm)
            
            if (ipm or cmlm):
                # self attention for image transformer
                attn_i = self.attentions_i[i](tensor_i, attn_mask_i, cache=cache)
                attn_i = F.dropout(attn_i, p=self.dropout, training=self.training)
                tensor_i = tensor_i + attn_i
                tensor_i = self.layer_norm1_i[i](tensor_i)

                # FFN for image transformer
                tensor_i = tensor_i + self.ffns_i[i](tensor_i)
                tensor_i = self.layer_norm2_i[i](tensor_i)
                tensor_i *= mask_i.unsqueeze(-1).to(tensor_i.dtype)
            
            if ((not ipm) or cmlm):
                # self attention for caption/language transformer
                attn_w = self.attentions_w[i](tensor_w, attn_mask_w, cache=cache)
                attn_w = F.dropout(attn_w, p=self.dropout, training=self.training)
                tensor_w = tensor_w + attn_w
                tensor_w = self.layer_norm1_w[i](tensor_w)

                # FFN for caption/language transformer
                tensor_w = tensor_w + self.ffns_w[i](tensor_w)
                tensor_w = self.layer_norm2_w[i](tensor_w)
                tensor_w *= mask_w.unsqueeze(-1).to(tensor_w.dtype)


        # update cache length
        if cache is not None:
            if(cmlm is False):
                if ipm: 
                    cache['inputlen_ipm'] += tensor_i.size(1)
                else:
                    cache['inputlen_mlm'] += tensor_w.size(1)

        # move back sequence length to dimension 0
        tensor_w = tensor_w.transpose(0, 1)
        tensor_i = tensor_i.transpose(0, 1)
        
        if cmlm:
            return tensor_i, tensor_w, candidates
        elif ipm:
            return tensor_i, candidates
        else:
            return tensor_w

    def predict(self, tensor, pred_mask, y, y_cap=None, y_img=None):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            'candidates' is FloatTensor of shape (pred_mask)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        
        masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        scores, loss = self.pred_layer(masked_tensor, y)  

        return scores, loss