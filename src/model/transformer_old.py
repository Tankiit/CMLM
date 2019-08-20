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


def get_masks_word(inputlen, lengths, cmlm=False):
    """
        inputlen = lengths.max.item()
        lengths = torch.size(32) -> length for each element(eg, for cap-img, it equals len(cap) + len(img)) in the lengths list 
        Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= inputlen
    bs = lengths.size(0)
    alen = torch.arange(inputlen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]
    
    if cmlm:
        mask_cls_token = mask.new_ones(bs,1)
        mask = torch.cat([mask_cls_token, mask], dim=-1)
        
        # sanity check
        assert mask.size() == (bs, inputlen + 1)
    else:
        # sanity check
        assert mask.size() == (bs, inputlen)
        
    attn_mask = mask

    return mask, attn_mask

def get_masks_image(x, cmlm=False):
    """
        Generate hidden states mask for image
    """
    bs, bptt_img, n_features = x.size()
    mask = torch.from_numpy(np.ones([bs, bptt_img], dtype=np.float32))
    mask[torch.sum(x, dim=-1) == 0] = 0.
    
    if cmlm:
        img_cls_token = torch.FloatTensor(bs, 1).fill_(1.)
        mask = torch.cat([img_cls_token, mask], 1)
        
        # sanity check
        assert mask.size() == (bs, bptt_img+1)
    else:
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

    def forward(self, x, y, get_scores=False, candidates=None, cmlm=False, ipm=False):
        """
            Compute the loss, and optionally the scores.
        """
        if cmlm:
            scores_cap = self.proj(x[0]).view(-1, self.n_words)
            scores_img = torch.mm(x[1], candidates.t())
            scores = [scores_cap, scores_img]
            loss = F.cross_entropy(scores_cap, y[0], reduction='mean') + F.cross_entropy(scores_img, y[1], reduction='mean')
        else:
            assert (y == self.pad_index).sum().item() == 0

            scores = self.proj(x).view(-1, self.n_words)
            loss = F.cross_entropy(scores, y, reduction='mean')

        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert x.dim() == 2
        return self.proj.log_prob(x) if self.asm else self.proj(x)


class TransformerModel(nn.Module):

    ATTRIBUTES = ['encoder', 'with_output', 'eos_index', 'pad_index', 'n_langs', 'n_words', 'dim', 'n_layers', 'n_heads', 'hidden_dim', 'dropout', 'attention_dropout', 'asm', 'asm_cutoffs', 'asm_div_value']

    def __init__(self, params, dico, with_output):
        """
            Transformer model (encoder or decoder).
        """
        super().__init__()

        # output layer
        self.with_output = with_output

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
            assert lengths[0].size()[0] == lengths[1].size()[0] == bs
            assert lengths[0].max().item() <= inputlen_cap and lengths[1].max().item() <= inputlen_img
            
            inputlen_cap, bs = x[0].size()
            inputlen_img, _, _ = x[1].size()
            assert spatials[0] is None # sanity check for word spatial, must be None
            spatials_img = spatials[1].transpose(0, 1) # bs, n_features, 6 
            
            mask_w, attn_mask_w = get_masks_word(inputlen_cap, lengths[0], cmlm=True)
            mask_i, attn_mask_i = get_masks_image(x[1], cmlm=True)
            mask_w, attn_mask_w, mask_i, attn_mask_i = to_cuda(mask_w, attn_mask_w, mask_i, attn_mask_i)
            
            x_cap = x[0].transpose(0, 1)  # batch size as dimension 0   
            x_img = x[1].transpose(0, 1)  # batch size as dimension 0
            
            positions = positions.transpose(0, 1)

            modality_cap = x_cap.new_zeros(bs, inputlen_cap+1).long()
            modality_img = x_img.new_ones(bs, inputlen_img+1).long()
            
            modality_cap, modality_img = to_cuda(modality_cap, modality_img)
            
            
        # add an ave img token in the 1st position of img input tokens
        # this ave img token is used to summarize the 36 img features/regions
        if cmlm:
            ave_spatial = torch.FloatTensor(bs, 1, N_IMG_SPATIAL_INFO).fill_(1.)
            ave_spatial[:, :, 0:2] = 0. # top_right x and y is 0
            spatials_img = torch.cat([ave_spatial, spatials_img],1)
            ave_imgToken = torch.LongTensor(bs, 1).fill_(0)
            ave_spatial, ave_imgToken = to_cuda(ave_spatial, ave_imgToken)
            
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
            tensor_i = self.img_feature_embeddings(x_img)
            tensor_i_ave = self.img_embedding(ave_imgToken)  
            tensor_i = torch.cat((tensor_i_ave, tensor_i_), 1)
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
            
            tensor_i, tensor_w = self.fusion[i](tensor_i, tensor_w)
            
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

    def predict(self, tensor, pred_mask, get_scores, y=None, y_cap=None, y_img=None, cmlm=False, ipm=False, candidates=None):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            'candidates' is FloatTensor of shape (pred_mask)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        
        if(cmlm is False):
            if ipm:
                masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
                scores, loss = self.pred_layer(masked_tensor, y, get_scores, candidates=candidates, ipm=ipm, cmlm=cmlm)  
            else:
                masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
                scores, loss = self.pred_layer(masked_tensor, y, get_scores, ipm=ipm, cmlm=cmlm)   
        else:
            pred_mask_cap = pred_mask[0]
            pred_mask_img = pred_mask[1]
            
            len_cap = pred_mask_cap.size()[0]
            
            tensor_cap = tensor[:len_cap,:,:]
            tensor_img = tensor[len_cap:,:,:]
            
            masked_tensor_cap = tensor_cap[pred_mask_cap.unsqueeze(-1).expand_as(tensor_cap)].view(-1, self.dim)
            masked_tensor_img = tensor_img[pred_mask_img.unsqueeze(-1).expand_as(tensor_img)].view(-1, self.dim) 
            
            masked_tensor = [masked_tensor_cap, masked_tensor_img]
            y = [y_cap, y_img]
            
            scores, loss = self.pred_layer(masked_tensor, y, get_scores, candidates=candidates, cmlm=cmlm, ipm=ipm)   

        return scores, loss

    def generate(self, src_enc, src_len, tgt_lang_id, max_len=200, sample_temperature=None):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """

        # input batch
        bs = len(src_len)
        assert src_enc.size(0) == bs

        # generated sentences
        generated = src_len.new(max_len, bs)  # upcoming output
        generated.fill_(self.pad_index)       # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)    # we use <EOS> for <BOS> everywhere

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)

        # language IDs
        langs = src_len.new(max_len).long().fill_(tgt_lang_id)
        langs = langs.unsqueeze(1).expand(max_len, bs)

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)

        # cache compute states
        cache = {'slen': 0}

        while cur_len < max_len:

            # compute word scores
            tensor = self.forward(
                'fwd',
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                langs=langs[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                cache=cache
            )
            assert tensor.size() == (1, bs, self.dim)
            tensor = tensor.data[-1, :, :]               # (bs, dim)
            scores = self.pred_layer.get_scores(tensor)  # (bs, n_words)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(F.softmax(scores / sample_temperature, dim=1), 1).squeeze(1)
            assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)

        # sanity check
        assert (generated == self.eos_index).sum() == 2 * bs

        return generated[:cur_len], gen_len

    def generate_beam(self, src_enc, src_len, tgt_lang_id, beam_size, length_penalty, early_stopping, max_len=200):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """

        # check inputs
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1

        # batch size / number of words
        bs = len(src_len)
        n_words = self.n_words

        # expand to beam size the source latent representations / source lengths
        src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view((bs * beam_size,) + src_enc.shape[1:])
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(self.pad_index)                   # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)                # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)

        # language IDs
        langs = positions.clone().fill_(tgt_lang_id)

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        cache = {'inputlen': 0}

        # done sentences
        done = [False for _ in range(bs)]

        while cur_len < max_len:

            # compute word scores
            tensor = self.forward(
                'fwd',
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                langs=langs[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                cache=cache
            )
            assert tensor.size() == (1, bs * beam_size, self.dim)
            tensor = tensor.data[-1, :, :]               # (bs * beam_size, dim)
            scores = self.pred_layer.get_scores(tensor)  # (bs * beam_size, n_words)
            scores = F.log_softmax(scores, dim=-1)       # (bs * beam_size, n_words)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)            # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[:cur_len, sent_id * beam_size + beam_id].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_index, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in cache.keys():
                if k != 'inputlen':
                    cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        # sanity check
        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty
