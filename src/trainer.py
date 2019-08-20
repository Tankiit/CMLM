# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import math
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from apex.fp16_utils import FP16_Optimizer
from .model.transformer import PredLayer_ce

from .utils import get_optimizer, to_cuda, concat_batches, concat_batches_mm
from .utils import parse_lambda_config, update_lambdas


logger = getLogger()


class Trainer(object):

    def __init__(self, data, params):
        """
            Initialize trainer.
        """
        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # stopping criterion used for early stopping
        # for example: "_valid_mlm_ppl;_ce,10" ==> 'ce' stands for cross_entropy_loss
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            crit = split[0].split(';')
            assert len(crit) == 2
            if crit[0][0] == '_':
                self.stopping_criterion = (crit[0][1:], crit[1][1:], False)
            else:
                self.stopping_criterion = (crit[0], crit[1], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[-1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # data iterators
        self.iterators = {}

        # probability of masking out / randomize / not modify words to predict
        params.word_pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])
        params.img_pred_probs = torch.FloatTensor([params.img_mask, params.img_keep, params.img_rand])

        # probabilty to predict a word
        counts = np.array(list(self.data['dico'].counts.values())) # frequency of the words
        params.mask_scores = np.maximum(counts, 1) ** -params.word_sample_alpha
        params.mask_scores[params.pad_index] = 0  # do not predict <PAD> index, we already know second index refers to <pad> token
        params.mask_scores[params.cls_word_index] = 0
        params.mask_scores[counts == 0] = 0       # do not predict special tokens

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0
        self.n_images = 0
        self.n_pairs = 0
        
        self.stats = OrderedDict(
            [('processed_s', 0), ('processed_w', 0)] +
            [('processed_unmasked_regions', 0), ('processed_images', 0)] +
            [('processed_pairs', 0)] +
            [('MLM-%s' % l, []) for l in params.langs] +
            [('IPM-%s' % m, []) for m in params.images] +
            [('CMLM-%s-%s' % (m1, m2), []) for m1,m2 in data['cross_modal'].keys()] +
            [('CMLM-%s' % m1, []) for m1,_ in data['cross_modal'].keys()] +
            [('CMLM-%s' % m2, []) for _,m2 in data['cross_modal'].keys()]
        )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params)
        
        # prediction layer for cross_entropy
        self.pred_ce = PredLayer_ce(params)

    def get_optimizer_fp(self, module):
        """
            Build optimizer.
        """
        assert module in ['model']
        optimizer = get_optimizer(getattr(self, module).parameters(), self.params.optimizer)
        if self.params.fp16:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        return optimizer

    def optimize(self, loss, modules):
        """
        Optimize.
        """
        if type(modules) is str:
            modules = [modules]

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # zero grad
        for module in modules:
            self.optimizers[module].zero_grad()

        # backward
        if self.params.fp16:
            assert len(modules) == 1, "fp16 not implemented for more than one module"
            self.optimizers[module].backward(loss)
        else:
            loss.backward()

        # clip gradients
        if self.params.clip_grad_norm > 0:
            for module in modules:
                if self.params.fp16:
                    self.optimizers[module].clip_master_grads(self.params.clip_grad_norm)
                else:
                    clip_grad_norm_(getattr(self, module).parameters(), self.params.clip_grad_norm)

        # optimization step
        for module in modules:
            self.optimizers[module].step()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        update_lambdas(self.params, self.n_total_iter)
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_iter % 5 != 0:
            return

        s_iter = "%7i - " % self.n_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # transformer learning rate
        lr = self.optimizers[self.MODEL_NAMES[0]].param_groups[0]['lr']
        s_lr = " - Transformer LR = {:.4e}".format(lr)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
            self.stats['processed_s'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_s'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def get_iterator_lang(self, iter_name, lang, stream):
        """
            Create a new iterator for a dataset.
        """
        logger.info("Creating new training data iterator (%s) ..." % ','.join([str(x) for x in [iter_name, lang] if x is not None]))
        if stream:
            iterator = self.data['lang_stream'][lang]['train'].get_iterator(shuffle=True)
        else:
            iterator = self.data['lang'][lang]['train'].get_iterator(
                shuffle=True,
                group_by_size=self.params.group_by_size,
                n_sentences=-1,
            )

        self.iterators[(iter_name, lang)] = iterator
        return iterator
    
    def get_iterator_mm(self, iter_name, m1, m2):
        """
            Create a new iterator for a dataset.
        """
        
        logger.info("Creating new training data iterator (%s) ..." % ','.join([str(x) for x in [iter_name, m1, m2] if x is not None]))

        _m1, _m2 = (m1, m2)
        iterator = self.data['cross_modal'][(_m1, _m2)]['train'].get_iterator(shuffle=True,
                                                                              group_by_size=self.params.group_by_size)

        self.iterators[(iter_name, m1, m2)] = iterator
        return iterator
    
    def get_iterator_img(self, iter_name, img, stream):
        """
            Create a new iterator for a img dataset.
        """
        logger.info("Creating new training data iterator (%s) ..." % ','.join([str(x) for x in [iter_name, img] if x is not None]))

        # current stage we consider only image input as stream format
        if stream:
            iterator = self.data['image_stream'][img]['train'].get_iterator(shuffle=True)

        self.iterators[(iter_name, img)] = iterator
        return iterator

    def get_batch_lang(self, iter_name, lang, stream=False):
        """
            Return a batch of sentences from a dataset.
        """
        assert lang in self.params.langs
        iterator = self.iterators.get((iter_name, lang), None)
        if iterator is None:
            iterator = self.get_iterator_lang(iter_name, lang, stream)
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator_lang(iter_name, lang, stream)
            x = next(iterator)
        return x
    
    def get_batch_mm(self, iter_name, m1, m2):
        """
            Return a batch of sentences/images/caption-images-pair from a dataset.
        """
        assert m1 in self.params.crossmodal
        assert m2 in self.params.crossmodal
        iterator = self.iterators.get((iter_name, m1, m2), None)
        if iterator is None:
            iterator = self.get_iterator_mm(iter_name, m1, m2)
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator_mm(iter_name, m1, m2)
            x = next(iterator)
        return x
    
    def get_batch_img(self, iter_name, img, stream=False):
        """
            Return a batch of sentences from a dataset.
        """
        assert img in self.params.images
        iterator = self.iterators.get((iter_name, img), None)
        if iterator is None:
            iterator = self.get_iterator_img(iter_name, img, stream)
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator_img(iter_name, img, stream)
            x = next(iterator)
        return x

    def word_shuffle(self, x, l):
        """
        Randomly shuffle input words.
        """
        if self.params.word_shuffle == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, self.params.word_shuffle, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        assert self.params.word_shuffle > 1
        x2 = x.clone()
        for i in range(l.size(0)):
            # generate a random permutation
            scores = np.arange(l[i] - 1) + noise[:l[i] - 1, i]
            permutation = scores.argsort()
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l

    def word_dropout(self, x, l):
        """
        Randomly drop input words.
        """
        if self.params.word_dropout == 0:
            return x, l
        assert 0 < self.params.word_dropout < 1

        # define words to drop
        eos = self.params.eos_index
        assert (x[0] == eos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[j, i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(eos)
            assert len(new_s) >= 3 and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.params.pad_index)
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_blank(self, x, l):
        """
        Randomly blank input words.
        """
        if self.params.word_blank == 0:
            return x, l
        assert 0 < self.params.word_blank < 1

        # define words to blank
        eos = self.params.eos_index
        assert (x[0] == eos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[j, i] else self.params.mask_index for j, w in enumerate(words)]
            new_s.append(eos)
            assert len(new_s) == l[i] and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.params.pad_index)
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l

    def add_noise(self, words, lengths):
        """
        Add noise to the encoder input.
        """
        words, lengths = self.word_shuffle(words, lengths)
        words, lengths = self.word_dropout(words, lengths)
        words, lengths = self.word_blank(words, lengths)
        return words, lengths
    
    def add_noise_img(self, words, lengths):
        """
            Add noise to the encoder input.
        """
        words, lengths = self.word_shuffle(words, lengths)
        words, lengths = self.word_dropout(words, lengths)
        words, lengths = self.word_blank(words, lengths)
        return words, lengths

    def mask_out_lang(self, x, lengths):
        """
            Decide of random words to mask out, and what target they get assigned.
        """
        params = self.params
        slen, bs = x.size()

        # define target words to predict
        if params.word_sample_alpha == 0:
            pred_mask = np.random.rand(slen, bs) <= params.word_pred
            pred_mask = torch.from_numpy(pred_mask.astype(np.uint8))
        else:
            x_prob = params.mask_scores[x.flatten()]
            n_tgt = math.ceil(params.word_pred * slen * bs)
            tgt_ids = np.random.choice(len(x_prob), n_tgt, replace=False, p=x_prob / x_prob.sum())
            pred_mask = torch.zeros(slen * bs, dtype=torch.uint8)
            pred_mask[tgt_ids] = 1
            pred_mask = pred_mask.view(slen, bs)

        # do not predict padding
        pred_mask[x == params.pad_index] = 0
        pred_mask[x == params.cls_word_index] = 0

#         # mask a number of words == 0 [8] (faster with fp16)
#         if params.fp16:
#             pred_mask = pred_mask.view(-1)
#             n1 = pred_mask.sum().item()
#             n2 = max(n1 % 8, 8 * (n1 // 8))
#             if n2 != n1:
#                 pred_mask[torch.nonzero(pred_mask).view(-1)[:n1 - n2]] = 0
#             pred_mask = pred_mask.view(slen, bs)
#             assert pred_mask.sum().item() % 8 == 0

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_rand = _x_real.clone().random_(params.n_words)
        _x_mask = _x_real.clone().fill_(params.mask_index)  # eg. params.mask_index = 5
        probs = torch.multinomial(params.word_pred_probs, len(_x_real), replacement=True)
        _x = _x_mask * (probs == 0).long() + _x_real * (probs == 1).long() + _x_rand * (probs == 2).long()
        x = x.masked_scatter(pred_mask, _x)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask
    
    def mask_out_img(self, x, lengths):
        """
            Decide of random features to mask out, and what target they get assigned.
        """
        params = self.params
        all_regions = x.size()[0] # n_bbox * n_features

        # define target img_feats to predict
        if params.img_sample_alpha == 0:
            pred_mask = np.random.rand(all_regions) <= params.img_pred
            pred_mask = torch.from_numpy(pred_mask.astype(np.uint8))

        # do not predict padding
        pred_mask[torch.sum(x, dim=-1) == params.img_pad_index] = 0

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_rand = x[np.random.choice(all_regions, _x_real.shape[0])]
        _x_mask = _x_real.clone().fill_(params.img_mask_index)  # eg. params.img_mask_index = 1
        
        probs = torch.multinomial(params.img_pred_probs, len(_x_real), replacement=True).unsqueeze(-1)
        _x = _x_mask * (probs == 0).float() + _x_real * (probs == 1).float() + _x_rand * (probs == 2).float()
        x = x.t().masked_scatter(pred_mask, _x.t()).t()
        
        # permutate the real patches sequence
        perm = torch.from_numpy(np.random.permutation(_x_real.shape[0]))
        _x_candidates = _x_real.clone()[perm]

        assert x.size()[0] == all_regions
        assert pred_mask.size()[0] == all_regions
        
        x = x.view(params.batch_size, params.bptt_img, -1).transpose(0,1) # bptt_img x bs x 2048
        pred_mask = pred_mask.view(params.batch_size, -1).transpose(0,1) # bptt_img x bs

        return x, perm, _x_candidates, pred_mask
    
    def mask_out_mm(self, x):
        """
            Decide of random words to mask out, and what target they get assigned.
            x = (x_cap, x_img)
            lengths = (lengths_cap, lengths_img)
        """
        
        x_cap = x[0]
        x_img = x[1]
        
        params = self.params
        slen, bs = x_cap.size()
        all_regions = x_img.size()[0] # n_bbox * n_features

        # define target cap-img pair to predict
        # should have use a separate alpha that is different from mlm_step and ipm_step
        assert params.word_sample_alpha == 0 and params.img_sample_alpha == 0
            
        pred_mask_cap = np.random.rand(slen, bs) <= params.word_pred
        pred_mask_cap = torch.from_numpy(pred_mask_cap.astype(np.uint8))
        # do not predict padding
        pred_mask_cap[x_cap == params.pad_index] = 0
        pred_mask_cap[x_cap == params.cls_word_index] = 0

        # generate possible cap targets / update x input
        _x_real_cap = x_cap[pred_mask_cap]
        _x_rand_cap = _x_real_cap.clone().random_(params.n_words)
        _x_mask_cap = _x_real_cap.clone().fill_(params.mask_index)  # eg. params.mask_index = 5
        probs_cap = torch.multinomial(params.word_pred_probs, len(_x_real_cap), replacement=True)
        _x_cap = _x_mask_cap * (probs_cap == 0).long() + \
                 _x_real_cap * (probs_cap == 1).long() + \
                 _x_rand_cap * (probs_cap == 2).long()
        x_cap = x_cap.masked_scatter(pred_mask_cap, _x_cap)
        
        pred_mask_img = np.random.rand(all_regions) <= params.img_pred
        pred_mask_img = torch.from_numpy(pred_mask_img.astype(np.uint8))
        # do not predict padding
        pred_mask_img[torch.sum(x_img, dim=-1) == params.img_pad_index] = 0
        
        # generate possible img targets / update x input
        _x_real_img = x_img[pred_mask_img]
        _x_rand_img = x_img[np.random.choice(all_regions, _x_real_img.shape[0])]
        _x_mask_img = _x_real_img.clone().fill_(params.img_mask_index)  # eg. params.img_mask_index = 1
        probs_img = torch.multinomial(params.img_pred_probs, len(_x_real_img), replacement=True).unsqueeze(-1)
        _x_img = _x_mask_img * (probs_img == 0).float() + \
                 _x_real_img * (probs_img == 1).float() + \
                 _x_rand_img * (probs_img == 2).float()
        
        x_img = x_img.t().masked_scatter(pred_mask_img, _x_img.t()).t()
        
        # permutate the real patches sequence
        perm_img = torch.from_numpy(np.random.permutation(_x_real_img.shape[0]))
        _x_candidates = _x_real_img.clone()[perm_img]

        assert 0 <= x_cap.min() <= x_cap.max() < params.n_words
        assert x_cap.size() == (slen, bs)
        assert x_img.size()[0] == all_regions
        assert pred_mask_cap.size() == (slen, bs)
        assert pred_mask_img.size()[0] == all_regions
        
        x_img = x_img.view(params.batch_size, -1, x_img.shape[-1]).transpose(0,1)
        pred_mask_img = pred_mask_img.view(params.batch_size, -1).transpose(0,1)
        
        x = [x_cap, x_img]
        pred_mask = [pred_mask_cap, pred_mask_img]
        
        return x, _x_real_cap, perm_img, _x_candidates, pred_mask
    
    def add_cls(self, x, spatial_x, lengths):
        
        params = self.params
        
        x_cap = x[0]
        x_img = x[1]
        
        slen, bs = x_cap.size()
        all_regions, features = x_img.size() # n_bbox * n_features
        
        x_img = x_img.view(bs, -1, features)
        ave_imgToken = x_img.new_zeros(bs, 1, features)
        x_img = torch.cat([ave_imgToken, x_img], dim=1)
        x_img = x_img.view(-1, features)
        lengths[1] = lengths[1] + 1
        
        spatial_img = spatial_x[1]
        spatial_img = spatial_img.transpose(0,1)
        
        ave_spatial = torch.FloatTensor(bs, 1, params.spatial_feat).fill_(1.)
        ave_spatial[:, :, 0:2] = 0. # top_right x and y is 0
        spatial_img = torch.cat([ave_spatial, spatial_img],1)
        
        spatial_x[1] = spatial_img.transpose(0,1)
        
        # add cls word token in front of the x_cap
        ave_clsToken = x_cap.new_ones(1, bs).fill_(params.cls_word_index)
        x_cap = torch.cat([ave_clsToken, x_cap],dim=0)
        lengths[0] = lengths[0] + 1
        
        x = [x_cap, x_img]
        
#         print(f'x_cap: {x_cap.size()}')
#         print(f'x_img: {x_img.size()}')
#         print(f'spatial_x[1]: {spatial_x[1].size()}')
#         exit()
        
        return x, spatial_x, lengths

    def generate_batch_lang(self, lang, name):
        """
            Prepare a batch
        """
        params = self.params
        x, lengths = self.get_batch_lang(name, lang, stream=True)
        positions = None

        return x, lengths, positions, (None, None)

    def generate_batch_img(self, img, name):
        """
            Prepare a batch (for causal or non-causal mode).
        """
        params = self.params
        
        x, spatial_x, feature_length = self.get_batch_img(name, img, stream=True)
        positions = None
    
        spatial_x = spatial_x.view(params.batch_size, params.bptt_img, -1).transpose(0,1)

        return x, spatial_x, feature_length, positions, (None, None)
    
    def generate_batch_mm(self, m1, m2, name):
        
        params = self.params
        
        modal1_id = params.modal2id[m1]
        modal2_id = params.modal2id[m2]

        (x_w, lengths_w), x_i, spatial_i, lengths_i = self.get_batch_mm(name, m1, m2)
        
#         print(f'x_w\t\t: {x_w.shape}')
#         print(f'lengths_w\t: {lengths_w.shape}')
#         print(f'x_i\t\t: {x_i.shape}')
#         print(f'spatial_i\t: {spatial_i.shape}')
#         print(f'lengths_i\t: {lengths_i.shape}')
#         exit()
        
        x, spatial_x, lengths, positions = concat_batches_mm(x_w, lengths_w, modal1_id, 
                                                             x_i, lengths_i, modal2_id,  
                                                             spatial_i=spatial_i, params=params)

        return x, spatial_x, lengths, positions, (None, None)
    
    def save_model(self, name):
        """
        Save the model.
        """
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info('Saving models to %s ...' % path)
        data = {}
        for name in self.MODEL_NAMES:
            if self.params.multi_gpu:
                data[name] = getattr(self, name).module.state_dict()
            else:
                data[name] = getattr(self, name).state_dict()

        data['dico_id2word'] = self.data['dico'].id2word
        data['dico_word2id'] = self.data['dico'].word2id
        data['dico_counts'] = self.data['dico'].counts
        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)

    def save_checkpoint(self):
        """
        Checkpoint the experiment.
        """
        if not self.params.is_master:
            return

        data = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
        }

        for name in self.MODEL_NAMES:
            data[name] = getattr(self, name).state_dict()
            data[name + '_optimizer'] = self.optimizers[name].state_dict()

        data['dico_id2word'] = self.data['dico'].id2word
        data['dico_word2id'] = self.data['dico'].word2id
        data['dico_counts'] = self.data['dico'].counts
        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        logger.info("Saving checkpoint to %s ..." % checkpoint_path)
        torch.save(data, checkpoint_path)

    def reload_checkpoint(self):
        """
            Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        logger.warning('Reloading checkpoint from %s ...' % checkpoint_path)
        data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(self.params.local_rank))

        # reload model parameters and optimizers
        for name in self.MODEL_NAMES:
            getattr(self, name).load_state_dict(data[name])
            # getattr(self, name).load_state_dict({k[len('module.'):]: v for k, v in data[name].items()})
            self.optimizers[name].load_state_dict(data[name + '_optimizer'])

        # reload main metrics
        self.epoch = data['epoch'] + 1
        self.n_total_iter = data['n_total_iter']
        self.best_metrics = data['best_metrics']
        self.best_stopping_criterion = data['best_stopping_criterion']
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_model('periodic-%i' % self.epoch)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_model('best-%s' % metric)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (self.params.is_master or not self.stopping_criterion[0].endswith('_mt_bleu')):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                if self.params.multi_gpu and 'SLURM_JOB_ID' in os.environ:
                    os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                exit()
        self.save_checkpoint()
        self.epoch += 1

    def round_batch(self, x, lengths, positions, langs):
        """
            For float16 only.
            Sub-sample sentences in a batch, and add padding,
            so that each dimension is a multiple of 8.
        """
        params = self.params
        if not params.fp16 or len(lengths) < 8:
            return x, lengths, positions, langs, None

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[:slen, idx]
            positions = None if positions is None else positions[:slen, idx]
            langs = None if langs is None else langs[:slen, idx]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(0)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            x = torch.cat([x, torch.LongTensor(pad, bs2).fill_(params.pad_index)], 0)
            if positions is not None:
                positions = torch.cat([positions, torch.arange(pad)[:, None] + positions[-1][None] + 1], 0)
            if langs is not None:
                langs = torch.cat([langs, langs[-1][None].expand(pad, bs2)], 0)
            assert x.size() == (ml2, bs2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths, positions, langs, idx


    def mlm_step(self, lang, lambda_coeff):
        """
            Masked word prediction step.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        model = getattr(self, 'model')
        model.train()

        # generate batch / select words to predict
        x, lengths, positions, _ = self.generate_batch_lang(lang, 'pred')       
        x, y, pred_mask = self.mask_out_lang(x, lengths)

        # cuda
        x, y, pred_mask, lengths, positions = to_cuda(x, y, pred_mask, lengths, positions)
        
        # forward / loss
        tensor_i = model('fwd', x=x, lengths=lengths, positions=positions)        
        _, loss = model('predict', tensor=tensor_i, pred_mask=pred_mask, y=y)
        self.stats[('MLM-%s' % lang)].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss, 'model')

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += lengths.size(0)
        self.stats['processed_w'] += pred_mask.sum().item()
        
    def ipm_step(self, img, lambda_coeff):
        """
            Masked image prediction step.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        model = getattr(self, 'model')
        model.train()

        # generate batch / select words to predict
        x, spatial_x, lengths, positions, _ = self.generate_batch_img(img, 'pred')
        x, y, candidates, pred_mask = self.mask_out_img(x, lengths)
        
        # cuda
        x, y, spatial_x, pred_mask, candidates, lengths, positions = to_cuda(x, y, spatial_x, pred_mask, 
                                                                                   candidates, lengths, positions)
        
#         print(f'x\t\t\t: {x.shape}')
#         print(f'y\t\t\t: {y.shape}')
#         print(f'spatial_x\t\t: {spatial_x.shape}')
#         print(f'pred_mask\t\t: {pred_mask.shape}')
#         print(f'candidates_before\t: {candidates.shape}')
        
        # forward / loss
        tensor_i, candidates = model('fwd', x=x, lengths=lengths, candidates=candidates, 
                                   spatials=spatial_x, positions=positions, ipm=True) 
        
#         print(f'tensor_i\t\t: {tensor_i.shape}')
#         print(f'candidates_after\t: {candidates.shape}')
        
#         _, loss = model('predict', tensor=tensor_i, candidates=candidates, pred_mask=pred_mask, y=y, get_scores=False, ipm=True)
        
        _, loss = self.pred_ce(tensor=tensor_i, candidates=candidates, pred_mask=pred_mask, y=y)
        
        self.stats[('IPM-%s' % img)].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss, 'model')

        # number of processed sentences / words
        self.n_images += params.batch_size
        self.stats['processed_unmasked_regions'] += pred_mask.sum().item()
        self.stats['processed_images'] += params.batch_size
        
    def cmlm_step(self, m1, m2, lambda_coeff):
        """
            Masked caption-image prediction step.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        model = getattr(self, 'model')
        model.train()

        # generate batch / select words to predict
        x, spatial_x, lengths, positions, _ = self.generate_batch_mm(m1, m2, 'pred')
        x, spatial_x, lengths = self.add_cls(x, spatial_x, lengths)
        x, y_cap, y_img, _x_candidates_img, pred_mask = self.mask_out_mm(x)
        x, y_cap, y_img, spatial_x, pred_mask = to_cuda(x, y_cap, y_img, spatial_x, pred_mask)
        _x_candidates_img, lengths, positions = to_cuda(_x_candidates_img, lengths, positions)

        # forward / loss
        tensor_i, tensor_w, candidates = model('fwd', x=x, lengths=lengths, candidates=_x_candidates_img, 
                                               spatials=spatial_x, positions=positions, cmlm=True)

        _, loss_w = model('predict', tensor=tensor_w, pred_mask=pred_mask[0], y=y_cap)
        _, loss_i = self.pred_ce(tensor=tensor_i, candidates=candidates, pred_mask=pred_mask[1], y=y_img)

        loss = loss_w + loss_i
        
        self.stats[('CMLM-%s-%s' %(m1, m2))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss, 'model')

        # number of processed sentences / words
        self.n_pairs += params.batch_size
        self.stats['processed_pairs'] += params.batch_size


class SingleTrainer(Trainer):

    def __init__(self, model, data, params):

        self.MODEL_NAMES = ['model']

        # model / data / params
        self.model = model
        self.data = data
        self.params = params

        # optimizers
        self.optimizers = {'model': self.get_optimizer_fp('model')}

        super().__init__(data, params)

        
        
        
        
        
        
        
        
########################################## Codes reserved for future use ###########################################
#         if params.stopping_criterion_lang != '':
#             split = params.stopping_criterion.split(',')
#             assert len(split) == 2 and split[1].isdigit()
#             self.decrease_counts_max_lang = int(split[1])
#             self.decrease_counts_lang = 0
#             if split[0][0] == '_':
#                 self.stopping_criterion_lang = (split[0][1:], False)
#             else:
#                 self.stopping_criterion_lang = (split[0], True)
#             self.best_stopping_criterion_lang = -1e12 if self.stopping_criterion_lang[1] else 1e12
#         else:
#             self.stopping_criterion_lang = None
#             self.best_stopping_criterion_lang = None
            
#         if params.stopping_criterion_image != '':
#             split = params.stopping_criterion_image.split(',')
#             assert len(split) == 2 and split[1].isdigit()
#             self.decrease_counts_max_image = int(split[1])
#             self.decrease_counts_image = 0
#             if split[0][0] == '_':
#                 self.stopping_criterion_image = (split[0][1:], False)
#             else:
#                 self.stopping_criterion_image = (split[0], True)
#             self.best_stopping_criterion_image = -1e12 if self.stopping_criterion_image[1] else 1e12
#         else:
#             self.stopping_criterion_image = None
#             self.best_stopping_criterion_image = None
            
#         if params.stopping_criterion_cmodal != '':
#             split = params.stopping_criterion_cmodal.split(',')
#             assert len(split) == 2 and split[1].isdigit()
#             self.decrease_counts_max_cmodal = int(split[1])
#             self.decrease_counts_cmodal = 0
#             if split[0][0] == '_':
#                 self.stopping_criterion_cmodal = (split[0][1:], False)
#             else:
#                 self.stopping_criterion_cmodal = (split[0], True)
#             self.best_stopping_criterion_cmodal = -1e12 if self.stopping_criterion_cmodal[1] else 1e12
#         else:
#             self.stopping_criterion_cmodal = None
#             self.best_stopping_criterion_cmodal = None
########################################## Codes reserved for future use ###########################################