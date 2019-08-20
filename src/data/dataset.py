# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import numpy as np
import torch
import h5py
import json


logger = getLogger()


class StreamDataset(object):

    def __init__(self, sent, pos, params):
        """
            Prepare batches for data iterator.
        """
        bptt = params.bptt_word
        bs = params.batch_size
        self.eos = params.eos_index

        # checks
        assert len(pos) == (sent == self.eos).sum()
        assert len(pos) == (sent[pos[:, 1]] == self.eos).sum()

        n_tokens = len(sent)
        n_batches = math.ceil(n_tokens / (bs * bptt))
        t_size = n_batches * bptt * bs

        buffer = np.zeros(t_size, dtype=sent.dtype) + self.eos
        buffer[t_size - n_tokens:] = sent
        buffer = buffer.reshape((bs, n_batches * bptt)).T
        self.data = np.zeros((n_batches * bptt + 1, bs), dtype=sent.dtype) + self.eos
        self.data[1:] = buffer

        self.bptt = bptt
        self.n_tokens = n_tokens
        self.n_batches = n_batches
        self.n_sentences = len(pos)
        self.lengths = torch.LongTensor(bs).fill_(bptt)

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return self.n_sentences

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        if not (0 <= a < b <= self.n_batches):
            logger.warning("Invalid split values: %i %i - %i" % (a, b, self.n_batches))
            return
        assert 0 <= a < b <= self.n_batches
        logger.info("Selecting batches from %i to %i ..." % (a, b))

        # sub-select
        self.data = self.data[a * self.bptt:b * self.bptt]
        self.n_batches = b - a
        self.n_sentences = (self.data == self.eos).sum().item()

    def get_iterator(self, shuffle, subsample=1):
        """
            Return a sentences iterator.
        """
        indexes = (np.random.permutation if shuffle else range)(self.n_batches // subsample)
        for i in indexes:
            a = self.bptt * i
            b = self.bptt * (i + 1)
            yield torch.from_numpy(self.data[a:b].astype(np.int64)), self.lengths
            
class ImageDataset(object):
    """
        Prepare image batches for data iterator
    """

    def __init__(self, data_folder, params, split, transform=None):
        """
            Unlike languages where data is stored in the memory and called out when needed during the training,
            here we store the image features as hdf5 format first and then extract the required batch set during training.
            By doing so, we can speed up the loading process.
            
            :param data_folder : folder where data files are stored
            :param param       : parameters
            :param split       : split, one of 'train', 'valid', or 'test'
            :param transform   : image transform pipeline
        """
        
        self.split = split
        assert self.split in {'train', 'valid', 'test'}
        
        # Open hdf5 file where images are stored
        # check if it is image to caption
        self.hf = h5py.File(data_folder[split], 'r')
        self.image_features = self.hf['image_features']
        self.spatial_features = self.hf['spatial_features']

        bptt = params.bptt_img
        bs = params.batch_size
        
        # image_features shape = {len(images_bbox)*len(images)+len(images), 2048}
        # spatial_features shape = {len(images_bbox)*len(images)+len(images), 6}
#         print(f'image_features: {self.image_features.shape}')
#         print(f'spatial_features: {self.spatial_features.shape}')
        
        n_regions = self.image_features.shape[0]
        n_batches = math.ceil(n_regions / (bs * bptt))
        t_size = n_batches * bptt * bs
        
        self.bs = bs
        self.bptt = bptt
        self.n_regions = n_regions
        self.n_batches = n_batches
        self.n_images = self.image_features.shape[0] / params.n_bbox
        self.n_features = torch.LongTensor(bs).fill_(bptt)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

    def get_iterator(self, shuffle, subsample=1):
        indexes = (np.random.permutation if shuffle else range)(self.n_batches // subsample)
        for i in indexes:
            a = self.bptt * self.bs * i
            b = self.bptt * self.bs * (i + 1)
            if (b > self.n_regions):
                r = b - self.n_regions
                self.feature = np.append(self.image_features[a:self.n_regions], 
                                      np.zeros((r, self.image_features.shape[-1])), 
                                      axis=0)
                self.spatial = np.append(self.spatial_features[a:self.n_regions], 
                                         np.zeros((r, self.spatial_features.shape[-1])), 
                                         axis=0)
            else:
                self.feature = self.image_features[a:b]
                self.spatial = self.spatial_features[a:b]
                
            assert self.feature.shape == (self.bs * self.bptt, self.image_features.shape[-1])
            assert self.spatial.shape == (self.bs * self.bptt, self.spatial_features.shape[-1])
                
            yield torch.from_numpy(self.feature.astype(np.float32)), torch.from_numpy(self.spatial.astype(np.float32)), self.n_features
            
    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        if not (0 <= a < b <= self.n_batches):
            logger.warning("Invalid split values: %i %i - %i" % (a, b, self.n_batches))
            return
        assert 0 <= a < b <= self.n_batches
        logger.info("Selecting batches from %i to %i ..." % (a, b))

        # sub-select
        self.data = self.data[a * self.bptt:b * self.bptt]
        self.n_batches = b - a
        self.n_sentences = (self.data == self.eos).sum().item()

    def __len__(self):
        return self.n_regions

class Dataset(object):

    def __init__(self, sent, pos, params):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.sent = sent
        self.pos = pos
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        # check number of sentences
        assert len(self.pos) == (self.sent == self.eos_index).sum()

        # # remove empty sentences
        # self.remove_empty_sentences()

        # sanity checks
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos)

    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.pos) == (self.sent[self.pos[:, 1]] == eos).sum()  # check sentences indices
        # assert self.lengths.min() > 0                                     # check empty sentences

    def batch_sentences(self, sentences):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        # sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)

        sent[0] = self.eos_index
        for i, s in enumerate(sentences):
            if lengths[i] > 2:  # if sentence not empty
                sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] > 0]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] <= max_len]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        assert 0 <= a < b <= len(self.pos)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.pos = self.pos[a:b]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        # re-index
        min_pos = self.pos.min()
        max_pos = self.pos.max()
        self.pos -= min_pos
        self.sent = self.sent[min_pos:max_pos + 1]

        # sanity checks
        self.check()

    def get_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            pos = self.pos[sentence_ids]
            sent = [self.sent[a:b] for a, b in pos]
            sent = self.batch_sentences(sent)
            yield (sent, sentence_ids) if return_indices else sent

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, seed=None, return_indices=False):
        """
        Return a sentences iterator.
        """
        assert seed is None or shuffle is True and type(seed) is int
        rng = np.random.RandomState(seed)
        n_sentences = len(self.pos) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos)
        assert type(shuffle) is bool and type(group_by_size) is bool
        assert group_by_size is False or shuffle is True

        # sentence lengths
        lengths = self.lengths + 2

        # select sentences to iterate over
        if shuffle:
            indices = rng.permutation(len(self.pos))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind='mergesort')]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            rng.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])
        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)
    
class CrossModalDataset(Dataset):
    
    def __init__(self, sent, pos, tgt_path, params):
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size
        
        self.bs = params.batch_size
        
        self.sent = sent
        self.pos = pos
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        self.pos_group = np.array_split(np.arange(len(self.pos)), len(self.pos)/5) # for 1 image we have exactly 5 captions
        
        self.hf = h5py.File(tgt_path, 'r')
        self.image_features = self.hf['image_features']        # image shape: (113287, 37, 2048)
        self.spatial_features = self.hf['spatial_features']    # spatial shape: (113287, 37, 6)
        self.n_features = torch.LongTensor(self.bs).fill_(params.n_bbox+1)
        
        self.caplen = str(len(self.pos))
        self.n_regions = str(self.image_features.shape[0])
        
#         print(f'len pos: {len(self.pos)}')
#         print(f'len pos_group: {len(self.pos_group)}')
#         print(f"length: {len(self.lengths)}")
#         print(f"Max length: {max(self.lengths)}")
        
        assert len(self.pos) == (self.sent == self.eos_index).sum()
        
    def __len__(self):
        """
            Number of caption-image pair in the dataset.
        """
        return len(self.pos_group)
        
    def get_batches_iterator(self, cap_batches, img_batches, return_indices):
        """
            Return a sentences-image iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for cap_ids, img_ids in zip(cap_batches, img_batches):
            
            # handle case where total cap_ids is less than the batch_size
            if (len(cap_ids) < self.bs):
                r = self.bs - len(cap_ids)
                
                tmp_img_list = []
                
                while(True):
                    k = np.random.choice(len(self.pos_group))
                    if(k not in img_ids and k not in tmp_img_list):
                        tmp_img_list.append(k)
                    if(len(tmp_img_list) == r):
                        break
                        
                tmp_cap_list = [np.random.choice(self.pos_group[i]) for i in tmp_img_list]
                
                cap_ids = np.array(list(cap_ids) + tmp_cap_list)
                img_ids = np.array(list(img_ids) + tmp_img_list)
            
            pos = self.pos[cap_ids]
            sent = self.batch_sentences([self.sent[a:b] for a, b in pos]) # contains sentences (longvectors) and lengths
            
            tmp_sorted_ids = np.argsort(img_ids.copy(), kind="mergesort")
            tmp_sorted_imgids = img_ids[tmp_sorted_ids]
            
            assert len(pos) == len(tmp_sorted_imgids), f"sent_ids: {len(pos)}; img_ids: {len(tmp_sorted_imgids)}"
            assert len(sent[1]) == len(tmp_sorted_imgids), f"sent_ids: {len(pos)}; img_ids: {len(tmp_sorted_imgids)}"
            
            img_features = torch.from_numpy((self.image_features[list(tmp_sorted_imgids)])[tmp_sorted_ids].astype(np.float32))
            img_spatials = torch.from_numpy((self.spatial_features[list(tmp_sorted_imgids)])[tmp_sorted_ids].astype(np.float32))
            
            img_features = img_features.view(-1,img_features.shape[-1]).transpose(0,1)
            img_spatials = img_spatials.view(-1,img_spatials.shape[-1]).transpose(0,1)
            
            yield (sent, img_features, img_spatials, self.n_features, sentence_ids) if return_indices else (sent, img_features, img_spatials, self.n_features)
        
    def get_iterator(self, shuffle, group_by_size=False, return_indices=False):
        """
            Return a sentences and image iterator.
        """

        assert len(self.pos_group) == self.image_features.shape[0] == self.spatial_features.shape[0]
        
        # caption lengths
        lengths = self.lengths
    
        # select sentence-image pair to iterate over
        if shuffle:
            img_indices = np.random.permutation(len(self.pos_group))
        else:
            img_indices = np.arange(len(self.pos_group))
            
        cap_indices = np.array([np.random.choice(self.pos_group[i]) for i in img_indices])
        
        # sanity check
        assert len(cap_indices) == len(img_indices)  
        
        # group sentences by lengths
        if group_by_size:
            tmp_indices_sort_by_len = np.argsort(lengths[cap_indices], kind='mergesort')
            cap_indices = cap_indices[tmp_indices_sort_by_len]
            img_indices = img_indices[tmp_indices_sort_by_len]

        cap_batches = np.array(np.array_split(cap_indices, math.ceil(len(cap_indices) * 1. / self.batch_size)))
        img_batches = np.array(np.array_split(img_indices, math.ceil(len(img_indices) * 1. / self.batch_size)))
        
        # optionally shuffle batches
        if shuffle:
            p = np.random.permutation(len(cap_batches))
            cap_batches = list(cap_batches[p])
            img_batches = list(img_batches[p])

        # sanity check
        assert len(self.pos_group) == sum([len(x) for x in cap_batches]) == sum([len(x) for x in img_batches])

        # return the iterator
        return self.get_batches_iterator(cap_batches, img_batches, return_indices)

    
    
    
    
    
    
# class ParallelDataset(Dataset):

#     def __init__(self, sent1, pos1, sent2, pos2, params):

#         self.eos_index = params.eos_index
#         self.pad_index = params.pad_index
#         self.batch_size = params.batch_size
#         self.tokens_per_batch = params.tokens_per_batch
#         self.max_batch_size = params.max_batch_size

#         self.sent1 = sent1
#         self.sent2 = sent2
#         self.pos1 = pos1
#         self.pos2 = pos2
#         self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
#         self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

#         # check number of sentences
#         assert len(self.pos1) == (self.sent1 == self.eos_index).sum()
#         assert len(self.pos2) == (self.sent2 == self.eos_index).sum()

#         # remove empty sentences
#         self.remove_empty_sentences()

#         # sanity checks
#         self.check()

#     def __len__(self):
#         """
#         Number of sentences in the dataset.
#         """
#         return len(self.pos1)

#     def check(self):
#         """
#         Sanity checks.
#         """
#         eos = self.eos_index
#         assert len(self.pos1) == len(self.pos2) > 0                          # check number of sentences
#         assert len(self.pos1) == (self.sent1[self.pos1[:, 1]] == eos).sum()  # check sentences indices
#         assert len(self.pos2) == (self.sent2[self.pos2[:, 1]] == eos).sum()  # check sentences indices
#         assert eos <= self.sent1.min() < self.sent1.max()                    # check dictionary indices
#         assert eos <= self.sent2.min() < self.sent2.max()                    # check dictionary indices
#         assert self.lengths1.min() > 0                                       # check empty sentences
#         assert self.lengths2.min() > 0                                       # check empty sentences

#     def remove_empty_sentences(self):
#         """
#         Remove empty sentences.
#         """
#         init_size = len(self.pos1)
#         indices = np.arange(len(self.pos1))
#         indices = indices[self.lengths1[indices] > 0]
#         indices = indices[self.lengths2[indices] > 0]
#         self.pos1 = self.pos1[indices]
#         self.pos2 = self.pos2[indices]
#         self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
#         self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
#         logger.info("Removed %i empty sentences." % (init_size - len(indices)))
#         self.check()

#     def remove_long_sentences(self, max_len):
#         """
#         Remove sentences exceeding a certain length.
#         """
#         assert max_len >= 0
#         if max_len == 0:
#             return
#         init_size = len(self.pos1)
#         indices = np.arange(len(self.pos1))
#         indices = indices[self.lengths1[indices] <= max_len]
#         indices = indices[self.lengths2[indices] <= max_len]
#         self.pos1 = self.pos1[indices]
#         self.pos2 = self.pos2[indices]
#         self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
#         self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
#         logger.info("Removed %i too long sentences." % (init_size - len(indices)))
#         self.check()

#     def select_data(self, a, b):
#         """
#         Only select a subset of the dataset.
#         """
#         assert 0 <= a < b <= len(self.pos1)
#         logger.info("Selecting sentences from %i to %i ..." % (a, b))

#         # sub-select
#         self.pos1 = self.pos1[a:b]
#         self.pos2 = self.pos2[a:b]
#         self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
#         self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

#         # re-index
#         min_pos1 = self.pos1.min()
#         max_pos1 = self.pos1.max()
#         min_pos2 = self.pos2.min()
#         max_pos2 = self.pos2.max()
#         self.pos1 -= min_pos1
#         self.pos2 -= min_pos2
#         self.sent1 = self.sent1[min_pos1:max_pos1 + 1]
#         self.sent2 = self.sent2[min_pos2:max_pos2 + 1]

#         # sanity checks
#         self.check()

#     def get_batches_iterator(self, batches, return_indices):
#         """
#         Return a sentences iterator, given the associated sentence batches.
#         """
#         assert type(return_indices) is bool

#         for sentence_ids in batches:
#             if 0 < self.max_batch_size < len(sentence_ids):
#                 np.random.shuffle(sentence_ids)
#                 sentence_ids = sentence_ids[:self.max_batch_size]
#             pos1 = self.pos1[sentence_ids]
#             pos2 = self.pos2[sentence_ids]
#             sent1 = self.batch_sentences([self.sent1[a:b] for a, b in pos1])
#             sent2 = self.batch_sentences([self.sent2[a:b] for a, b in pos2])
#             yield (sent1, sent2, sentence_ids) if return_indices else (sent1, sent2)

#     def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, return_indices=False):
#         """
#         Return a sentences iterator.
#         """
#         n_sentences = len(self.pos1) if n_sentences == -1 else n_sentences
#         assert 0 < n_sentences <= len(self.pos1)
#         assert type(shuffle) is bool and type(group_by_size) is bool

#         # sentence lengths
#         lengths = self.lengths1 + self.lengths2 + 4

#         # select sentences to iterate over
#         if shuffle:
#             indices = np.random.permutation(len(self.pos1))[:n_sentences]
#         else:
#             indices = np.arange(n_sentences)

#         # group sentences by lengths
#         if group_by_size:
#             indices = indices[np.argsort(lengths[indices], kind='mergesort')]

#         # create batches - either have a fixed number of sentences, or a similar number of tokens
#         if self.tokens_per_batch == -1:
#             batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
#         else:
#             batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
#             _, bounds = np.unique(batch_ids, return_index=True)
#             batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
#             if bounds[-1] < len(indices):
#                 batches.append(indices[bounds[-1]:])

#         # optionally shuffle batches
#         if shuffle:
#             np.random.shuffle(batches)

#         # sanity checks
#         assert n_sentences == sum([len(x) for x in batches])
#         assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])
#         # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

#         # return the iterator
#         return self.get_batches_iterator(batches, return_indices)
