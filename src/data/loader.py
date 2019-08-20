# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import numpy as np
import torch

from .dataset import Dataset, StreamDataset, ImageDataset, CrossModalDataset
from .dictionary import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD, CLS_WORD

logger = getLogger()


def process_binarized(data, params):
    """
        Process a binarized dataset and log main statistics.
    """
    dico = data['dico']
    assert ((data['sentences'].dtype == np.uint16) and (len(dico) < 1 << 16) or
            (data['sentences'].dtype == np.int32) and (1 << 16 <= len(dico) < 1 << 31))
    logger.info("%i words (%i unique) in %i sentences. %i unknown words (%i unique) covering %.2f%% of the data." % (
        len(data['sentences']) - len(data['positions']),
        len(dico), len(data['positions']),
        sum(data['unk_words'].values()), len(data['unk_words']),
        100. * sum(data['unk_words'].values()) / (len(data['sentences']) - len(data['positions']))
    ))
    
    if params.max_vocab != -1:
        assert params.max_vocab > 0
        logger.info("Selecting %i most frequent words ..." % params.max_vocab)
        dico.max_vocab(params.max_vocab)
        data['sentences'][data['sentences'] >= params.max_vocab] = dico.index(UNK_WORD)
        unk_count = (data['sentences'] == dico.index(UNK_WORD)).sum()
        logger.info("Now %i unknown words covering %.2f%% of the data."
                    % (unk_count, 100. * unk_count / (len(data['sentences']) - len(data['positions']))))
        
    if params.min_count > 0:
        logger.info("Selecting words with >= %i occurrences ..." % params.min_count)
        dico.min_count(params.min_count)
        data['sentences'][data['sentences'] >= len(dico)] = dico.index(UNK_WORD)
        unk_count = (data['sentences'] == dico.index(UNK_WORD)).sum()
        logger.info("Now %i unknown words covering %.2f%% of the data."
                    % (unk_count, 100. * unk_count / (len(data['sentences']) - len(data['positions']))))
        
    if (data['sentences'].dtype == np.int32) and (len(dico) < 1 << 16):
        logger.info("Less than 65536 words. Moving data from int32 to uint16 ...")
        data['sentences'] = data['sentences'].astype(np.uint16)
        
    return data


def load_binarized(path, params):
    """
        Load a binarized dataset.
    """
    
    assert path.endswith('.pth')
    if params.debug_train:
        path = path.replace('train', 'valid')
    if getattr(params, 'multi_gpu', False):
        split_path = '%s.%i.pth' % (path[:-4], params.local_rank)
        if os.path.isfile(split_path):
            assert params.split_data is False
            path = split_path
    assert os.path.isfile(path), path
    logger.info("Loading data from %s ..." % path)
    data = torch.load(path)
    
    data = process_binarized(data, params)
    return data


def set_dico_parameters(params, data, dico):
    """
        Update dictionary parameters.
    """
    if 'dico' in data:
        assert data['dico'] == dico
    else:
        data['dico'] = dico

    n_words = len(dico)
    bos_index = dico.index(BOS_WORD)
    eos_index = dico.index(EOS_WORD)
    pad_index = dico.index(PAD_WORD)
    unk_index = dico.index(UNK_WORD)
    mask_index = dico.index(MASK_WORD)
    cls_word_index = dico.index(CLS_WORD)
    
    img_pad_index = 0
    img_mask_index = 1
    
    if hasattr(params, 'bos_index'):
        assert params.n_words == n_words
        assert params.bos_index == bos_index
        assert params.eos_index == eos_index
        assert params.pad_index == pad_index
        assert params.unk_index == unk_index
        assert params.mask_index == mask_index
        assert params.cls_word_index == cls_word_index
    else:
        params.n_words = n_words
        params.bos_index = bos_index
        params.eos_index = eos_index
        params.pad_index = pad_index
        params.unk_index = unk_index
        params.mask_index = mask_index
        params.cls_word_index = cls_word_index 
        
    params.img_pad_index = img_pad_index
    params.img_mask_index = img_mask_index

def load_lang_data(params, data):
    """
        Load language data.
    """
    data['lang'] = {}
    data['lang_stream'] = {}

    for lang in params.lang_dataset.keys():

        logger.info('============ Language data (%s)' % lang)

        assert lang in params.langs and lang not in data['lang']
        data['lang'][lang] = {}
        data['lang_stream'][lang] = {}

        for splt in ['train', 'valid', 'test']:

            # no need to load training data for evaluation
            if splt == 'train' and params.eval_only:
                continue

            # load data / update dictionary parameters / update data
            lang_data = load_binarized(params.lang_dataset[lang][splt], params)
            set_dico_parameters(params, data, lang_data['dico'])

            # create stream dataset
            data['lang_stream'][lang][splt] = StreamDataset(lang_data['sentences'], lang_data['positions'], params)

#             # if there are several processes on the same machine, we can split the dataset
#             if splt == 'train' and params.split_data and 1 < params.n_gpu_per_node <= data['lang_stream'][lang][splt].n_batches:
#                 n_batches = data['lang_stream'][lang][splt].n_batches // params.n_gpu_per_node
#                 a = n_batches * params.local_rank
#                 b = n_batches * params.local_rank + n_batches
#                 data['lang_stream'][lang][splt].select_data(a, b)

            logger.info("")

    logger.info("")
    

def load_image_data(params, data):
    """
        Load monolingual data.
    """
    data['image_stream'] = {}

    for image in params.img_dataset.keys():

        logger.info('============ Image data (%s)' % image)

        assert image in params.imgs
        data['image_stream'][image] = {}

        for splt in ['train', 'valid', 'test']:

            # no need to load training data for evaluation
            if splt == 'train' and params.eval_only:
                continue
            
            # create stream dataset
            data['image_stream'][image][splt] = ImageDataset(params.img_dataset[image], params, splt)
            
#             # if there are several processes on the same machine, we can split the dataset
#             if splt == 'train' and params.split_data and 1 < params.n_gpu_per_node <= data['image_stream'][lang][splt].n_batches:
#                 n_batches = data['image_stream'][lang][splt].n_batches // params.n_gpu_per_node
#                 a = n_batches * params.local_rank
#                 b = n_batches * params.local_rank + n_batches
#                 data['image_stream'][lang][splt].select_data(a, b)
            
#             n_images = data['image_stream'][image][splt].image_features.shape[0] / 37
#             n_regions = data['image_stream'][image][splt].image_features.shape[0]
            
#             logger.info("Loading data from %s ..." % params.img_features[image][splt])
#             logger.info("%i region features in %i images." % (n_regions, n_images))

    logger.info("")
            
    
def load_cross_modal_data(params, data):
    """
        Load cross modal data.
    """
    data['cross_modal'] = {}

    for src, tgt in params.crossmodal_dataset:

        logger.info('============ Cross modal data (%s-%s)' % (src, tgt))
        
        assert (src, tgt) not in data['cross_modal']
        data['cross_modal'][(src, tgt)] = {}
        
        # for splt in ['train','valid','test']:
        for splt in ['train', 'valid', 'test']:

            # no need to load training data for evaluation
            if splt == 'train' and params.eval_only:
                continue

            # load binarized datasets
            src_path = params.crossmodal_dataset[(src, tgt)][src][splt]
            tgt_path = params.crossmodal_dataset[(src, tgt)][tgt][splt]

            assert src=='cap' and tgt=='img'
            src_data = load_binarized(src_path, params)

            # update dictionary parameters
            set_dico_parameters(params, data, src_data['dico'])

            # create CrossModalDataset
            cross_dataset = CrossModalDataset(
                src_data['sentences'], src_data['positions'],
                tgt_path,
                params
            )

            # for validation and test set, enumerate sentence per sentence
            if splt != 'train':
                cross_dataset.tokens_per_batch = -1

            data['cross_modal'][(src, tgt)][splt] = cross_dataset
            logger.info("")

    logger.info("")
    
    
# def load_para_data(params, data):
#     """
#     Load parallel data.
#     """
#     data['para'] = {}

#     required_para_train = set(params.clm_steps + params.mlm_steps + params.pc_steps + params.mt_steps)

#     for src, tgt in params.para_dataset.keys():

#         logger.info('============ Parallel data (%s-%s)' % (src, tgt))

#         assert (src, tgt) not in data['para']
#         data['para'][(src, tgt)] = {}

#         for splt in ['train', 'valid', 'test']:

#             # no need to load training data for evaluation
#             if splt == 'train' and params.eval_only:
#                 continue

#             # for back-translation, we can't load training data
#             if splt == 'train' and (src, tgt) not in required_para_train and (tgt, src) not in required_para_train:
#                 continue

#             # load binarized datasets
#             src_path, tgt_path = params.para_dataset[(src, tgt)][splt]
#             src_data = load_binarized(src_path, params)
#             tgt_data = load_binarized(tgt_path, params)

#             # update dictionary parameters
#             set_dico_parameters(params, data, src_data['dico'])
#             set_dico_parameters(params, data, tgt_data['dico'])

#             # create ParallelDataset
#             dataset = ParallelDataset(
#                 src_data['sentences'], src_data['positions'],
#                 tgt_data['sentences'], tgt_data['positions'],
#                 params
#             )

#             # remove empty and too long sentences
#             if splt == 'train':
#                 dataset.remove_empty_sentences()
#                 dataset.remove_long_sentences(params.max_len)

#             # for validation and test set, enumerate sentence per sentence
#             if splt != 'train':
#                 dataset.tokens_per_batch = -1

#             # if there are several processes on the same machine, we can split the dataset
#             if splt == 'train' and params.n_gpu_per_node > 1 and params.split_data:
#                 n_sent = len(dataset) // params.n_gpu_per_node
#                 a = n_sent * params.local_rank
#                 b = n_sent * params.local_rank + n_sent
#                 dataset.select_data(a, b)

#             data['para'][(src, tgt)][splt] = dataset
#             logger.info("")

#     logger.info("")


# def load_caption_data(params, data):
#     """
#         Load caption data.
#     """
#     data['caption_stream'] = {}

#     for dataset in params.img_captions.keys():

#         logger.info('============ Dataset (%s) caption' % dataset)

#         assert dataset in params.imageset
#         data['caption_stream'][dataset] = {}

#         for splt in ['train']:

#             # no need to load training data for evaluation
#             if splt == 'train' and params.eval_only:
#                 continue

#             # load data / update dictionary parameters / update data
#             caption_data = load_binarized(params.img_captions[dataset][splt], params)  
            
#             # set_dico_parameters(params, data, caption_data['dico'])
            
#             # create stream dataset
#             data['caption_stream'][dataset][splt] = StreamDataset(caption_data['sentences'], caption_data['positions'], params)
            
#             logger.info("")

#     logger.info("")
    

def check_data_params(params):
    """
        Check datasets parameters.
    """
    
    # data path
    assert os.path.isdir(params.data_path), params.data_path
    
    # check languages
    params.langs = params.lngs.split(',')
    assert all(lang in params.mlm_steps.split(',') for lang in params.langs)
    
    # check images
    params.images = params.imgs.split(',')
    assert all(image in params.ipm_steps.split(',') for image in params.images)
        
    # check cross-modal
    params.crossmodal = params.cmodal.split('-')
    assert all(modal in {'cap','img'} for modal in params.crossmodal)
    params.id2modal = {k: v for k, v in enumerate(sorted(params.crossmodal))}
    params.modal2id = {k: v for v, k in params.id2modal.items()}
    params.n_modality = len(params.crossmodal)

    # MLM steps (Masked language pretraining)
    params.mlm_steps = [s for s in params.mlm_steps.split(',')]
    
    # IPM steps (Image pretraining)
    params.ipm_steps = [s for s in params.ipm_steps.split(',')]
    
    # CMLM steps
    cmlm_steps = [s.split('-') for s in params.cmlm_steps.split(',') if len(s) > 0]
    params.cmlm_steps = [(s[0], None) if len(s) == 1 else tuple(s) for s in cmlm_steps]
    assert all([(m1 in params.crossmodal) and (m2 in params.crossmodal or m2 is None) for m1, m2 in params.cmlm_steps])
    assert len(params.cmlm_steps) == len(set(params.cmlm_steps))

#     # parallel classification steps
#     params.pc_steps = [tuple(s.split('-')) for s in params.pc_steps.split(',') if len(s) > 0]
#     assert all([len(x) == 2 for x in params.pc_steps])
#     assert all([l1 in params.langs and l2 in params.langs for l1, l2 in params.pc_steps])
#     assert all([l1 != l2 for l1, l2 in params.pc_steps])
#     assert len(params.pc_steps) == len(set(params.pc_steps))

#     # machine translation steps
#     params.mt_steps = [tuple(s.split('-')) for s in params.mt_steps.split(',') if len(s) > 0]
#     assert all([len(x) == 2 for x in params.mt_steps])
#     assert all([l1 in params.langs and l2 in params.langs for l1, l2 in params.mt_steps])
#     assert all([l1 != l2 for l1, l2 in params.mt_steps])
#     assert len(params.mt_steps) == len(set(params.mt_steps))
#     assert len(params.mt_steps) == 0 or not params.encoder_only

#     # denoising auto-encoder steps
#     params.ae_steps = [s for s in params.ae_steps.split(',') if len(s) > 0]
#     assert all([lang in params.langs for lang in params.ae_steps])
#     assert len(params.ae_steps) == len(set(params.ae_steps))
#     assert len(params.ae_steps) == 0 or not params.encoder_only

#     # back-translation steps
#     params.bt_steps = [tuple(s.split('-')) for s in params.bt_steps.split(',') if len(s) > 0]
#     assert all([len(x) == 3 for x in params.bt_steps])
#     assert all([l1 in params.langs and l2 in params.langs and l3 in params.langs for l1, l2, l3 in params.bt_steps])
#     assert all([l1 == l3 and l1 != l2 for l1, l2, l3 in params.bt_steps])
#     assert len(params.bt_steps) == len(set(params.bt_steps))
#     assert len(params.bt_steps) == 0 or not params.encoder_only
#     params.bt_src_langs = [l1 for l1, _, _ in params.bt_steps]

    # check language dataset
    params.lang_dataset = {
        lang: {
            splt: os.path.join(params.data_path, lang ,f'en.{splt}.pth')
            for splt in ['train', 'valid', 'test']
        } for lang in params.langs
    }
    assert all([all([os.path.isfile(p) for p in paths.values()]) for paths in params.lang_dataset.values()])
    
    # check image datasets
    params.img_dataset = {
        img: {
            splt: os.path.join(params.data_path, img, f'{splt}.hdf5')
            for splt in ['train', 'valid', 'test']
        } for img in params.images
    }
    assert all([all([os.path.isfile(p) for p in paths.values()]) for paths in params.img_dataset.values()])
    
    # check image captions
#     params.img_captions = {
#         imgs: {
#             splt: os.path.join(params.img_path, '%s/processed_cap/%s.cap.en.pth' % (imgs, splt))
#             for splt in ['train', 'valid', 'test']
#         }for imgs in params.imageset
#     }
#     assert all([all([os.path.isfile(p) for p in paths.values()]) for paths in params.img_captions.values()])    
    
#     # check image feature and caption length
#     params.img_features = {
#         imgs: {
#             splt: {
#                 feature.split('.')[0]: os.path.join(params.img_path, '%s/%s_%s' % (imgs, splt, feature))
#                 for feature in ['caplens.json', 'captions.json', 'features.hdf5']
#             } for splt in ['train', 'valid', 'test']
#         } for imgs in params.imageset
#     }
#     assert all([all([all([os.path.isfile(p) for p in paths.values()]) for paths in splt.values()]) for splt in params.img_features.values()])
    
    # check cross-modal datasets
    assert all([src in {'cap'} and tgt in {'img'} for src,tgt in params.cmlm_steps])
    params.crossmodal_dataset = {
        (src, tgt): {
            src: {
                splt: os.path.join(params.data_path, src, f'cap.{splt}.pth')  
                for splt in ['train', 'valid', 'test']
            },
            tgt: {
                splt: os.path.join(params.data_path, tgt, f'{splt}_features.hdf5')
                for splt in ['train', 'valid', 'test']
            }
        }for src,tgt in params.cmlm_steps if src is not None and tgt is not None
    }
    assert all([all([os.path.isfile(p) for p in paths.values()]) for paths in params.img_dataset.values()])

#     # check parallel datasets
#     required_para_train = set(params.clm_steps + params.mlm_steps + params.pc_steps + params.mt_steps)
#     required_para = required_para_train | set([(l2, l3) for _, l2, l3 in params.bt_steps])
#     params.para_dataset = {
#         (src, tgt): {
#             splt: (os.path.join(params.data_path, '%s.%s-%s.%s.pth' % (splt, src, tgt, src)),
#                    os.path.join(params.data_path, '%s.%s-%s.%s.pth' % (splt, src, tgt, tgt)))
#             for splt in ['train', 'valid', 'test']
#             if splt != 'train' or (src, tgt) in required_para_train or (tgt, src) in required_para_train
#         } for src in params.langs for tgt in params.langs
#         if src < tgt and ((src, tgt) in required_para or (tgt, src) in required_para)
#     }
#     assert all([all([os.path.isfile(p1) and os.path.isfile(p2) for p1, p2 in paths.values()]) for paths in params.para_dataset.values()])

#     # check that we can evaluate on BLEU
#     assert params.eval_bleu is False or len(params.mt_steps + params.bt_steps) > 0


def load_data(params):
    
    data = {}

    # language datasets
    load_lang_data(params, data)
    
    # image datasets
    load_image_data(params, data)
    
    # cross modal datasets
    load_cross_modal_data(params, data)

    # Language data summary
    logger.info('============ Data summary')
    for lang, v in data['lang_stream'].items():
        for data_set in v.keys():
            logger.info('{: <18} - {: >5} - {: >12}:{: >10}'.format('Language data', data_set, lang, len(v[data_set])))

    logger.info('==========================')
            
    # Image data summary
    for image, v in data['image_stream'].items():
        for data_set in v.keys():
            logger.info('{: <18} - {: >5} - {: >12}:{: >10} regions'.format('Image data', data_set, image, len(v[data_set])))
            
    logger.info('==========================')
    
    # cross modal data summary
    for (src, tgt), v in data['cross_modal'].items():
        for data_set in v.keys():
            logger.info('{: <18} - {: >5} - {: >12}:{: >10}'.format('Cross-modal data', data_set, 'total_pair', len(v[data_set])))
            logger.info('{: <18} - {: >5} - {: >12}:{: >10}'.format('Cross-modal data', data_set, '%s' % (src), v[data_set].caplen))
            logger.info('{: <18} - {: >5} - {: >12}:{: >10}'.format('Cross-modal data', data_set, '%s' % (tgt), v[data_set].n_regions))
            logger.info('============')

    logger.info("")
    return data
