# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch

from .pretrain import load_embeddings
from .transformer import DECODER_ONLY_PARAMS, TransformerModel  # , TRANSFORMER_LAYER_PARAMS


logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt_word >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.word_sample_alpha < 1
    s = params.word_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]
    
    assert params.bptt_img >= 1
    assert 0 <= params.img_pred < 1
    assert 0 <= params.img_sample_alpha < 1
    i = params.img_mask_keep_rand.split(',')
    assert len(i) == 3
    i = [float(x) for x in i]
    assert all([0 <= x <= 1 for x in i]) and sum(i) == 1
    params.img_mask = i[0]
    params.img_keep = i[1]
    params.img_rand = i[2]

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # share input and output embeddings
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # reload pretrained embeddings
    if params.reload_emb != '':
        assert os.path.isfile(params.reload_emb)

    # reload a pretrained model
    if params.reload_model != '':
        assert os.path.isfile(params.reload_model)


def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    logger.info("Pretrained %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))


def build_model(params, dico):
    """
        Build model.
    """
    # build
    model = TransformerModel(params, dico, with_output=True)

    # reload pretrained word embeddings
    if params.reload_emb != '':
        word2id, embeddings = load_embeddings(params.reload_emb, params)
        set_pretrain_emb(model, dico, word2id, embeddings)

    # reload a pretrained model
    if params.reload_model != '':
        logger.info("Reloading model from %s ..." % params.reload_model)
        reloaded = torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(params.local_rank))['model']
        if all([k.startswith('module.') for k in reloaded.keys()]):
            reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

        model.load_state_dict(reloaded)

    logger.debug("Model: {}".format(model))
    logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))

    return model.cuda()
