# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import subprocess
from collections import OrderedDict
import numpy as np
import torch

from ..utils import to_cuda, restore_segmentation, concat_batches


BLEU_SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH)


logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.data = data
        self.dico = data['dico']
        self.params = params

        # create directory to store hypotheses, and reference files for BLEU evaluation
        if self.params.is_master:
            params.hyp_path = os.path.join(params.dump_path, 'hypotheses')
            subprocess.Popen('mkdir -p %s' % params.hyp_path, shell=True).wait()
            self.create_reference_files()

    def create_reference_files(self):
        """
            Create reference files for BLEU evaluation.
        """
        params = self.params
        params.ref_paths = {}

#         for (lang1, lang2), v in self.data['para'].items():

#             assert lang1 < lang2

#             for data_set in ['valid', 'test']:

#                 # define data paths
#                 lang1_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_set))
#                 lang2_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set))

#                 # store data paths
#                 params.ref_paths[(lang2, lang1, data_set)] = lang1_path
#                 params.ref_paths[(lang1, lang2, data_set)] = lang2_path

#                 # text sentences
#                 lang1_txt = []
#                 lang2_txt = []

#                 # convert to text
#                 for (sent1, len1), (sent2, len2) in self.get_iterator(data_set, lang1, lang2):
#                     lang1_txt.extend(convert_to_text(sent1, len1, self.dico, params))
#                     lang2_txt.extend(convert_to_text(sent2, len2, self.dico, params))

#                 # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
#                 lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
#                 lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

#                 # export hypothesis
#                 with open(lang1_path, 'w', encoding='utf-8') as f:
#                     f.write('\n'.join(lang1_txt) + '\n')
#                 with open(lang2_path, 'w', encoding='utf-8') as f:
#                     f.write('\n'.join(lang2_txt) + '\n')

#                 # restore original segmentation
#                 restore_segmentation(lang1_path)
#                 restore_segmentation(lang2_path)

    def mask_out(self, x, lengths, rng):
        """
        Decide of random words to mask out.
        We specify the random generator to ensure that the test is the same at each epoch.
        """
        params = self.params
        slen, bs = x.size()

        # words to predict - be sure there is at least one word per sentence
        to_predict = rng.rand(slen, bs) <= params.word_pred
        to_predict[0] = 0
        for i in range(bs):
            to_predict[lengths[i] - 1:, i] = 0
            if not np.any(to_predict[:lengths[i] - 1, i]):
                v = rng.randint(1, lengths[i] - 1)
                to_predict[v, i] = 1
        pred_mask = torch.from_numpy(to_predict.astype(np.uint8))

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_mask = _x_real.clone().fill_(params.mask_index)
        x = x.masked_scatter(pred_mask, _x_mask)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():

            for data_set in ['valid', 'test']:

                # prediction task (evaluate perplexity and accuracy)
                for lang in params.mlm_steps:
                    self.evaluate_mlm(scores, data_set, lang)
                    
                # prediction task (evaluate cross_entropy loss and accuracy)
                for lang in params.mlm_steps:
                    self.evaluate_mlm(scores, data_set, lang)

#                 # report average metrics per language
#                 _clm_mono = [l1 for (l1, l2) in params.clm_steps if l2 is None]
#                 if len(_clm_mono) > 0:
#                     scores['%s_clm_ppl' % data_set] = np.mean([scores['%s_%s_clm_ppl' % (data_set, lang)] for lang in _clm_mono])
#                     scores['%s_clm_acc' % data_set] = np.mean([scores['%s_%s_clm_acc' % (data_set, lang)] for lang in _clm_mono])
#                 _mlm_mono = [l1 for (l1, l2) in params.mlm_steps if l2 is None]
#                 if len(_mlm_mono) > 0:
#                     scores['%s_mlm_ppl' % data_set] = np.mean([scores['%s_%s_mlm_ppl' % (data_set, lang)] for lang in _mlm_mono])
#                     scores['%s_mlm_acc' % data_set] = np.mean([scores['%s_%s_mlm_acc' % (data_set, lang)] for lang in _mlm_mono])

        return scores


    def evaluate_mlm(self, scores, data_set, lang):
        """
            Evaluate perplexity.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang in params.langs

        model = self.model
        model.eval()
        model = model.module if params.multi_gpu else model

        rng = np.random.RandomState(0)

        n_words = 0
        xe_loss = 0
        n_valid = 0

        for batch in self.get_iterator_lang(data_set, lang):

            # batch
            x, lengths = batch
            positions = None

            # words to predict
            x, y, pred_mask = self.mask_out(x, lengths, rng)

            # cuda
            x, y, pred_mask, lengths, positions, langs = to_cuda(x, y, pred_mask, lengths, positions, langs)

            # forward / loss
            tensor_i = model('fwd', x=x, lengths=lengths, positions=positions)
            word_scores, loss = model('predict', tensor=tensor_i, pred_mask=pred_mask, y=y)

            # update stats
            n_words += len(y)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_mlm_ppl' % (data_set, lang)
        acc_name = '%s_%s_mlm_acc' % (data_set, lang)
        scores[ppl_name] = np.exp(xe_loss / n_words) if n_words > 0 else 1e9
        scores[acc_name] = 100. * n_valid / n_words if n_words > 0 else 0.
        
        
    def get_iterator_lang(self, data_set, lang):
        """
            Create a new iterator for a dataset.
        """
        assert data_set in ['valid', 'test']
        assert lang in self.params.langs

        n_sentences = -1
        subsample = 1

        iterator = self.data['lang_stream'][lang][data_set].get_iterator(shuffle=False, subsample=subsample)

        for batch in iterator:
            yield batch


class SingleEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
            Build language model evaluator.
        """
        super().__init__(trainer, data, params)
        self.model = trainer.model



def convert_to_text(batch, lengths, dico, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + '0')
    assert os.path.isfile(BLEU_SCRIPT_PATH)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1
