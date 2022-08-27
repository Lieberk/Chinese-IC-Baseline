import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import misc.utils as utils
import numpy as np
from models.BeamSearch import BeamSearch


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        att_feats = module(att_feats)
        scores_mask = paddle.full(shape=att_masks.shape, dtype=att_masks.dtype, fill_value=1e-9)
        scores = paddle.where(paddle.broadcast_to(att_masks, shape=scores_mask.shape) != 0, att_masks, scores_mask)
        att_feats *= paddle.unsqueeze(scores, axis=[-1])
        return att_feats
    else:
        return module(att_feats)


class CaptionModel(BeamSearch):
    def __init__(self, opt, d_model=1024, feat_size=2048, att_hid_size=512):
        super(CaptionModel, self).__init__()

        self.vocab_size = opt.vocab_size
        self.d_model = d_model
        self.seq_length = getattr(opt, 'max_length', 20) or opt.seq_length
        self.fc_feat_size = feat_size
        self.att_feat_size = feat_size
        self.att_hid_size = att_hid_size
        self.num_layers = 1
        self.ss_prob = 0.0

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.d_model),
                                   nn.ReLU(),
                                   nn.Dropout(0.5))

        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.d_model),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.d_model),
                                       nn.ReLU(),
                                       nn.Dropout(0.5))

        self.logit = nn.Linear(self.d_model, self.vocab_size + 1)
        self.ctx2att = nn.Linear(self.d_model, self.att_hid_size)

    def init_hidden(self, batch_size):
        return (paddle.zeros([self.num_layers, batch_size, self.d_model]),
                paddle.zeros([self.num_layers, batch_size, self.d_model]))

    def prepare_feature(self, fc_feats, att_feats, att_masks):
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        p_att_feats = self.ctx2att(att_feats)
        return fc_feats, att_feats, p_att_feats, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.shape[0]
        seq_per_img = seq.shape[0] // batch_size

        state = self.init_hidden(batch_size * seq_per_img)

        outputs = []
        if seq_per_img > 1:
            fc_feats, att_feats, att_masks = utils.repeat_tensors(seq_per_img, [fc_feats, att_feats, att_masks])
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.prepare_feature(fc_feats, att_feats, att_masks)

        for i in range(seq.shape[1] - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                sample_prob = paddle.uniform([batch_size * seq_per_img], min=0.0, max=1.0)
                sample_mask = (sample_prob < self.ss_prob).astype('int64')
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero()
                    it = seq[:, i].clone()
                    prob_prev = paddle.exp(outputs[i - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                    prob_tensor = paddle.multinomial(prob_prev, 1).index_select(sample_ind).squeeze(-1)
                    for k, ind in enumerate(sample_ind):
                        it[ind] = prob_tensor[k]
            else:
                it = seq[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            outputs.append(output)

        outputs = paddle.stack(outputs, axis=1)
        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        xt = self.embed(it)
        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), 1)

        return logprobs, state

    def _beam_search(self, fc_feats, att_feats, att_masks=None, beam_size=3):

        batch_size = att_feats.shape[0]
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.prepare_feature(fc_feats, att_feats, att_masks)
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the ' \
                                                 'road. can be dealt with in future if needed '
        seqs = paddle.zeros([batch_size, self.seq_length], dtype='int64')
        seqLogprobs = paddle.zeros([self.seq_length, batch_size])
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k + 1].expand([beam_size, p_fc_feats.shape[1]])
            tmp_att_feats = p_att_feats[k:k + 1].expand([beam_size, p_att_feats.shape[-2], p_att_feats.shape[-1]])
            tmp_p_att_feats = pp_att_feats[k:k + 1].expand([beam_size, pp_att_feats.shape[-2],
                                                            pp_att_feats.shape[-1]])
            tmp_att_masks = p_att_masks[k:k + 1].expand([
                beam_size, p_att_masks.shape[-1]]) if att_masks is not None else None
            tokens = self.beam_search(beam_size, state, tmp_fc_feats,
                                      tmp_att_feats, tmp_p_att_feats, tmp_att_masks)
            seqs[k, :len(tokens)] = paddle.to_tensor(tokens)
        return seqs, seqLogprobs

    def _sample(self, fc_feats, att_feats, att_masks=None, opt=None):

        temperature = opt.get('temperature', 1.0)
        sample_n = opt.get('sample_n', 1)
        beam_size = opt.get('beam_size', 1)
        sample_method = opt.get('sample_method', 'greedy')

        if beam_size > 1:
            return self._beam_search(fc_feats, att_feats, att_masks, beam_size)

        batch_size = fc_feats.shape[0]
        state = self.init_hidden(batch_size * sample_n)

        if sample_n > 1:
            fc_feats, att_feats, att_masks = utils.repeat_tensors(sample_n, [fc_feats, att_feats, att_masks])
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.prepare_feature(fc_feats, att_feats, att_masks)

        seq = paddle.zeros([batch_size * sample_n, self.seq_length], dtype='int64')
        seqLogprobs = []
        it = paddle.zeros([batch_size * sample_n], dtype='int64')

        for t in range(self.seq_length + 1):

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            unfilg = (it > 0).cast('int64')
            if t == 0:
                unfinished = unfilg
            else:
                unfinished = unfinished * unfilg

            seq[:, t] = it
            seqLogprobs.append(sampleLogprobs)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        seqLogprobs = paddle.stack(seqLogprobs, axis=1)
        sl = seqLogprobs.shape[-1]
        if sl != self.seq_length:
            seqL_zeros = paddle.zeros([batch_size * sample_n, self.seq_length - sl])
            seqLogprobs = paddle.concat([seqLogprobs, seqL_zeros], 1)

        return seq, seqLogprobs