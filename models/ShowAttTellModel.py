import paddle
import paddle.nn as nn
from .CaptionModel import CaptionModel
import paddle.nn.functional as F


class Attention(nn.Layer):
    def __init__(self, d_model, att_hid_size):
        super(Attention, self).__init__()
        self.rnn_size = d_model
        self.att_hid_size = att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.shape[0] // att_feats.shape[-1]
        att = p_att_feats.reshape([-1, att_size, self.att_hid_size])

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = paddle.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.reshape([-1, self.att_hid_size])  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.reshape([-1, att_size])  # batch * att_size

        weight = F.softmax(dot, 1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.reshape([-1, att_size])
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.reshape([-1, att_size, att_feats.shape[-1]])  # batch * att_size * att_feat_size
        att_res = paddle.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res


class LSTMCore(nn.Layer):
    def __init__(self, d_model, att_hid_size):
        super(LSTMCore, self).__init__()
        self.input_encoding_size = d_model
        self.rnn_size = d_model

        # Build a LSTM
        self.i2h = nn.Linear(self.input_encoding_size + self.rnn_size, 4 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.dropout = nn.Dropout(0.5)
        self.attention = Attention(d_model, att_hid_size)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(paddle.concat([xt, att_res], 1)) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.slice([1], [0], [3 * self.rnn_size])
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.slice([1], [0], [self.rnn_size])
        forget_gate = sigmoid_chunk.slice([1], [self.rnn_size], [self.rnn_size * 2])
        out_gate = sigmoid_chunk.slice([1], [self.rnn_size * 2], [self.rnn_size * 3])

        in_transform = F.tanh(all_input_sums.slice([1], [3 * self.rnn_size], [4 * self.rnn_size]))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state


class ShowAttTellModel(CaptionModel):
    def __init__(self, opt):
        super(ShowAttTellModel, self).__init__(opt)
        del self.embed, self.fc_embed
        self.embed = nn.Embedding(self.vocab_size + 1, self.d_model)
        self.fc_embed = lambda x: x
        self.core = LSTMCore(self.d_model, self.att_hid_size)
