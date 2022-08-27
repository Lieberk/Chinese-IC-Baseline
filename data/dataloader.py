import json
import h5py
import os
import numpy as np
import random

import paddle
import paddle.io as data
import numpy.random as npr


class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.
    """

    def __init__(self, db_path, ext):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(x)
        else:
            self.loader = lambda x: np.load(x)['feat']
        if db_path.endswith('.pth'):  # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = paddle.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        else:
            self.db_type = 'dir'

    def get(self, key):
        if self.db_type == 'pth':
            f_input = self.feat_file[key]
        else:
            f_input = os.path.join(self.db_path, key + self.ext)

        # load image
        feat = self.loader(f_input)
        return feat


class Dataset(data.Dataset):
    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.seq_per_img = opt.seq_per_img

        # feature related options
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_att = getattr(opt, 'use_att', True)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        if 'ix_to_word' in self.info:
            self.ix_to_word = self.info['ix_to_word']
            self.vocab_size = len(self.ix_to_word)
            print('vocab size is ', self.vocab_size)

        """
        Setting input_label_h5 to none is used when only doing generation.
        For example, when you need to test on coco test set.
        """
        if self.opt.input_label_h5 != 'none':
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
            # load in the sequence data
            seq_size = self.h5_label_file['labels'].shape
            self.label = self.h5_label_file['labels'][:]
            self.seq_length = seq_size[1]
            print('max sequence length in data is', self.seq_length)
            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        else:
            self.seq_length = 1

        self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy')
        self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz')

        self.num_images = len(self.info['images'])  # self.label_start_ix.shape[0]
        print('read %d image features' % (self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}

        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
                self.num_images_train = len(self.split_ix['train'])

            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
                self.num_images_val = len(self.split_ix['val'])
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
                self.num_images_test = len(self.split_ix['test'])

        print('assigned %d images to split train' % self.num_images_train)
        print('assigned %d images to split val' % self.num_images_val)
        print('assigned %d images to split test' % self.num_images_test)

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1  # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int64')
            for q in range(seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def collate_func(self, batch):
        seq_per_img = self.seq_per_img

        fc_batch = []
        att_batch = []
        label_batch = []

        infos = []
        gts = []

        for sample in batch:
            # fetch image
            tmp_fc, tmp_att, tmp_seq, ix = sample

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)

            tmp_label = np.zeros([seq_per_img, self.seq_length + 2], dtype='int64')
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                tmp_label[:, 1: self.seq_length + 1] = tmp_seq
            label_batch.append(tmp_label)

            # Used for reward evaluation
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                gts.append(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]].tolist())
            else:
                gts.append([])

            # record associated info as well
            info_dict = {'ix': ix, 'id': self.info['images'][ix]['id'], 'file_path': self.info['images'][ix].get('file_path', '')}
            infos.append(info_dict)

        # sort by att_feat length
        fc_batch, att_batch, label_batch, infos = \
            zip(*sorted(zip(fc_batch, att_batch, label_batch, infos), key=lambda x: 0, reverse=True))

        data = {'fc_feats': np.stack(fc_batch)}
        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype='float32')
        for i in range(len(att_batch)):
            data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i, :att_batch[i].shape[0]] = 1

        data['labels'] = np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype='float32')
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts  # all ground truth captions of each images
        data['infos'] = infos

        data = {k: paddle.to_tensor(v) if type(v) is np.ndarray else v for k, v in
                data.items()}

        return data

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        if self.use_att:
            att_feat = self.att_loader.get(str(self.info['images'][ix]['id']))
            # Reshape to K x C
            att_feat = att_feat.reshape([-1, att_feat.shape[-1]])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
        else:
            att_feat = np.zeros((0, 0), dtype='float32')
        if self.use_fc:
            fc_feat = self.fc_loader.get(str(self.info['images'][ix]['id']))
        else:
            fc_feat = np.zeros(1, dtype='float32')
        if hasattr(self, 'h5_label_file'):
            seq = self.get_captions(ix, self.seq_per_img)
        else:
            seq = None
        return fc_feat, att_feat, seq, ix

    def __len__(self):
        return self.num_images


class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.dataset = Dataset(opt)
        self.loaders = {}

    def reset_iterator(self, split):
        if split == 'train':
            batch_sampler = data.BatchSampler(sampler=MySampler(self.dataset.split_ix[split], shuffle=True),
                                              batch_size=self.batch_size,
                                              drop_last=True)
        else:
            batch_sampler = data.BatchSampler(sampler=MySampler(self.dataset.split_ix[split], shuffle=False),
                                              batch_size=self.batch_size,
                                              drop_last=True)
        self.loaders[split] = data.DataLoader(dataset=self.dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=0,
                                              collate_fn=lambda x: self.dataset.collate_func(x),
                                              )

    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    def get_vocab(self):
        return self.dataset.get_vocab()

    def get_seq_length(self):
        return self.dataset.get_seq_length()


class MySampler(data.Sampler):
    def __init__(self, index_list, shuffle):
        super().__init__()
        self.index_list = index_list
        self.shuffle = shuffle
        self._reset_iter()

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_counter == len(self._index_list):
            raise StopIteration()
        elem = self._index_list[self.iter_counter]
        self.iter_counter += 1
        return elem

    def next(self):
        return self.__next__()

    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list
        self.iter_counter = 0

    def __len__(self):
        return len(self.index_list)
