import models
import os
import json
import argparse
import numpy as np
import paddle
import skimage.io
from paddle.vision.transforms import Normalize, CenterCrop, Resize, Compose
normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = Compose([Resize(size=256), CenterCrop(224)])
from resnext.resnext_utils import MyResnext
from resnext import resnext101_wsl


def feature_extract(args):
    net = resnext101_wsl.ResNeXt101_32x48d_wsl(pretrained=True)
    my_resnext = MyResnext(net)
    my_resnext.eval()

    I = skimage.io.imread(os.path.join(args.img_root, args.img_name))
    if len(I.shape) == 2:
        I = I[:, :, np.newaxis]
        I = np.concatenate((I, I, I), axis=2)

    I = transform(I).astype('float32') / 255.0
    I = normalize(I.transpose([2, 0, 1]))
    I = paddle.to_tensor(I)
    with paddle.no_grad():
        tmp_fc, tmp_att = my_resnext(I)
    tmp_att = tmp_att.reshape([-1, tmp_att.shape[-1]])

    return tmp_fc, tmp_att


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Captioning Model')
    parser.add_argument('--dataset', type=str, default="flickr8k")
    parser.add_argument('--input_json', default='filelists/flickr8k-cn/dataset_flickr8k_cn.json')
    parser.add_argument('--img_root', default='images', type=str)
    parser.add_argument('--img_name', default='27782020_4dab210360.jpg', type=str)
    parser.add_argument('--model_name', type=str, default="vatt")
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--word_count_threshold', default=0, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    count_thr = args.word_count_threshold
    imgs = json.load(open(args.input_json, 'r'))
    imgs = imgs['images']
    counts = {}
    raw_capitons = []
    raw_flag = 0

    for img in imgs:
        if img['filename'] == args.img_name:
            raw_flag = 1
        for sent in img['sentences']:
            if raw_flag:
                raw_capitons.append(sent['raw'])
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1
        raw_flag = 0
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    if bad_count > 0:
        vocab.append('UNK')
    itow = {str(i + 1): w for i, w in enumerate(vocab)}

    args.vocab_size = len(itow)
    model = models.setup(args)
    data = paddle.load('saved_models/%s_best.pdparams' % args.model_name)
    model.load_dict(data['state_dict'])

    fc_feats, att_feats = feature_extract(args)
    fc_feats = paddle.to_tensor(fc_feats.unsqueeze(0))
    att_feats = paddle.to_tensor(att_feats.unsqueeze(0))
    att_masks = paddle.ones(att_feats.shape[:2], dtype='float32')

    seqs, _ = model(fc_feats, att_feats, att_masks,
                        opt={'beam_size': args.beam_size}, mode='sample')
    seqs = seqs.numpy()
    prediction = ''
    for j in range(seqs.shape[-1]):
        ix = seqs[0, j]
        if ix > 0:
            prediction = prediction + itow[str(ix)]
        else:
            break

    if len(raw_capitons) != 0:
        print("参照描述：")
        for raw_capiton in raw_capitons:
            print(raw_capiton)
    else:
        print("无参照描述")

    print("预测描述：", prediction)







