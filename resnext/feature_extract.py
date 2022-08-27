import os
import argparse
import numpy as np
import paddle
from paddle.vision.transforms import Normalize, CenterCrop, Resize, Compose
import json
import skimage.io

normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = Compose([Resize(size=224), CenterCrop(224)])

from resnext_utils import MyResnext
import resnext101_wsl

split_ix = ['train', 'valid', 'test']


def main(params):
    net = resnext101_wsl.ResNeXt101_32x48d_wsl(pretrained=True)
    my_resnext = MyResnext(net)
    my_resnext.eval()

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    N = len(imgs)

    dir_fc = params['output_dir'] + '_fc_rxt'
    dir_att = params['output_dir'] + '_att_rxt'
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    if not os.path.isdir(dir_att):
        os.mkdir(dir_att)

    for i, img in enumerate(imgs):
        # load the image
        I = skimage.io.imread(os.path.join(params['images_root'], img.get('filepath', ''), img['filename']))
        # handle grayscale input images
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        I = transform(I).astype('float32') / 255.0
        I = normalize(I.transpose([2, 0, 1]))
        I = paddle.to_tensor(I)
        with paddle.no_grad():
            tmp_fc, tmp_att = my_resnext(I)

        np.save(os.path.join(dir_fc, str(img['imgid'])), tmp_fc.numpy())
        np.savez_compressed(os.path.join(dir_att, str(img['imgid'])), feat=tmp_att.numpy())

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N))

    print('wrote ', params['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default='../filelists/flickr8k-cn/dataset_flickr8k_cn.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default='../filelists/f8ktalk', help='output file')
    parser.add_argument('--images_root', default='../filelists/Flicker8k_Dataset',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)