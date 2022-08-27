from misc.eval_utils import eval_split
import models
from data.dataloader import *
import argparse
import paddle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Captioning Model')
    parser.add_argument('--dataset', type=str, default="flickr8k")
    parser.add_argument('--input_json', type=str, default='filelists/flickr8k.json',
                        help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_label_h5', type=str, default='filelists/flickr8k_label.h5',
                        help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_fc_dir', type=str, default='filelists/f8ktalk_fc_rxt',
                        help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='filelists/f8ktalk_att_rxt',
                        help='path to the directory containing the preprocessed att feats')

    parser.add_argument('--seq_per_img', type=int, default=5, help='5 sents/image')
    parser.add_argument('--model_name', type=str, default="vatt")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--beam_size', type=int, default=1)

    args = parser.parse_args()
    print('Captioning Model Evaluation')

    loader = DataLoader(args)
    args.vocab_size = loader.get_vocab_size()
    model = models.setup(args)

    fname = 'saved_models/%s_best.pdparams' % args.model_name
    data = paddle.load(fname)
    model.load_dict(data['state_dict'])

    eval_kwargs = {'split': 'test',
                   'beam_size': args.beam_size,
                   'dataset': args.model_name}
    lang_scores, predictions = eval_split(model, loader, eval_kwargs)
    print(lang_scores)

    if not os.path.isdir('vis'):
        os.mkdir('vis')
    json.dump(predictions, open('vis/vis.json', 'w'))