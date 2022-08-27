import misc.losses as loss
from misc.rewards import get_self_critical_reward
import models
from data.dataloader import *
from misc.eval_utils import eval_split
from tqdm import tqdm
import argparse
from shutil import copyfile
import paddle.optimizer as optim

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None


def train_xe(model, dataloader, optimizer):
    model.train()
    running_loss = 0.0
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, data in enumerate(dataloader):
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp
            model_out = model(fc_feats, att_feats, labels, att_masks)
            loss = crit(model_out, labels[:, 1:], masks[:, 1:]).mean()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            this_loss = loss.item()
            running_loss += this_loss
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optimizer):
    running_loss = 0.0
    running_reward = 0.0
    beam_size = 1
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, data in enumerate(dataloader):
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['att_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, att_masks = tmp
            gts = data['gts']
            gt_indices = paddle.arange(0, len(data['gts']))

            model.eval()
            with paddle.no_grad():
                greedy_res, _ = model(fc_feats, att_feats, att_masks, beam_size, mode='beam_search')
            model.train()
            gen_result, sample_logprobs = model(fc_feats, att_feats, att_masks,
                                                opt={'sample_n': 5}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result)
            reward = paddle.to_tensor(reward)
            loss = rl_crit(sample_logprobs, gen_result, reward).mean()
            this_reward = reward[:, 0].mean().item()

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            this_loss = loss.item()
            running_loss += this_loss
            running_reward += this_reward
            pbar.set_postfix(reward=running_reward / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    return loss, reward


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

    parser.add_argument('--model_name', type=str, default="vatt")
    parser.add_argument('--seq_per_img', type=int, default=5, help='5 sents/image')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--sc_flag', type=int, default=1)
    parser.add_argument('--resume_best', type=int, default=0)
    parser.add_argument('--resume_last', type=int, default=1)
    parser.add_argument('--label_smoothing', type=float, default=0.2)
    parser.add_argument('--logs_folder', type=str, default='logs')
    parser.add_argument('--max_epochs', type=int, default=25, help='number of epochs')

    args = parser.parse_args()
    print('Captioning Model Training')

    tb_summary_writer = tb and tb.SummaryWriter(os.path.join(args.logs_folder, args.model_name))  # tensorboard --logdir=/

    args.scheduled_sampling_start = 0
    loader = DataLoader(args)
    args.vocab_size = loader.get_vocab_size()
    args.seq_length = loader.get_seq_length()
    model = models.setup(args)

    optimizer = optim.Adam(learning_rate=args.learning_rate,
                           parameters=model.parameters(),
                           beta1=0.9,
                           beta2=0.999,
                           epsilon=1e-8,
                           weight_decay=0.0,
                           grad_clip=paddle.nn.ClipGradByValue(0.1))

    use_sc = args.sc_flag
    best_val_cider = .0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s_last.pdparams' % args.model_name
        else:
            fname = 'saved_models/%s_best.pdparams' % args.model_name

        if os.path.exists(fname):
            data = paddle.load(fname)
            model.load_dict(data['state_dict'])
            optimizer.set_state_dict(data['optimizer'])
            start_epoch = data['epoch'] + 1
            best_val_cider = data['best_val_cider']

    if args.label_smoothing > 0:
        crit = loss.LabelSmoothing(args.vocab_size, smoothing=args.label_smoothing)
    else:
        crit = loss.LanguageModelCriterion()

    rl_crit = loss.RewardCriterion()

    if use_sc:
        args.scheduled_sampling_start = -1

    end_epoch = args.max_epochs
    for epoch in range(start_epoch, end_epoch):

        # Assign the scheduled sampling prob
        if epoch > args.scheduled_sampling_start >= 0:
            model.ss_prob = min(0.05 * (epoch - args.scheduled_sampling_start) // 5, 0.5)

        loader.reset_iterator('train')
        dataloader_train = loader.loaders['train']

        # If start self critical training
        if not use_sc:
            train_loss = train_xe(model, dataloader_train, optimizer)
        else:
            train_loss, reward = train_scst(model, dataloader_train, optimizer)
            tb_summary_writer.add_scalar('avg_reward', reward, epoch)

        tb_summary_writer.add_scalar('train_loss', train_loss, epoch)

        # eval model
        eval_kwargs = {'split': 'val',
                       'beam_size': 1,
                       'dataset': args.dataset}
        lang_scores, _ = eval_split(model, loader, eval_kwargs)
        print("Validation scores", lang_scores)

        tb_summary_writer.add_scalar('BLEU1', lang_scores['BLEU'][0], epoch)
        tb_summary_writer.add_scalar('BLEU4', lang_scores['BLEU'][3], epoch)
        tb_summary_writer.add_scalar('METEOR', lang_scores['METEOR'], epoch)
        tb_summary_writer.add_scalar('ROUGE', lang_scores['ROUGE'], epoch)
        tb_summary_writer.add_scalar('CIDEr', lang_scores['CIDEr'], epoch)

        current_cider = lang_scores['CIDEr']

        best_flag = False
        if current_cider > best_val_cider:
            best_val_cider = current_cider
            best_flag = True

        paddle.save({
            'epoch': epoch,
            'current_cider': current_cider,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_cider': best_val_cider,
            'use_sc': use_sc,
        }, 'saved_models/%s_last.pdparams' % args.model_name)

        if best_flag:
            copyfile('saved_models/%s_last.pdparams' % args.model_name,
                     'saved_models/%s_best.pdparams' % args.model_name)