import evaluation
from data.coco import COCO
from tqdm import tqdm
from misc.utils import decode_sequence


def language_eval(dataset, preds):
    if 'coco' in dataset:
        annFile = 'filelists/coco2014_captions4eval_cn.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'filelists/f30k_captions4eval_cn.json'
    else:
        annFile = 'filelists/f8k_captions4eval_cn.json'

    coco = COCO(annFile)
    preds_filt = [p for p in preds]
    cocoRes = coco.loadRes(preds_filt)

    imgIds = cocoRes.getImgIds()
    gts = {}
    gen = {}
    for imgId in imgIds:
        gts[imgId] = coco.imgToAnns[imgId]
        gen[imgId] = cocoRes.imgToAnns[imgId]
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def eval_split(model, loader, eval_kwargs):
    split = eval_kwargs['split']
    beam_size = eval_kwargs['beam_size']
    dataset = eval_kwargs['dataset']
    loader.reset_iterator(split)
    dataloader = loader.loaders[split]
    model.eval()
    predictions = []
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, data in enumerate(dataloader):
            tmp = [data['fc_feats'], data['att_feats'],data['att_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, att_masks = tmp
            seqs, _ = model(fc_feats, att_feats, att_masks,
                         opt={'beam_size': eval_kwargs['beam_size']}, mode='sample')
            sents = decode_sequence(loader.get_vocab(), seqs.numpy())
            for k, sent in enumerate(sents):
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                predictions.append(entry)
            pbar.update()

    lang_scores = language_eval(dataset, predictions)
    return lang_scores, predictions
