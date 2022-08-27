import os
import json
import jieba
import re
from tqdm import tqdm

images_root = "../filelists/Flicker8k_Dataset"
file_dir = "../filelists/flickr8k-cn"
output_json = "../filelists/flickr8k-cn/dataset_flickr8k_cn.json"
split_ix = ['train', 'val', 'test']

imgs_caption_dir = os.path.join(file_dir, 'flickr8kzhc.caption.txt')
with open(imgs_caption_dir, 'r', encoding='utf-8') as imgs_caption:
    contents = imgs_caption.readlines()

out = {"dataset": "flickr8k", "images": []}
imgid = 0
for split in split_ix:
    file_split = 'flickr8k' + split + '.txt'
    imags_dir = os.path.join(file_dir, file_split)

    with open(imags_dir, 'r', encoding='utf-8') as imgs_caption:
        imgs = imgs_caption.readlines()

    for img in tqdm(imgs):
        image = {}
        img = img.replace('\n', '') + '.jpg'
        image['split'] = split
        image['filename'] = img
        image['imgid'] = imgid
        sentences = []
        for j in range(len(contents)):
            if contents[j].find(img) != -1:
                sentence = {"imgid": imgid}
                content = contents[j]
                cap_r = content[content.rfind('#zhc#'):-2]
                cap = re.sub('[0-9]', '', cap_r.replace('#zhc#', '')).strip()
                sentence["raw"] = cap
                cut_cap = list(jieba.cut(cap, cut_all=False))
                sentence["tokens"] = cut_cap
                sentences.append(sentence)
        imgid += 1
        image["sentences"] = sentences
        out["images"].append(image)

json.dump(out, open(output_json, 'w'))