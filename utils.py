import json
import os
import random
import re
import sys
from pathlib import Path
from queue import PriorityQueue as PQ

import gensim
import jieba

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
dir_path = Path(__file__).parent / "resources"
IMG_PATH = str((dir_path / "meme").absolute()) + "/"
RECORD_PATH = str((dir_path / "record").absolute()) + "/"
DATA_PATH = str((dir_path / "data").absolute()) + "/"

def similar_meme(sentence, logger, n=3):
    vector_path = DATA_PATH + 'sgns.weibo.bigram-char.bin'
    wv = gensim.models.KeyedVectors.load(vector_path, mmap='r')

    with open(DATA_PATH + 'meme.json', 'r', encoding='utf-8') as f:
        meme = json.load(f)
    with open(DATA_PATH + 'cn_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read().split()

    cut = [token for token in jieba.lcut(sentence) if token not in stopwords]
    logger.info(f"分词结果：{cut}")

    try:
        sims = PQ()
        for k, v in meme.items():
            sm = wv.n_similarity(cut, jieba.lcut(v))
            sims.put([1 - sm, k])

    except ZeroDivisionError:
        logger.error("分词结果为空或匹配失败，随机发送表情")
        return random_file(IMG_PATH)

    res = [sims.get() for _ in range(n)]
    logger.info(f"相似度Top3: {res[:3]}")

    if 1 - res[0][0] > 0.5 and res[1][0] > 0.5:
        return f"file:///{IMG_PATH}/" + res[0][1] + '.jpg'
    else:
        return f"file:///{IMG_PATH}/" + random.choice(res[:3])[1] + '.jpg'


def random_file(path=IMG_PATH, regex='\d+', end='\.\w+'):
    file_list = os.listdir(path)
    match_list = []

    for file in file_list:
        match = re.match(f'{regex}{end}', file)
        if match:
            match_list.append(file)

    return f"file:///{path}/{random.choice(match_list)}"


def trans_mark(str):
    # E_pun = u'，。！？【】（）《》“‘：；［］｛｝&，．？（）＼％－＋￣~＄#＠=＿、／'
    # C_pun = u',.!?[]()<>"\':;[]{}&,.?()\\%-+~~$#@=_//'
    E_pun = u'，。！？“‘：；'
    C_pun = u',.!?"\':;'
    table = {ord(f): ord(t) for f, t in zip(C_pun, E_pun)}
    return str.translate(table)
