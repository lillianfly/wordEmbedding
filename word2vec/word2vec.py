# -*- coding: utf-8 -*-

import logging
import pandas as pd
import jieba
import chardet
from gensim.models import word2vec

# 查看文件编码
file_path = "data_train.csv"
f = open(file_path, 'rb')
data = f.read()
print(chardet.detect(data))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
data = pd.read_csv("data_train.csv", sep="\t", encoding='gb2312', header=None)
sentence = list(data[0])


# 对句子进行分词分词
def segment_sen(sen):
    sen_list = []
    try:
        sen_list = jieba.lcut(sen)
    except:
        pass
    return sen_list


# 将数据变成gensim中 word2wec函数的数据格式
sens_list = [segment_sen(i) for i in sentence]
model = word2vec.Word2Vec(sens_list, sg=1, workers=1, sample=0, min_count=1, iter=20)
#model.save("word2vec.model")
print(model.wv.most_similar("吃饭"))
