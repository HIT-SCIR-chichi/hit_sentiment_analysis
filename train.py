#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
训练模型.
"""
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
import numpy as np
import jieba

Embed_dim = 100  # 观点的词中的每一个输出的向量维度
Epochs = 6
Batch = 16


def parse_data(path: str):  # 解析教师给的源标签文件，并将评论中的换行与空格字符去除
    with open(path, 'r', encoding='utf-8') as f:
        res, data_id, data_info = {}, -1, ''  # res保存dict结果{id: info}，data_id标识xml中的id值，data_info标识评论
        for line in f:
            line = line.replace('\n', '').replace(' ', '')  # 去掉评论中的空格以及读取的换行符
            if line:
                if '<review' in line:
                    data_id = int(line.split('"')[1])
                elif '</review>' in line:
                    res[data_id], data_id, data_info = data_info, -1, ''
                else:
                    data_info += line
        return res


class Analysis:
    def __init__(self, test=0.1, seed=0, fold=10):
        self.test = test  # 测试集比例
        self.seed = seed  # 随机种子，用于重复生成相同数据
        self.fold = fold  # 交叉验证集划分的数目

        self.x_train = None  # 训练集数据，索引形式
        self.y_train = None  # 训练集标签
        self.x_test = None  # 测试集数据，索引形式
        self.y_test = None  # 测试集标签

        self.words_dic = {}  # 词典
        self.max_len = 0  # 观点句经分词后得到列表的长度最大值
        self.word2idx = {}  # 将词语映射到索引 todo 将idx按照词频高低设置；或者将单个字作为索引，而非词语

        self._split_test_data()

    def _split_test_data(self):  # 将数据划分为训练集、测试集
        pos_res, neg_res = parse_data('./source/sample.positive.txt'), parse_data('./source/sample.negative.txt')
        x, y = [''] * (len(pos_res) + len(neg_res)), [''] * (len(pos_res) + len(neg_res))  # 数据集，标签集
        for res, label in ((pos_res, 1), (neg_res, 0)):
            for idx in res:
                x[idx], y[idx] = jieba.lcut(res[idx]), label
                for word in x[idx]:
                    if word not in self.words_dic:
                        self.word2idx[word] = len(self.word2idx) + 1  # 设置0为填充，所以idx=0不表示词语
                        self.words_dic[word] = 0
                    self.words_dic[word] += 1

        self.max_len = max(len(words) for words in x)  # 一句评论最长分词数目
        x = [[self.word2idx[word] for word in words] for words in x]  # 将数据转换为词语索引表示
        x = pad_sequences(x, self.max_len)  # 在左侧填充数据集，填充数据为0

        res = train_test_split(x, y, shuffle=True, random_state=self.seed, test_size=self.test, stratify=y)
        self.x_train, self.x_test, self.y_train, self.y_test = res

    def iter_train_dev(self):  # 从训练集中取验证集
        fold = StratifiedKFold(n_splits=self.fold, shuffle=True, random_state=self.seed)
        for train_idx, dev_idx in fold.split(self.x_train, self.y_train):
            x_train, x_dev, y_train, y_dev = [], [], [], []
            for idx in train_idx:
                x_train.append(self.x_train[idx])
                y_train.append(self.y_train[idx])
            for idx in dev_idx:
                x_dev.append(self.x_train[idx])
                y_dev.append(self.y_train[idx])
            # yield np.array(x_train), np.array(x_dev), to_categorical(y_train, 2), to_categorical(y_dev, 2)
            yield np.array(x_train), np.array(x_dev), y_train, y_dev


def main():
    analysis = Analysis()
    for x_train, x_dev, y_train, y_dev in analysis.iter_train_dev():
        model = Sequential()
        model.add(Embedding(len(analysis.words_dic), Embed_dim, mask_zero=True, input_length=analysis.max_len))
        model.add(LSTM(units=128))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, Batch, Epochs, validation_data=(x_dev, y_dev))

        # res = model.evaluate(analysis.x_test, to_categorical(analysis.y_test, 2), 128)
        res = model.evaluate(analysis.x_test, analysis.y_test, 128)
        print(res)
        break


if __name__ == '__main__':
    main()
