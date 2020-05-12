#!/usr/bin/python
# -*- coding: utf-8 -*-


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


def split_data(dev=0.1, val=0.1, seed=12):  # 将数据划分为训练集、验证集、测试集
    pos_res, neg_res = parse_data('./source/sample.positive.txt'), parse_data('./source/sample.negative.txt')
    data, labels = [''] * (len(pos_res) + len(neg_res)), [''] * (len(pos_res) + len(neg_res))  # 数据集，标签集
    for res, label in ((pos_res, 1), (neg_res, 0)):
        for idx in res:
            data[idx], labels[idx] = res[idx], label

    from numpy import random  # 以一个固定的随机种子打乱数据集
    random.seed(seed)
    random.shuffle(data)
    random.seed(seed)
    random.shuffle(labels)

    dev_num, val_num = int(len(data) * dev), int(len(data) * val)  # 划分数据集
    train_num = len(data) - dev_num - val_num
    data = (data[:train_num], data[train_num:train_num + dev_num], data[train_num + dev_num:])
    labels = (labels[:train_num], labels[train_num:train_num + dev_num], labels[train_num + dev_num:])
    return data, labels


def output_res(res: dict, path: str = './source/1172510217.csv'):  # 将dict形式的预测结果以csv的形式输出到结果文件
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(['%d,%s' % (idx, res[idx]) for idx in range(len(res))]))


def main():
    data, labels = split_data()


if __name__ == '__main__':
    main()
