#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow


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


def output_res(res: dict, path: str = './source/1172510217.csv'):  # 将dict形式的预测结果以csv的形式输出到结果文件
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(['%d,%s' % (idx, res[idx]) for idx in range(len(res))]))


if __name__ == '__main__':
    pos_res = parse_data('./source/sample.positive.txt')
    neg_res = parse_data('./source/sample.negative.txt')
    print(tensorflow.VERSION)
