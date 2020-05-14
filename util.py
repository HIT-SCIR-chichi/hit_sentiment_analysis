#!/usr/bin/python
# -*- coding: utf-8 -*-


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='macro')


def output_res(res: dict, path: str = './source/1172510217.csv'):  # 将dict形式的预测结果以csv的形式输出到结果文件
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(['%d,%s' % (idx, res[idx]) for idx in range(len(res))]))
