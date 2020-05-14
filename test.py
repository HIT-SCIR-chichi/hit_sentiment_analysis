"""
执行测试集预测.
"""

from train import parse_data, data2vec, word2vec_init
from keras.models import load_model
import jieba

model = load_model('./model/train')
word2vec_init()  # 读取词向量模型，并初始化权重矩阵等

test_data = parse_data('./source/test.txt')  # 读取测试文本
test_data = [jieba.lcut(test_data[idx]) for idx in range(len(test_data))]  # 读取测试数据并分词
test_data = data2vec(test_data)  # 将数据转换为模型的输入格式

res = model.predict_classes(test_data)  # 预测结果
with open('./source/1172510217.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(['%d,%s' % (idx, value[0]) for idx, value in enumerate(res)]))
