"""
训练模型.
"""
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, recall_score
from gensim.models.word2vec import Word2Vec
from numpy import random, zeros, asarray
from keras.utils import to_categorical
from keras.callbacks import Callback
from gensim.corpora import Dictionary
from keras.models import Sequential
import jieba

Embed_dim = 128  # 观点的词中的每一个输出的向量维度
Min_count = 1  # 训练词向量中使用的最小词频
Epochs = 8
Batch = 128
dev_size, test_size = 0, 0.1
x_train, x_dev, x_test = None, None, None
y_train, y_dev, y_test = None, None, None
seed = 0
max_len = 676  # 观点句经分词后得到列表的长度最大值
word2idx, word2vec, embed_weight = {}, {}, []


class Metrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_predict = (asarray(self.model.predict(self.validation_data[0]))).round()
        val_labels = self.validation_data[1]
        val_recall = recall_score(val_labels, val_predict, average='macro')
        macro_f1 = f1_score(val_labels, val_predict, average='macro')
        print('- val_recall: %.4f - val_f1: %.4f' % (macro_f1, val_recall))


def parse_data(path: str):  # 解析教师给的源标签文件，并将评论中的换行与空格字符去除，返回值为词典{index:comment}
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


def split_data():  # 将数据划分为训练集、验证集、测试集，格式为二维列表，第二维度为词列表
    global x_train, x_dev, x_test, y_train, y_dev, y_test
    pos_res, neg_res = parse_data('./source/sample.positive.txt'), parse_data('./source/sample.negative.txt')
    x, y = [''] * (len(pos_res) + len(neg_res)), [''] * (len(pos_res) + len(neg_res))  # 数据集，标签集
    for res, label in ((pos_res, 1), (neg_res, 0)):
        for idx in res:
            x[idx], y[idx] = jieba.lcut(res[idx]), label

    # 用固定的随机种子打乱数据及其对应的标签
    random.seed(seed)
    random.shuffle(x)
    random.seed(seed)
    random.shuffle(y)

    dev_num, test_num = int(len(x) * dev_size), int(len(x) * test_size)
    train_num = len(x) - dev_num - test_num
    y = to_categorical(y, 2, 'int32')
    x_train, x_dev, x_test = x[:train_num], x[train_num:train_num + dev_num], x[train_num + dev_num:]
    y_train, y_dev, y_test = y[:train_num], y[train_num:train_num + dev_num], y[train_num + dev_num:]


def word2vec_train():  # 训练词向量模型，并输出结果到模型文件中
    model = Word2Vec(x_train, min_count=Min_count, size=Embed_dim, hs=1, window=3)  # todo 扩大训练集
    model.save('./model/word2vec')
    return model


def word2vec_init(model=None):  # 加载词向量模型，并计算对应的词嵌入向量矩阵
    global word2vec, word2idx, embed_weight
    model = Word2Vec.load('./model/word2vec') if not model else model  # 加载词向量模型

    dic = Dictionary()
    dic.doc2bow(model.wv.vocab.keys(), allow_update=True)
    word2idx = {token: idx + 1 for idx, token in dic.items()}
    word2vec = {word: model[word] for word in dic.values()}

    embed_weight = zeros((len(word2idx) + 1, Embed_dim))
    for word, idx in word2idx.items():
        embed_weight[idx, :] = word2vec[word]  # 词向量矩阵，第一行是0向量


def data2vec(words_lst):  # 将评论分词列表数据转换为对应的词向量形式，输入二维列表，输出shape(评论数, max_len)
    res = [[word2idx[word] if word in word2idx else 0 for word in words] for words in words_lst]
    return pad_sequences(res, max_len)  # 将第二维填充


def main():
    split_data()
    word2vec_model = word2vec_train()
    word2vec_init(word2vec_model)
    metrics = Metrics()  # 回调函数用于计算宏F1值

    model = Sequential()
    model.add(Embedding(len(embed_weight), Embed_dim, mask_zero=True, input_length=max_len, weights=[embed_weight]))
    model.add(SpatialDropout1D(0.5))
    model.add(LSTM(units=64, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # model.fit(data2vec(x_train), y_train, Batch, Epochs, validation_data=(data2vec(x_dev), y_dev), callbacks=[metrics])
    model.fit(data2vec(x_train), y_train, Batch, Epochs)
    model.save('./model/train')

    y_test_pred = (asarray(model.predict(data2vec(x_test)))).round()  # 对测试集进行预测
    res = f1_score(y_test, y_test_pred, average='macro')
    print('f1: %.4f' % res)


if __name__ == '__main__':
    main()
