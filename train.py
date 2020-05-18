"""
训练模型.
"""
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec
from numpy import random, zeros, asarray
from keras.utils import to_categorical
from gensim.corpora import Dictionary
from keras.callbacks import Callback
from sklearn.metrics import f1_score
from keras.models import Sequential
import matplotlib.pyplot as plt
import jieba

embed_dim = 128  # 观点的词中的每一个输出的向量维度
min_count = 1  # 训练词向量中使用的最小词频
dev_size, test_size = 0.1, 0.1  # 调参阶段，设置dev_size为0.1；调参完毕后，将dev划入到train中，所以dev_size为0
epochs, batch, seed, max_len = 8, 128, 0, 676  # max_len表示观点句经分词后得到词语数目的最大值
word2idx, word2vec, embed_weight = {}, {}, []
x_train, x_dev, x_test = None, None, None
y_train, y_dev, y_test = None, None, None


class Metrics(Callback):
    def __init__(self):
        super().__init__()
        self.macro_f1 = []

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val, batch_size = self.validation_data[0], self.validation_data[1], len(self.validation_data[0])
        val_predict = asarray(self.model.predict(x_val, batch_size)).round()
        macro_f1 = f1_score(y_val, val_predict, average='macro')
        self.macro_f1.append(macro_f1)
        print('- val_f1: %.4f' % macro_f1)


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
    model = Word2Vec(x_train, min_count=min_count, size=embed_dim, hs=1, window=3)
    model.save('./model/word2vec')
    return model


def word2vec_init(model=None):  # 加载词向量模型，并计算对应的词嵌入向量矩阵
    global word2vec, word2idx, embed_weight
    model = Word2Vec.load('./model/word2vec') if not model else model  # 加载词向量模型

    dic = Dictionary()
    dic.doc2bow(model.wv.vocab.keys(), allow_update=True)
    word2idx = {token: idx + 1 for idx, token in dic.items()}
    word2vec = {word: model[word] for word in dic.values()}

    embed_weight = zeros((len(word2idx) + 1, embed_dim))
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
    model.add(Embedding(len(embed_weight), embed_dim, mask_zero=True, input_length=max_len, weights=[embed_weight]))
    model.add(SpatialDropout1D(0.5))
    model.add(LSTM(units=64, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    hist = model.fit(data2vec(x_train), y_train, batch, epochs, validation_data=(data2vec(x_dev), y_dev),
                     callbacks=[metrics])  # 调参阶段采用此行代码
    # model.fit(data2vec(x_train), y_train, Batch, Epochs)  # 调参完毕采用此行代码
    model.save('./model/train')

    x2test = data2vec(x_test)
    y_test_pred = asarray(model.predict(x2test, len(x2test))).round()  # 对测试集进行预测
    print('测试集macro-f1: %.4f' % f1_score(y_test, y_test_pred, average='macro'))

    plt.plot(range(epochs), hist.history['accuracy'], range(epochs), hist.history['val_accuracy'], range(epochs),
             metrics.macro_f1)
    plt.legend(['acc', 'val_acc', 'val_f1'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
