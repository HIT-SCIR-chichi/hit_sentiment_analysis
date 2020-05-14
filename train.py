"""
训练模型.
"""
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec
from gensim.corpora import Dictionary
from keras.models import Sequential
from numpy import random, zeros
import jieba

Embed_dim = 100  # 观点的词中的每一个输出的向量维度
Min_count = 1  # 训练词向量中使用的最小词频
Epochs = 2
Batch = 16
dev_size, test_size = 0.1, 0.1
x_train, x_dev, x_test = None, None, None
y_train, y_dev, y_test = None, None, None
seed = 0
max_len = 676  # 观点句经分词后得到列表的长度最大值
word2idx, word2vec, embed_weight = {}, {}, []


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='macro')


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

    model = Sequential()
    model.add(Embedding(len(embed_weight), Embed_dim, mask_zero=True, input_length=max_len, weights=[embed_weight]))
    model.add(LSTM(units=50, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('relu'))
    model.summary()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(data2vec(x_train), y_train, Batch, Epochs, validation_data=(data2vec(x_dev), y_dev))
    model.save('./model/train')

    res = model.evaluate(data2vec(x_test), y_test)  # res[0]为loss值，res[1]为选定指标值
    print('%s: %.2f' % (model.metrics_names[1], res[1]))


if __name__ == '__main__':
    main()
