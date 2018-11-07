#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import csr
from collections import OrderedDict
from tqdm import tqdm_notebook as tqdm


# Utility function to convert list to sparse matrix
# 数据预处理,样本数据one-hot编码并压缩(样本长度为用户数+电影数)
def onehot_compress(dic, idx=None, dim=None):
    """
    dic -- dictionary of feature lists. Keys are the name of features
    idx -- index generator (default None)
    dim -- dimension of feature space (number of columns in the sparse matrix) (default None)
    """

    if idx is None:
        idx = OrderedDict()

    dic_keys = list(dic.keys())
    n_samples = len(dic[dic_keys[0]])
    n_feature = len(dic.keys())
    nz = n_samples * n_feature

    # 计算csr_matrix的入口参数(data,(row_ind,col_ind))
    # data   : 代表data数据
    # row_ind: 代表data数据在原始稀疏矩阵对应的行索引
    # col_ind: 代表data数据在原始稀疏矩阵对应的列索引
    # 原始稀疏矩阵的维度为num*dim(num表示样本数量,dim表示特征数量[不同的user和item])
    row_ix = np.repeat(np.arange(0, n_samples), n_feature)
    col_ix = np.empty(nz, dtype=int)
    i = 0
    i_feature = 0
    for k1, v in dic.items():
        for ids in range(len(v)):
            # 获得新特征,并定于新特征的列索引位置
            new_k1 = str(v[ids]) + str(k1)
            if new_k1 in idx.keys():        # key已经存在
                col_ix[i + ids * n_feature] = idx[new_k1]
            else:
                idx[new_k1] = i_feature     # key代表特征,value代表特征的列索引
                col_ix[i + ids * n_feature] = idx[new_k1]
                i_feature += 1
        i += 1

    data = np.ones(nz)
    if dim is None:
        dim = len(idx)

    ixx = np.where(col_ix < dim)
    csr_mat = csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n_samples, dim))
    return csr_mat, idx


# min-batch取数
def batcher(x_, y_=None, _size=-1):
    n_samples = x_.shape[0]

    if _size == -1:
        _size = n_samples
    if _size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(_size))

    for k1 in range(0, n_samples, _size):
        upper_bound = min(k1 + _size, n_samples)
        ret_x = x_[k1:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[k1:upper_bound]
            yield (ret_x, ret_y)


# 数据MovieLens 100K Dataset
# 数据包括四列: 用户ID/电影ID/评分/时间,且按用户ID升序排列,然后按电影ID升序排列
print('========== 1.Loading dataset...')
cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('data/ua.base', delimiter='\t', names=cols)
tests = pd.read_csv('data/ua.test', delimiter='\t', names=cols)
print(len(train.index))

# 数据预处理: 样本数据one-hot编码并压缩(样本长度为用户数+电影数)
print('========== 2.Preprocessing dataset...')
x_train, ix = onehot_compress({'users': train['user'].values,
                               'items': train['item'].values})


x_tests, ix = onehot_compress({'users': tests['user'].values,
                               'items': tests['item'].values},
                              ix, x_train.shape[1])

y_train = train['rating'].values
y_tests = tests['rating'].values
print(x_train)

x_train = x_train.todense()
x_tests = x_tests.todense()
print(x_train)

print('----- x_train shape:', x_train.shape)
print('----- x_tests shape:', x_tests.shape)

# FM模型
print('========== 3.Building model...')
k = 10          # 特征向量xi的辅助向量vi的长度
x = tf.placeholder('float', [None, x_train.shape[1]])
y = tf.placeholder('float', [None, 1])
w0 = tf.Variable(tf.zeros([1]))
w1 = tf.Variable(tf.zeros([x_train.shape[1]]))
v1 = tf.Variable(tf.random_normal([k, x_train.shape[1]], mean=0, stddev=0.01))

linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(w1, x), 1, keep_dims=True))
second_terms = 0.5 * tf.reduce_sum(
    tf.subtract(tf.pow(tf.matmul(x, tf.transpose(v1)), 2),
                tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(v1, 2)))), axis=1, keep_dims=True)
y_hat = tf.add(linear_terms, second_terms)

lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')
l2_norm = tf.reduce_sum(tf.add(tf.multiply(lambda_w, tf.pow(w1, 2)),
                               tf.multiply(lambda_v, tf.pow(v1, 2))))
error = tf.reduce_mean(tf.square(y-y_hat))
loss = tf.add(error, l2_norm)
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

print('========== 4.Evaluating model...')
epochs = 10
batch_size = 1000
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in tqdm(range(epochs), unit='epoch'):
        perm = np.random.permutation(x_train.shape[0])
        # iterate over batches
        for bX, bY in batcher(x_train[perm], y_train[perm], batch_size):
            _, t = sess.run([train_op, loss],
                            feed_dict={x: bX.reshape(-1, x_train.shape[1]),
                                       y: bY.reshape(-1, 1)})
            print('----- Epoch{0:2d}, Train batch error:{1:.4f}'.format(epoch, t))

    errors = []
    for bX, bY in batcher(x_tests, y_tests):
        errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, x_train.shape[1]),
                                                 y: bY.reshape(-1, 1)}))
        print('----- Test batch error:', errors)
    RMSE = np.sqrt(np.array(errors).mean())
    print('----- Test rmse:', RMSE)
