#!/usr/bin/python
# -*- coding: utf-8 -*-

# Trains a simple convnet on the MNIST dataset. Gets to 99.25% test accuracy after 12 epochs
# (there is still a lot of margin for parameter tuning). 16 seconds per epoch on a GRID K520 GPU.
# 训练CNN模型对MNIST数据集分类
# Output after 12 epochs on CPU(i5-7500): ~0.9925

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import csr
from collections import OrderedDict
from tqdm import tqdm_notebook as tqdm


# Utility function to convert list to sparse matrix
# 数据预处理,转换为矩阵(用户数*电影数)
def vectorize_dic(dic, idx=None, dim=None):
    """
    dic -- dictionary of feature lists. Keys are the name of features
    idx -- index generator (default None)
    dim -- dimension of feature space (number of columns in the sparse matrix) (default None)
    """
    if idx is None:
        idx = OrderedDict()
    dic_keys = list(dic.keys())
    n_sample = len(dic[dic_keys[0]])
    n_groups = len(dic.keys())
    n_nzeros = n_sample * n_groups
    col_ix = np.empty(n_nzeros, dtype=int)

    # 计算csr_matrix的入口参数(data, (row_ind,col_ind))
    # row_ix代表data数据在行压缩矩阵对应的行索引,col_idx代表数据在行压缩矩阵对应的列索引
    # 该矩阵的完全形式维度为n*dim(dim包括userID和itemID,排列格式为1user,2user,...,1item,...)
    data = np.ones(n_nzeros)
    row_ix = np.repeat(np.arange(0, n_sample), n_groups)
    i = 0
    for k1, v1 in dic.items():
        for ids in range(len(v1)):
            new_k1 = str(v1[ids]) + str(k1)
            if new_k1 in idx.keys():    # key已经存在
                idx[new_k1] += 1
            else:
                idx[new_k1] = 1
            idx_keys = list(idx.keys())
            col_idxs = idx_keys.index(new_k1)
            col_ix[i + ids * n_groups] = col_idxs
        i += 1

    if dim is None:
        dim = len(idx)
    ixx = np.where(col_ix < dim)
    csr_mat = csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n_sample, dim))

    return csr_mat, idx


# min-batch取数
def batcher(x_, y_=None, _size=-1):
    n_samples = x_.shape[0]

    if _size == -1:
        _size = n_samples
    if _size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(_size))

    for i in range(0, n_samples, _size):
        upper_bound = min(i + _size, n_samples)
        ret_x = x_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:upper_bound]
            yield (ret_x, ret_y)


# 数据MovieLens 100K Dataset
# 数据包括四列: 用户ID/电影ID/打分/时间,且按用户ID升序排列,然后按电影ID升序排列
print('========== 1.Loading data...')
cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('data/ua.base', delimiter='\t', names=cols)
tests = pd.read_csv('data/ua.test', delimiter='\t', names=cols)

# 数据预处理: 原始数据转换为矩阵
print('========== 2.Preprocessing data...')
x_train, ix = vectorize_dic({'users': train['user'].values,
                             'items': train['item'].values})
x_tests, ix = vectorize_dic({'users': tests['user'].values,
                             'items': tests['item'].values}, ix, x_train.shape[1])
y_train = train['rating'].values
y_test = tests['rating'].values
x_train = x_train.todense()
x_tests = x_tests.todense()
print(x_train)
print('----- x_train shape:', x_train.shape)
print('----- x_tests shape:', x_tests.shape)

#
print('========== 3.Building model...')
n, p = x_train.shape
k = 10
x = tf.placeholder('float', [None, p])
y = tf.placeholder('float', [None, 1])
w0 = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.zeros([p]))
v = tf.Variable(tf.random_normal([k, p], mean=0, stddev=0.01))

linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(w, x), 1, keep_dims=True))
pair_interactions = 0.5 * tf.reduce_sum(
    tf.subtract(tf.pow(tf.matmul(x, tf.transpose(v)), 2),
                tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(v, 2)))), axis=1, keep_dims=True)
y_hat = tf.add(linear_terms, pair_interactions)

lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')
l2_norm = tf.reduce_sum(tf.add(tf.multiply(lambda_w, tf.pow(w, 2)),
                               tf.multiply(lambda_v, tf.pow(v, 2))))
error = tf.reduce_mean(tf.square(y-y_hat))
loss = tf.add(error, l2_norm)

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
epochs = 10
batch_size = 1000

# Launch the graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in tqdm(range(epochs), unit='epoch'):
        perm = np.random.permutation(x_train.shape[0])
        # iterate over batches
        for bX, bY in batcher(x_train[perm], y_train[perm], batch_size):
            _, t = sess.run([train_op, loss], feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)})
            print('----- Epoch{0:2d}, Train batch error:{1:.4f}'.format(epoch, t))

    errors = []
    for bX, bY in batcher(x_tests, y_test):
        errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
        print('----- Test batch error:', errors)
    RMSE = np.sqrt(np.array(errors).mean())
    print('----- Test rmse:', RMSE)
