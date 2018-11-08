#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf


# 生成数据,_feature特征数量,_samples样本数量,_field类别数量
def data_gen(_feature, _samples, _field):
    labels = [-1, 1]
    _y = [np.random.choice(labels, 1)[0] for _ in range(_samples)]
    _f = [_k // int(_feature/_field) for _k in range(_feature)]
    _x = np.random.randint(0, 2, size=(_samples, _feature))
    return _x, _y, _f


# w0权值初始化
def w0_init():
    w0 = tf.truncated_normal([1])
    tf_w0 = tf.Variable(w0)
    return tf_w0


# w1权值初始化
def w1_init(_size):
    w1 = tf.truncated_normal([_size])
    tf_w1 = tf.Variable(w1)
    return tf_w1


# w2权值初始化
def w2_init(_size, _field, _dim):
    w2 = tf.truncated_normal([_size, _field, _dim])
    tf_w2 = tf.Variable(w2)
    return tf_w2


# 计算y的预测值
def inference(_x, _field, _w0, _w1, _w2, _feature, _dim):
    _linear = tf.add(_w0, tf.reduce_sum(tf.multiply(_w1, _x)), name="linear")
    _second = tf.Variable(0.0, dtype=tf.float32)

    for i1 in range(_feature):  # 遍历特征
        idx1_feature = i1
        idx1_field = int(_field[i1])
        for j1 in range(i1+1, _feature):
            idx2_feature = j1
            idx2_field = int(_field[j1])

            vec_i1 = tf.convert_to_tensor([[idx1_feature, idx2_field, ii] for ii in range(_dim)])
            vec_lt = tf.gather_nd(_w2, vec_i1)
            vec_i_fj = tf.squeeze(vec_lt)

            vec_j1 = tf.convert_to_tensor([[idx2_feature, idx1_field, jj] for jj in range(_dim)])
            vec_rt = tf.gather_nd(_w2, vec_j1)
            vec_j_fi = tf.squeeze(vec_rt)
            vec_value = tf.reduce_sum(tf.multiply(vec_i_fj, vec_j_fi))

            indices2 = [i1]
            indices3 = [j1]
            xi = tf.squeeze(tf.gather_nd(_x, indices2))
            xj = tf.squeeze(tf.gather_nd(_x, indices3))
            product = tf.reduce_sum(tf.multiply(xi, xj))
            temp_value = tf.multiply(vec_value, product)
            tf.assign(_second, tf.add(_second, temp_value))

    return tf.add(_linear, _second)


if __name__ == '__main__':
    MODEL_NAME = "FFM"
    MODEL_SAVE_PATH = "FFModel"
    num_field = 2
    num_feature = 20
    num_samples = 100

    dim_v = 3
    lr = 0.01
    epochs = 50
    print('========== 1.Generating data...')
    train_x, train_y, train_field = data_gen(num_feature, num_samples, num_field)

    print('========== 2.Building model...')
    global_step = tf.Variable(0, trainable=False)
    input_x = tf.placeholder(tf.float32, [num_feature])
    input_y = tf.placeholder(tf.float32)

    init_w0 = w0_init()
    init_w1 = w1_init(num_feature)
    init_w2 = w2_init(num_feature, num_field, dim_v)    # n * f * k
    y_hat = inference(input_x, train_field, init_w0, init_w1, init_w2, num_feature, dim_v)

    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')
    l2_w1 = tf.multiply(lambda_w, tf.pow(init_w1, 2))
    l2_w2 = tf.reduce_sum(tf.multiply(lambda_v, tf.pow(init_w2, 2)), axis=[1, 2])
    l2_norm = tf.reduce_sum(tf.add(l2_w1, l2_w2))
    loss = tf.log(1 + tf.exp(input_y * y_hat)) + l2_norm

    train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            for t in range(num_samples):
                input_x_batch = train_x[t]
                input_y_batch = train_y[t]
                predict_loss, _, steps = sess.run([loss, train_step, global_step],
                                                  feed_dict={input_x: input_x_batch, input_y: input_y_batch})
                print("After {step} training step(s),   loss on training batch is {predict_loss}"
                      .format(step=t, predict_loss=predict_loss))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=steps)
            writer = tf.summary.FileWriter(os.path.join(MODEL_SAVE_PATH, MODEL_NAME), tf.get_default_graph())
            writer.close()
