#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow_estimator import estimator


# LR: Predicting Clicks - Estimating the Click-Through Rate for New Ads.
def lr(features, labels, mode, params):

    # --------------- hyper-parameters --------------- #
    feature_size = params["feature_size"]
    field_size = params["field_size"]
    loss_mode = params["loss_mode"]
    optimizer = params["optimizer"]
    learning_rate = params["learning_rate"]
    l2_reg_lambda = params["l2_reg_lambda"]

    # --------------- initial weights ---------------- #
    # [numeric_feature, one-hot categorical_feature]
    coe_b = tf.get_variable(name="coe_b", shape=[1], initializer=tf.constant_initializer(0.0))
    coe_w = tf.get_variable(name="coe_w", shape=[feature_size], initializer=tf.glorot_normal_initializer())

    # --------------- reshape feature ---------------- #
    feat_idx = features["feat_idx"]         # 非零特征位置[batch_size, field_size, 1]
    feat_idx = tf.reshape(feat_idx, shape=[-1, field_size])     # [Batch, Field]
    feat_val = features["feat_val"]         # 非零特征的值[batch_size, field_size, 1]
    feat_val = tf.reshape(feat_val, shape=[-1, field_size])     # [Batch, Field]

    # ------------------ define f(x) ----------------- #
    # LR: y = b + sum<wi,xi>
    with tf.variable_scope("First-Order"):
        feat_wgt = tf.nn.embedding_lookup(coe_w, feat_idx)              # [Batch, Field]
        y_w = tf.reduce_sum(tf.multiply(feat_wgt, feat_val), 1)         # [Batch]

    with tf.variable_scope("LR-Out"):
        y_b = coe_b * tf.ones_like(y_w, dtype=tf.float32)               # [Batch]
        y_hat = y_b + y_w                                               # [Batch]
        y_pred = tf.nn.sigmoid(y_hat)                                   # [Batch]

    # ---------- mode: predict/evaluate/train ---------- #
    # predict: 不计算loss/metric; evaluate: 不进行梯度下降和参数更新

    # Provide an estimator spec for 'ModeKeys.PREDICT'
    predictions = {"prob": y_pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            estimator.export.PredictOutput(predictions)}
    if mode == estimator.ModeKeys.PREDICT:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Provide an estimator spec for 'ModeKeys.EVAL'
    if loss_mode == "log_loss":
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_hat)) +\
               l2_reg_lambda * tf.nn.l2_loss(coe_w)
    else:
        loss = tf.reduce_mean(tf.square(labels-y_pred))
    eval_metric_ops = {"auc": tf.metrics.auc(labels, y_pred)}
    if mode == estimator.ModeKeys.EVAL:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                       eval_metric_ops=eval_metric_ops)

    # Provide an estimator spec for 'ModeKeys.TRAIN'
    if optimizer == "Adam":
        opt_mode = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif optimizer == "Adagrad":
        opt_mode = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif optimizer == "Momentum":
        opt_mode = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif optimizer == "Ftrl":
        opt_mode = tf.train.FtrlOptimizer(learning_rate)
    else:
        opt_mode = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = opt_mode.minimize(loss, global_step=tf.train.get_global_step())

    if mode == estimator.ModeKeys.TRAIN:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


# FM: Factorization Machines./Factorization Machines with libFM.
# Fast Context-aware Recommendations with Factorization Machines.
def fm(features, labels, mode, params):

    # ---------- hyper-parameters ---------- #
    feature_size = params["feature_size"]
    field_size = params["field_size"]
    embed_size = params["embed_size"]
    loss_mode = params["loss_mode"]
    optimizer = params["optimizer"]
    learning_rate = params["learning_rate"]
    l2_reg_lambda = params["l2_reg_lambda"]

    # ---------- initial weights ----------- #
    # [numeric_feature, one-hot categorical_feature]统一做embedding
    coe_b = tf.get_variable(name="coe_b", shape=[1], initializer=tf.constant_initializer(0.0))
    coe_w = tf.get_variable(name="coe_w", shape=[feature_size], initializer=tf.glorot_normal_initializer())
    coe_v = tf.get_variable(name="coe_v", shape=[feature_size, embed_size],
                            initializer=tf.glorot_normal_initializer())

    # ---------- reshape feature ----------- #
    feat_idx = features["feat_idx"]         # 非零特征位置[batch_size, field_size, 1]
    feat_idx = tf.reshape(feat_idx, shape=[-1, field_size])     # [Batch, Field]
    feat_val = features["feat_val"]         # 非零特征的值[batch_size, field_size, 1]
    feat_val = tf.reshape(feat_val, shape=[-1, field_size])     # [Batch, Field]

    # ------------- define f(x) ------------ #
    # FM: y = b + sum<wi,xi> + sum(<vi,vj>xi*xj)
    with tf.variable_scope("First-Order"):
        feat_wgt = tf.nn.embedding_lookup(coe_w, feat_idx)              # [Batch, Field]
        y_w = tf.reduce_sum(tf.multiply(feat_wgt, feat_val), 1)         # [Batch]

    with tf.variable_scope("Second-Order"):
        embeddings = tf.nn.embedding_lookup(coe_v, feat_idx)            # [Batch, Field, K]
        feat_vals = tf.reshape(feat_val, shape=[-1, field_size, 1])     # [Batch, Field, 1]
        embeddings = tf.multiply(embeddings, feat_vals)                 # [Batch, Field, K]
        sum_square = tf.square(tf.reduce_sum(embeddings, 1))            # [Batch, K]
        square_sum = tf.reduce_sum(tf.square(embeddings), 1)            # [Batch, K]
        y_v = 0.5*tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)     # [Batch]

    with tf.variable_scope("FM-Out"):
        y_b = coe_b * tf.ones_like(y_w, dtype=tf.float32)       # [Batch]
        y_hat = y_b + y_w + y_v                                 # [Batch]
        y_pred = tf.nn.sigmoid(y_hat)                           # [Batch]

    # ----- mode: predict/evaluate/train ----- #
    # predict: 不计算loss/metric; evaluate: 不进行梯度下降和参数更新

    # Provide an estimator spec for 'ModeKeys.PREDICT'
    predictions = {"prob": y_pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            estimator.export.PredictOutput(predictions)}
    if mode == estimator.ModeKeys.PREDICT:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Provide an estimator spec for 'ModeKeys.EVAL'
    if loss_mode == "log_loss":
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_hat)) +\
               l2_reg_lambda * tf.nn.l2_loss(coe_w) + l2_reg_lambda * tf.nn.l2_loss(coe_v)
    else:
        loss = tf.reduce_mean(tf.square(labels-y_pred))
    eval_metric_ops = {"auc": tf.metrics.auc(labels, y_pred)}
    if mode == estimator.ModeKeys.EVAL:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                       eval_metric_ops=eval_metric_ops)

    # Provide an estimator spec for 'ModeKeys.TRAIN'
    if optimizer == "Adam":
        opt_mode = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif optimizer == "Adagrad":
        opt_mode = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif optimizer == "Momentum":
        opt_mode = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif optimizer == "Ftrl":
        opt_mode = tf.train.FtrlOptimizer(learning_rate)
    else:
        opt_mode = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = opt_mode.minimize(loss, global_step=tf.train.get_global_step())

    if mode == estimator.ModeKeys.TRAIN:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


# Deep Crossing - Web-Scale Modeling without Manually Crafted Combinatorial Features.
def deepcrossing(features, labels, mode, params):

    # ---------- hyper-parameters ---------- #
    feature_size = params["feature_size"]
    field_size = params["field_size"]
    embed_size = params["embed_size"]
    loss_mode = params["loss_mode"]
    optimizer = params["optimizer"]
    learning_rate = params["learning_rate"]
    l2_reg_lambda = params["l2_reg_lambda"]
    layers = list(map(int, params["deep_layers"].split(',')))       # l1神经元数量等于D1长度
    dropout = list(map(float, params["dropout"].split(',')))

    # ---------- initial weights ----------- #
    # [numeric_feature, one-hot categorical_feature]统一做embedding
    coe_v = tf.get_variable(name="coe_v", shape=[feature_size, embed_size],
                            initializer=tf.glorot_normal_initializer())

    # ---------- reshape feature ----------- #
    feat_idx = features["feat_idx"]         # 非零特征位置[batch_size, field_size, 1]
    feat_idx = tf.reshape(feat_idx, shape=[-1, field_size])     # [Batch, Field]
    feat_val = features["feat_val"]         # 非零特征的值[batch_size, field_size, 1]
    feat_val = tf.reshape(feat_val, shape=[-1, field_size])     # [Batch, Field]

    # ------------- define f(x) ------------ #
    with tf.variable_scope("Embed-Layer"):
        embeddings = tf.nn.embedding_lookup(coe_v, feat_idx)            # [Batch, Field, K]
        feat_vals = tf.reshape(feat_val, shape=[-1, field_size, 1])     # [Batch, Field, 1]
        embeddings = tf.multiply(embeddings, feat_vals)                 # [Batch, Field, K]

    with tf.variable_scope("Stack-Layer"):
        deep_inputs = tf.reshape(embeddings, shape=[-1, field_size*embed_size])     # [Batch, Field*K]

    with tf.variable_scope("Deep-Layer"):
        # 论文采用ResNet,代码采用fully connected
        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(
                inputs=deep_inputs, num_outputs=layers[i], scope="mlp_%d" % i,
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda))
            if mode == estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

        # output layer
        y_d = tf.contrib.layers.fully_connected(
            inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda), scope='deep_out')

    with tf.variable_scope("DC-Out"):
        y_hat = tf.reshape(y_d, shape=[-1])
        y_pred = tf.nn.sigmoid(y_hat)

    # ----- mode: predict/evaluate/train ----- #
    # predict: 不计算loss/metric; evaluate: 不进行梯度下降和参数更新

    # Provide an estimator spec for 'ModeKeys.PREDICT'
    predictions = {"prob": y_pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            estimator.export.PredictOutput(predictions)}
    if mode == estimator.ModeKeys.PREDICT:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Provide an estimator spec for 'ModeKeys.EVAL'
    if loss_mode == "log_loss":
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_hat)) +\
               l2_reg_lambda * tf.nn.l2_loss(coe_v)
    else:
        loss = tf.reduce_mean(tf.square(labels-y_pred))
    eval_metric_ops = {"auc": tf.metrics.auc(labels, y_pred)}
    if mode == estimator.ModeKeys.EVAL:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                       eval_metric_ops=eval_metric_ops)

    # Provide an estimator spec for 'ModeKeys.TRAIN'
    if optimizer == "Adam":
        opt_mode = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif optimizer == "Adagrad":
        opt_mode = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif optimizer == "Momentum":
        opt_mode = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif optimizer == "Ftrl":
        opt_mode = tf.train.FtrlOptimizer(learning_rate)
    else:
        opt_mode = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = opt_mode.minimize(loss, global_step=tf.train.get_global_step())

    if mode == estimator.ModeKeys.TRAIN:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


# FNN: Deep Learning over Multi-Field Categorical Data: A Case Study on User Response Prediction.
# PNN: Product-based Neural Networks for User Response Prediction.
def fpnn(features, labels, mode, params):

    # ---------- hyper-parameters ---------- #
    algorithm = params["algorithm"]
    feature_size = params["feature_size"]
    field_size = params["field_size"]
    embed_size = params["embed_size"]
    loss_mode = params["loss_mode"]
    optimizer = params["optimizer"]
    learning_rate = params["learning_rate"]
    l2_reg_lambda = params["l2_reg_lambda"]
    layers = list(map(int, params["deep_layers"].split(',')))       # l1神经元数量等于D1长度
    dropout = list(map(float, params["dropout"].split(',')))

    # ---------- initial weights ----------- #
    # [numeric_feature, one-hot categorical_feature]统一做embedding
    coe_b = tf.get_variable(name="coe_b", shape=[1], initializer=tf.constant_initializer(0.0))
    coe_w = tf.get_variable(name="coe_w", shape=[feature_size], initializer=tf.glorot_normal_initializer())
    coe_v = tf.get_variable(name="coe_v", shape=[feature_size, embed_size],
                            initializer=tf.glorot_normal_initializer())
    coe_line = tf.get_variable(name="coe_line", shape=[layers[0], field_size, embed_size],
                               initializer=tf.glorot_normal_initializer())
    coe_ipnn = tf.get_variable(name="coe_ipnn", shape=[layers[0], field_size],
                               initializer=tf.glorot_normal_initializer())
    coe_opnn = tf.get_variable(name="coe_opnn", shape=[layers[0], embed_size, embed_size],
                               initializer=tf.glorot_normal_initializer())

    # ---------- reshape feature ----------- #
    feat_idx = features["feat_idx"]         # 非零特征位置[batch_size, field_size, 1]
    feat_idx = tf.reshape(feat_idx, shape=[-1, field_size])     # [Batch, Field]
    feat_val = features["feat_val"]         # 非零特征的值[batch_size, field_size, 1]
    feat_val = tf.reshape(feat_val, shape=[-1, field_size])     # [Batch, Field]

    # ------------- define f(x) ------------ #
    with tf.variable_scope("Linear-Part"):
        feat_wgt = tf.nn.embedding_lookup(coe_w, feat_idx)              # [Batch, Field]
        y_linear = tf.reduce_sum(tf.multiply(feat_wgt, feat_val), 1)    # [Batch]

    with tf.variable_scope("Embed-Layer"):
        embeddings = tf.nn.embedding_lookup(coe_v, feat_idx)            # [Batch, Field, K]
        feat_vals = tf.reshape(feat_val, shape=[-1, field_size, 1])     # [Batch, Field, 1]
        embeddings = tf.multiply(embeddings, feat_vals)                 # [Batch, Field, K]

    with tf.variable_scope("Product-Layer"):
        if algorithm == "FNN":
            feat_vec = tf.reshape(embeddings, shape=[-1, field_size*embed_size])
            feat_bias = coe_b * tf.reshape(tf.ones_like(y_linear, dtype=tf.float32), shape=[-1, 1])
            deep_inputs = tf.concat([feat_wgt, feat_vec, feat_bias], 1)     # [Batch, (Field+1)*K+1]
        elif algorithm == "IPNN":
            # linear signal
            z = tf.reshape(embeddings, shape=[-1, field_size*embed_size])   # [Batch, Field*K]
            wz = tf.reshape(coe_line, shape=[-1, field_size*embed_size])    # [D1, Field*K]
            lz = tf.matmul(z, tf.transpose(wz))                             # [Batch, D1]

            # quadratic signal
            row_i = []
            col_j = []
            for i in range(field_size - 1):
                for j in range(i + 1, field_size):
                    row_i.append(i)
                    col_j.append(j)
            fi = tf.gather(embeddings, row_i, axis=1)           # 根据索引从参数轴上收集切片[Batch, num_pairs, K]
            fj = tf.gather(embeddings, col_j, axis=1)           # 根据索引从参数轴上收集切片[Batch, num_pairs, K]

            # p_ij = g(fi,fj)=<fi,fj> 特征i和特征j的隐向量的内积
            p = tf.reduce_sum(tf.multiply(fi, fj), 2)           # p矩阵展成向量[Batch, num_pairs]
            wpi = tf.gather(coe_ipnn, row_i, axis=1)            # 根据索引从参数轴上收集切片[D1, num_pairs]
            wpj = tf.gather(coe_ipnn, col_j, axis=1)            # 根据索引从参数轴上收集切片[D1, num_pairs]
            wp = tf.multiply(wpi, wpj)                          # D1个W矩阵组成的矩阵(每行代表一个W)[D1, num_pairs]
            lp = tf.matmul(p, tf.transpose(wp))                 # [Batch, D1]

            lb = coe_b * tf.reshape(tf.ones_like(y_linear, dtype=tf.float32), shape=[-1, 1])
            deep_inputs = lz + lp + lb                          # [Batch, D1]
        elif algorithm == "OPNN":
            # linear signal
            z = tf.reshape(embeddings, shape=[-1, field_size*embed_size])   # [Batch, Field*K]
            wz = tf.reshape(coe_line, shape=[-1, field_size*embed_size])    # [D1, Field*K]
            lz = tf.matmul(z, tf.transpose(wz))                             # [Batch, D1]

            # quadratic signal
            f_sigma = tf.reduce_sum(embeddings, axis=1)                     # [Batch, K]
            p = tf.matmul(tf.reshape(f_sigma, shape=[-1, embed_size, 1]),
                          tf.reshape(f_sigma, shape=[-1, 1, embed_size]))   # [Batch, K, K]
            p = tf.reshape(p, shape=[-1, embed_size*embed_size])            # [Batch, K*K]
            wp = tf.reshape(coe_opnn, shape=[-1, embed_size*embed_size])    # [D1, K*K]
            lp = tf.matmul(p, tf.transpose(wp))                             # [Batch, D1]

            lb = coe_b * tf.reshape(tf.ones_like(y_linear, dtype=tf.float32), shape=[-1, 1])
            deep_inputs = lz + lp + lb                                      # [Batch, D1]

    with tf.variable_scope("Deep-Layer"):
        # hidden layer
        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(
                inputs=deep_inputs, num_outputs=layers[i], scope="mlp_%d" % i,
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda))
            if mode == estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

        # output layer
        y_d = tf.contrib.layers.fully_connected(
            inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda), scope='deep_out')

    with tf.variable_scope("FPNN-Out"):
        y_hat = tf.reshape(y_d, shape=[-1])
        y_pred = tf.nn.sigmoid(y_hat)

    # ----- mode: predict/evaluate/train ----- #
    # predict: 不计算loss/metric; evaluate: 不进行梯度下降和参数更新

    # Provide an estimator spec for 'ModeKeys.PREDICT'
    predictions = {"prob": y_pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            estimator.export.PredictOutput(predictions)}
    if mode == estimator.ModeKeys.PREDICT:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Provide an estimator spec for 'ModeKeys.EVAL'
    if loss_mode == "log_loss":
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_hat)) +\
               l2_reg_lambda * tf.nn.l2_loss(coe_w) + l2_reg_lambda * tf.nn.l2_loss(coe_v)
    else:
        loss = tf.reduce_mean(tf.square(labels-y_pred))
    eval_metric_ops = {"auc": tf.metrics.auc(labels, y_pred)}
    if mode == estimator.ModeKeys.EVAL:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                       eval_metric_ops=eval_metric_ops)

    # Provide an estimator spec for 'ModeKeys.TRAIN'
    if optimizer == "Adam":
        opt_mode = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif optimizer == "Adagrad":
        opt_mode = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif optimizer == "Momentum":
        opt_mode = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif optimizer == "Ftrl":
        opt_mode = tf.train.FtrlOptimizer(learning_rate)
    else:
        opt_mode = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = opt_mode.minimize(loss, global_step=tf.train.get_global_step())

    if mode == estimator.ModeKeys.TRAIN:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


# Wide&Deep: Wide & Deep Learning for Recommender Systems.
def wd(features, labels, mode, params):

    # ---------- hyper-parameters ---------- #
    feature_size = params["feature_size"]
    field_size = params["field_size"]
    embed_size = params["embed_size"]
    loss_mode = params["loss_mode"]
    optimizer = params["optimizer"]
    learning_rate = params["learning_rate"]
    l2_reg_lambda = params["l2_reg_lambda"]
    layers = list(map(int, params["deep_layers"].split(',')))
    dropout = list(map(float, params["dropout"].split(',')))

    # ---------- initial weights ----------- #
    # [numeric_feature, one-hot categorical_feature]统一做embedding
    coe_b = tf.get_variable(name="coe_b", shape=[1], initializer=tf.constant_initializer(0.0))
    coe_w = tf.get_variable(name="coe_w", shape=[feature_size], initializer=tf.glorot_normal_initializer())
    coe_v = tf.get_variable(name="coe_v", shape=[feature_size, embed_size],
                            initializer=tf.glorot_normal_initializer())

    # ---------- reshape feature ----------- #
    feat_idx = features["feat_idx"]         # 非零特征位置[batch_size, field_size, 1]
    feat_idx = tf.reshape(feat_idx, shape=[-1, field_size])     # [Batch, Field]
    feat_val = features["feat_val"]         # 非零特征的值[batch_size, field_size, 1]
    feat_val = tf.reshape(feat_val, shape=[-1, field_size])     # [Batch, Field]

    # ------------- define f(x) ------------ #
    with tf.variable_scope("Wide-Layer"):
        # 论文里面包含人工组合的特征
        feat_wgt = tf.nn.embedding_lookup(coe_w, feat_idx)              # [Batch, Field]
        y_wide = tf.reduce_sum(tf.multiply(feat_wgt, feat_val), 1)      # [Batch]

    with tf.variable_scope("Embed-Layer"):
        embeddings = tf.nn.embedding_lookup(coe_v, feat_idx)            # [Batch, Field, K]
        feat_vals = tf.reshape(feat_val, shape=[-1, field_size, 1])     # [Batch, Field, 1]
        embeddings = tf.multiply(embeddings, feat_vals)                 # [Batch, Field, K]

    with tf.variable_scope("Deep-Layer"):
        deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embed_size])
        # hidden layer
        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(
                inputs=deep_inputs, num_outputs=layers[i], scope="mlp_%d" % i,
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda))
            if mode == estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

        # output layer
        y_d = tf.contrib.layers.fully_connected(
            inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda), scope='deep_out')

    with tf.variable_scope("W_D-Out"):
        y_deep = tf.reshape(y_d, shape=[-1])
        y_bias = coe_b * tf.ones_like(y_wide, dtype=tf.float32)
        y_hat = y_wide + y_deep + y_bias
        y_pred = tf.nn.sigmoid(y_hat)

    # ----- mode: predict/evaluate/train ----- #
    # predict: 不计算loss/metric; evaluate: 不进行梯度下降和参数更新

    # Provide an estimator spec for 'ModeKeys.PREDICT'
    predictions = {"prob": y_pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            estimator.export.PredictOutput(predictions)}
    if mode == estimator.ModeKeys.PREDICT:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Provide an estimator spec for 'ModeKeys.EVAL'
    if loss_mode == "log_loss":
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_hat)) +\
               l2_reg_lambda * tf.nn.l2_loss(coe_w) + l2_reg_lambda * tf.nn.l2_loss(coe_v)
    else:
        loss = tf.reduce_mean(tf.square(labels-y_pred))
    eval_metric_ops = {"auc": tf.metrics.auc(labels, y_pred)}
    if mode == estimator.ModeKeys.EVAL:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                       eval_metric_ops=eval_metric_ops)

    # Provide an estimator spec for 'ModeKeys.TRAIN'
    if optimizer == "Adam":
        opt_mode = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif optimizer == "Adagrad":
        opt_mode = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif optimizer == "Momentum":
        opt_mode = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif optimizer == "Ftrl":
        opt_mode = tf.train.FtrlOptimizer(learning_rate)
    else:
        opt_mode = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = opt_mode.minimize(loss, global_step=tf.train.get_global_step())

    if mode == estimator.ModeKeys.TRAIN:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


# DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.
def deepfm(features, labels, mode, params):

    # ---------- hyper-parameters ---------- #
    feature_size = params["feature_size"]
    field_size = params["field_size"]
    embed_size = params["embed_size"]
    loss_mode = params["loss_mode"]
    optimizer = params["optimizer"]
    learning_rate = params["learning_rate"]
    l2_reg_lambda = params["l2_reg_lambda"]
    layers = list(map(int, params["deep_layers"].split(',')))
    dropout = list(map(float, params["dropout"].split(',')))

    # ---------- initial weights ----------- #
    # [numeric_feature, one-hot categorical_feature]统一做embedding
    coe_b = tf.get_variable(name="coe_b", shape=[1], initializer=tf.constant_initializer(0.0))
    coe_w = tf.get_variable(name="coe_w", shape=[feature_size], initializer=tf.glorot_normal_initializer())
    coe_v = tf.get_variable(name="coe_v", shape=[feature_size, embed_size],
                            initializer=tf.glorot_normal_initializer())

    # ---------- reshape feature ----------- #
    feat_idx = features["feat_idx"]         # 非零特征位置[batch_size, field_size, 1]
    feat_idx = tf.reshape(feat_idx, shape=[-1, field_size])     # [Batch, Field]
    feat_val = features["feat_val"]         # 非零特征的值[batch_size, field_size, 1]
    feat_val = tf.reshape(feat_val, shape=[-1, field_size])     # [Batch, Field]

    # ------------- define f(x) ------------ #
    with tf.variable_scope("First-Order"):
        feat_wgt = tf.nn.embedding_lookup(coe_w, feat_idx)              # [Batch, Field]
        y_w = tf.reduce_sum(tf.multiply(feat_wgt, feat_val), 1)         # [Batch]

    with tf.variable_scope("Second-Order"):
        embeddings = tf.nn.embedding_lookup(coe_v, feat_idx)            # [Batch, Field, K]
        feat_vals = tf.reshape(feat_val, shape=[-1, field_size, 1])     # [Batch, Field, 1]
        embeddings = tf.multiply(embeddings, feat_vals)                 # [Batch, Field, K]
        sum_square = tf.square(tf.reduce_sum(embeddings, 1))            # [Batch, K]
        square_sum = tf.reduce_sum(tf.square(embeddings), 1)            # [Batch, K]
        y_v = 0.5*tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)     # [Batch]

    with tf.variable_scope("Deep-Layer"):
        deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embed_size])
        # hidden layer
        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(
                inputs=deep_inputs, num_outputs=layers[i], scope="mlp_%d" % i,
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda))
            if mode == estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

        # output layer
        y_d = tf.contrib.layers.fully_connected(
            inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda), scope='deep_out')

    with tf.variable_scope("DeepFM-Out"):
        y_deep = tf.reshape(y_d, shape=[-1])
        y_bias = coe_b * tf.ones_like(y_w, dtype=tf.float32)    # [Batch]
        y_hat = y_bias + y_w + y_v + y_deep                     # [Batch]
        y_pred = tf.nn.sigmoid(y_hat)                           # [Batch]

    # ----- mode: predict/evaluate/train ----- #
    # predict: 不计算loss/metric; evaluate: 不进行梯度下降和参数更新

    # Provide an estimator spec for 'ModeKeys.PREDICT'
    predictions = {"prob": y_pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            estimator.export.PredictOutput(predictions)}
    if mode == estimator.ModeKeys.PREDICT:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Provide an estimator spec for 'ModeKeys.EVAL'
    if loss_mode == "log_loss":
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_hat)) +\
               l2_reg_lambda * tf.nn.l2_loss(coe_w) + l2_reg_lambda * tf.nn.l2_loss(coe_v)
    else:
        loss = tf.reduce_mean(tf.square(labels-y_pred))
    eval_metric_ops = {"auc": tf.metrics.auc(labels, y_pred)}
    if mode == estimator.ModeKeys.EVAL:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                       eval_metric_ops=eval_metric_ops)

    # Provide an estimator spec for 'ModeKeys.TRAIN'
    if optimizer == "Adam":
        opt_mode = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif optimizer == "Adagrad":
        opt_mode = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif optimizer == "Momentum":
        opt_mode = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif optimizer == "Ftrl":
        opt_mode = tf.train.FtrlOptimizer(learning_rate)
    else:
        opt_mode = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = opt_mode.minimize(loss, global_step=tf.train.get_global_step())

    if mode == estimator.ModeKeys.TRAIN:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


# DCN: Deep & Cross Network for Ad Click Predictions.
def dcn(features, labels, mode, params):

    # ---------- hyper-parameters ---------- #
    feature_size = params["feature_size"]
    field_size = params["field_size"]
    embed_size = params["embed_size"]
    loss_mode = params["loss_mode"]
    optimizer = params["optimizer"]
    learning_rate = params["learning_rate"]
    l2_reg_lambda = params["l2_reg_lambda"]
    layers = list(map(int, params["deep_layers"].split(',')))
    cross_layers = params["cross_layers"]
    dropout = list(map(float, params["dropout"].split(',')))

    # ---------- initial weights ----------- #
    # [numeric_feature, one-hot categorical_feature]统一做embedding
    coe_b = tf.get_variable(name="coe_b", shape=[1], initializer=tf.constant_initializer(0.0))
    coe_v = tf.get_variable(name="coe_v", shape=[feature_size, embed_size],
                            initializer=tf.glorot_normal_initializer())
    cross_b = tf.get_variable(name="cross_b", shape=[cross_layers, field_size*embed_size],
                              initializer=tf.glorot_uniform_initializer())
    cross_w = tf.get_variable(name="cross_w", shape=[cross_layers, field_size*embed_size],
                              initializer=tf.glorot_uniform_initializer())

    # ---------- reshape feature ----------- #
    feat_idx = features["feat_idx"]         # 非零特征位置[batch_size, field_size, 1]
    feat_idx = tf.reshape(feat_idx, shape=[-1, field_size])     # [Batch, Field]
    feat_val = features["feat_val"]         # 非零特征的值[batch_size, field_size, 1]
    feat_val = tf.reshape(feat_val, shape=[-1, field_size])     # [Batch, Field]

    # ------------- define f(x) ------------ #
    with tf.variable_scope("Embed-Layer"):
        embeddings = tf.nn.embedding_lookup(coe_v, feat_idx)                # [Batch, Field, K]
        feat_vals = tf.reshape(feat_val, shape=[-1, field_size, 1])         # [Batch, Field, 1]
        embeddings = tf.multiply(embeddings, feat_vals)                     # [Batch, Field, K]
        x0 = tf.reshape(embeddings, shape=[-1, field_size*embed_size])      # [Batch, Field*K]

    with tf.variable_scope("Cross-Layer"):
        xl = x0
        for l in range(cross_layers):
            wl = tf.reshape(cross_w[l], shape=[-1, 1])      # [Field*K,1]
            xlw = tf.matmul(xl, wl)                         # [Batch, 1]
            xl = x0 * xlw + cross_b[l]                      # [Batch, Field*K]

    with tf.variable_scope("Deep-Layer"):
        deep_inputs = x0                                    # [Batch, Field*K]
        # hidden layer
        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(
                inputs=deep_inputs, num_outputs=layers[i], scope="mlp_%d" % i,
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda))
            if mode == estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

    with tf.variable_scope("DCN-Out"):
        x_stack = tf.concat([xl, deep_inputs], 1)
        y_comb = tf.contrib.layers.fully_connected(
            inputs=x_stack, num_outputs=1, activation_fn=tf.identity,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda), scope="comb_layer")
        y_d = tf.reshape(y_comb, shape=[-1])                    # [Batch]
        y_bias = coe_b * tf.ones_like(y_d, dtype=tf.float32)
        y_hat = y_d + y_bias                                    # [Batch]
        y_pred = tf.nn.sigmoid(y_hat)                           # [Batch]

    # ----- mode: predict/evaluate/train ----- #
    # predict: 不计算loss/metric; evaluate: 不进行梯度下降和参数更新

    # Provide an estimator spec for 'ModeKeys.PREDICT'
    predictions = {"prob": y_pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            estimator.export.PredictOutput(predictions)}
    if mode == estimator.ModeKeys.PREDICT:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Provide an estimator spec for 'ModeKeys.EVAL'
    if loss_mode == "log_loss":
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_hat)) \
               + l2_reg_lambda * tf.nn.l2_loss(coe_v) + l2_reg_lambda * tf.nn.l2_loss(cross_b) \
               + l2_reg_lambda * tf.nn.l2_loss(cross_w)
    else:
        loss = tf.reduce_mean(tf.square(labels-y_pred)) + l2_reg_lambda * tf.nn.l2_loss(coe_v) \
               + l2_reg_lambda * tf.nn.l2_loss(cross_b) + l2_reg_lambda * tf.nn.l2_loss(cross_w)
    eval_metric_ops = {"auc": tf.metrics.auc(labels, y_pred)}
    if mode == estimator.ModeKeys.EVAL:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                       eval_metric_ops=eval_metric_ops)

    # Provide an estimator spec for 'ModeKeys.TRAIN'
    if optimizer == "Adam":
        opt_mode = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif optimizer == "Adagrad":
        opt_mode = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif optimizer == "Momentum":
        opt_mode = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif optimizer == "Ftrl":
        opt_mode = tf.train.FtrlOptimizer(learning_rate)
    else:
        opt_mode = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = opt_mode.minimize(loss, global_step=tf.train.get_global_step())

    if mode == estimator.ModeKeys.TRAIN:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


# NFM: Neural Factorization Machines for Sparse Predictive Analytics.
def nfm(features, labels, mode, params):

    # ---------- hyper-parameters ---------- #
    feature_size = params["feature_size"]
    field_size = params["field_size"]
    embed_size = params["embed_size"]
    loss_mode = params["loss_mode"]
    optimizer = params["optimizer"]
    learning_rate = params["learning_rate"]
    l2_reg_lambda = params["l2_reg_lambda"]
    layers = list(map(int, params["deep_layers"].split(',')))
    dropout = list(map(float, params["dropout"].split(',')))

    # ---------- initial weights ----------- #
    # [numeric_feature, one-hot categorical_feature]统一做embedding
    coe_b = tf.get_variable(name="coe_b", shape=[1], initializer=tf.constant_initializer(0.0))
    coe_w = tf.get_variable(name="coe_w", shape=[feature_size], initializer=tf.glorot_normal_initializer())
    coe_v = tf.get_variable(name="coe_v", shape=[feature_size, embed_size],
                            initializer=tf.glorot_normal_initializer())

    # ---------- reshape feature ----------- #
    feat_idx = features["feat_idx"]         # 非零特征位置[batch_size, field_size, 1]
    feat_idx = tf.reshape(feat_idx, shape=[-1, field_size])     # [Batch, Field]
    feat_val = features["feat_val"]         # 非零特征的值[batch_size, field_size, 1]
    feat_val = tf.reshape(feat_val, shape=[-1, field_size])     # [Batch, Field]

    # ------------- define f(x) ------------ #
    with tf.variable_scope("First-Order"):
        feat_wgt = tf.nn.embedding_lookup(coe_w, feat_idx)              # [Batch, Field]
        y_w = tf.reduce_sum(tf.multiply(feat_wgt, feat_val), 1)         # [Batch]

    with tf.variable_scope("Bi-Interaction-Layer"):
        embeddings = tf.nn.embedding_lookup(coe_v, feat_idx)            # [Batch, Field, K]
        feat_vals = tf.reshape(feat_val, shape=[-1, field_size, 1])     # [Batch, Field, 1]
        embeddings = tf.multiply(embeddings, feat_vals)                 # [Batch, Field, K]
        sum_square = tf.square(tf.reduce_sum(embeddings, 1))            # [Batch, K]
        square_sum = tf.reduce_sum(tf.square(embeddings), 1)            # [Batch, K]
        bi_out = 0.5*(tf.subtract(sum_square, square_sum))              # [Batch, K]

    with tf.variable_scope("Deep-Layer"):
        deep_inputs = bi_out
        # hidden layer
        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(
                inputs=deep_inputs, num_outputs=layers[i], scope="mlp_%d" % i,
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda))
            if mode == estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

        # output layer
        y_d = tf.contrib.layers.fully_connected(
            inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda), scope='deep_out')

    with tf.variable_scope("NFM-Out"):
        y_deep = tf.reshape(y_d, shape=[-1])
        y_bias = coe_b * tf.ones_like(y_w, dtype=tf.float32)    # [Batch]
        y_hat = y_bias + y_w + y_deep                           # [Batch]
        y_pred = tf.nn.sigmoid(y_hat)                           # [Batch]

    # ----- mode: predict/evaluate/train ----- #
    # predict: 不计算loss/metric; evaluate: 不进行梯度下降和参数更新

    # Provide an estimator spec for 'ModeKeys.PREDICT'
    predictions = {"prob": y_pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            estimator.export.PredictOutput(predictions)}
    if mode == estimator.ModeKeys.PREDICT:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Provide an estimator spec for 'ModeKeys.EVAL'
    if loss_mode == "log_loss":
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_hat)) +\
               l2_reg_lambda * tf.nn.l2_loss(coe_w) + l2_reg_lambda * tf.nn.l2_loss(coe_v)
    else:
        loss = tf.reduce_mean(tf.square(labels-y_pred))
    eval_metric_ops = {"auc": tf.metrics.auc(labels, y_pred)}
    if mode == estimator.ModeKeys.EVAL:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                       eval_metric_ops=eval_metric_ops)

    # Provide an estimator spec for 'ModeKeys.TRAIN'
    if optimizer == "Adam":
        opt_mode = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif optimizer == "Adagrad":
        opt_mode = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif optimizer == "Momentum":
        opt_mode = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif optimizer == "Ftrl":
        opt_mode = tf.train.FtrlOptimizer(learning_rate)
    else:
        opt_mode = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = opt_mode.minimize(loss, global_step=tf.train.get_global_step())

    if mode == estimator.ModeKeys.TRAIN:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)
