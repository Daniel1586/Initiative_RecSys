#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


# LR: Predicting Clicks - Estimating the Click-Through Rate for New Ads.
def model_lr(features, labels, mode, params):

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
            tf.estimator.export.PredictOutput(predictions)}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Provide an estimator spec for 'ModeKeys.EVAL'
    if loss_mode == "log_loss":
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_hat)) +\
               l2_reg_lambda * tf.nn.l2_loss(coe_w)
    else:
        loss = tf.reduce_mean(tf.square(labels-y_pred))
    eval_metric_ops = {"auc": tf.metrics.auc(labels, y_pred)}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
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

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)
