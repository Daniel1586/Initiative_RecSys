#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
<<DeepFM: A Factorization-Machine based Neural Network for CTR Prediction>>
Implementation of DeepFM model with the following features：
#1 Input pipeline using Dataset API, Support parallel and prefetch
#2 Train pipeline using Custom Estimator by rewriting model_fn
#3 Support distributed training by TF_CONFIG
#4 Support export_model for TensorFlow Serving
"""

import shutil
#import sys
import os
import json
import glob
from datetime import date, timedelta
from time import time
#import gc
#from multiprocessing import Process

#import math
import random
#import pandas as pd
#import numpy as np
import tensorflow as tf

# =================== CMD Arguments for DeepFM =================== #
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("run_mode", 0, "run mode {0-local, 1-single_dist, 2-multi_dist}")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 16, "Number of threads")
tf.app.flags.DEFINE_integer("feature_size", 0, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 0, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normalization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("task_mode", 'train', "task mode {train, infer, eval, export}")
tf.app.flags.DEFINE_string("model_dir", '', "model check point dir")
tf.app.flags.DEFINE_string("data_dir", '', "data dir")
tf.app.flags.DEFINE_string("flag_dir", '', "flag name for different test time")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")


# 0 1:0.05 2:0.004983 3:0.05 4:0 5:0.021594 6:0.008 7:0.15 8:0.04 9:0.362 10:0.166667 11:0.2 12:0 13:0.04
def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing ---- ', filenames)

    def decode_dataset(line):
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        id_vals = tf.reshape(splits.values, splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    # multi-thread pre-process then prefetch
    dataset = tf.data.TextLineDataset(filenames).map(decode_dataset, num_parallel_calls=10).prefetch(2000)

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def model_fn(features, labels, mode, params):

    # ----- hyper-parameters ----- #
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    layers = map(int, params["deep_layers"].split(','))
    dropout = map(float, params["dropout"].split(','))

    # ----- initial weights ----- #
    FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
    FM_W = tf.get_variable(name='fm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    FM_V = tf.get_variable(name='fm_v', shape=[feature_size, embedding_size], initializer=tf.glorot_normal_initializer())

    #------build feaure-------
    feat_ids  = features['feat_ids']
    feat_ids = tf.reshape(feat_ids,shape=[-1,field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])

    #------build f(x)------
    with tf.variable_scope("First-order"):
        feat_wgts = tf.nn.embedding_lookup(FM_W, feat_ids)              # None * F * 1
        y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals),1)

    with tf.variable_scope("Second-order"):
        embeddings = tf.nn.embedding_lookup(FM_V, feat_ids)             # None * F * K
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals)                 #vij*xi
        sum_square = tf.square(tf.reduce_sum(embeddings,1))
        square_sum = tf.reduce_sum(tf.square(embeddings),1)
        y_v = 0.5*tf.reduce_sum(tf.subtract(sum_square, square_sum),1)	# None * 1

    with tf.variable_scope("Deep-part"):
        if FLAGS.batch_norm:
            #normalizer_fn = tf.contrib.layers.batch_norm
            #normalizer_fn = tf.layers.batch_normalization
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
                #normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': True, 'reuse': None}
            else:
                train_phase = False
                #normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': False, 'reuse': True}
        else:
            normalizer_fn = None
            normalizer_params = None

        deep_inputs = tf.reshape(embeddings,shape=[-1,field_size*embedding_size]) # None * (F*K)
        for i in range(len(layers)):
            #if FLAGS.batch_norm:
            #    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)
                #normalizer_params.update({'scope': 'bn_%d' %i})
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], \
                #normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp%d' % i)
            if FLAGS.batch_norm:
                deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)   #放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])                              #Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)
                #deep_inputs = tf.layers.dropout(inputs=deep_inputs, rate=dropout[i], training=mode == tf.estimator.ModeKeys.TRAIN)

        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='deep_out')
        y_d = tf.reshape(y_deep,shape=[-1])
        #sig_wgts = tf.get_variable(name='sigmoid_weights', shape=[layers[-1]], initializer=tf.glorot_normal_initializer())
        #sig_bias = tf.get_variable(name='sigmoid_bias', shape=[1], initializer=tf.constant_initializer(0.0))
        #deep_out = tf.nn.xw_plus_b(deep_inputs,sig_wgts,sig_bias,name='deep_out')

    with tf.variable_scope("DeepFM-out"):
        #y_bias = FM_B * tf.ones_like(labels, dtype=tf.float32)  # None * 1  warning;这里不能用label，否则调用predict/export函数会出错，train/evaluate正常；初步判断estimator做了优化，用不到label时不传
        y_bias = FM_B * tf.ones_like(y_d, dtype=tf.float32)      # None * 1
        y = y_bias + y_w + y_v + y_d
        pred = tf.sigmoid(y)

    predictions={"prob": pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    #------bulid loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
        l2_reg * tf.nn.l2_loss(FM_W) + \
        l2_reg * tf.nn.l2_loss(FM_V)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    #------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    #return tf.estimator.EstimatorSpec(
    #        mode=mode,
    #        loss=loss,
    #        train_op=train_op,
    #        predictions={"prob": pred},
    #        eval_metric_ops=eval_metric_ops)


def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


# Initialized Environment,初始化程序运行环境
def env_set():
    if FLAGS.run_mode == 1:        # 单机分布式
        ps_hosts = FLAGS.ps_hosts.split(',')
        chief_hosts = FLAGS.chief_hosts.split(',')
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host ------', ps_hosts)
        print('chief_hosts --', chief_hosts)
        print('job_name -----', job_name)
        print('task_index ---', str(task_index))
        # 无worker参数
        tf_config = {
            'cluster': {'chief': chief_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index}
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    elif FLAGS.run_mode == 2:      # 集群分布式
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1]     # get first worker as chief
        worker_hosts = worker_hosts[2:]     # the rest as worker
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host ------', ps_hosts)
        print('worker_host --', worker_hosts)
        print('chief_hosts --', chief_hosts)
        print('job_name -----', job_name)
        print('task_index ---', str(task_index))
        # use #worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        if job_name == "worker" and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 1:
            task_index -= 2

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index}
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)


def main(_):
    print('========== 1.Check and Print Arguments...')
    if FLAGS.flag_dir == "":
        FLAGS.flag_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.flag_dir
    print('task_mode --------- ', FLAGS.task_mode)
    print('model_dir --------- ', FLAGS.model_dir)
    print('data_dir ---------- ', FLAGS.data_dir)
    print('flag_dir ---------- ', FLAGS.flag_dir)
    print('num_epochs -------- ', FLAGS.num_epochs)
    print('feature_size ------ ', FLAGS.feature_size)
    print('field_size -------- ', FLAGS.field_size)
    print('embedding_size ---- ', FLAGS.embedding_size)
    print('batch_size -------- ', FLAGS.batch_size)
    print('deep_layers ------- ', FLAGS.deep_layers)
    print('dropout ----------- ', FLAGS.dropout)
    print('loss_type --------- ', FLAGS.loss_type)
    print('optimizer --------- ', FLAGS.optimizer)
    print('learning_rate ----- ', FLAGS.learning_rate)
    print('batch_norm_decay -- ', FLAGS.batch_norm_decay)
    print('batch_norm -------- ', FLAGS.batch_norm)
    print('l2_reg ------------ ', FLAGS.l2_reg)

    print('========== 2.Check and Print Dataset files...')
    if FLAGS.data_dir == "":    # windows环境测试
        root_dir = os.path.abspath(os.path.dirname(os.getcwd()))
        FLAGS.data_dir = root_dir + '\\criteo_dataout\\'

    train_files = glob.glob("%s/train*set" % FLAGS.data_dir)
    random.shuffle(train_files)
    print("train_files: ", train_files)
    valid_files = glob.glob("%s/valid*set" % FLAGS.data_dir)
    print("valid_files: ", valid_files)
    tests_files = glob.glob("%s/tests*set" % FLAGS.data_dir)
    print("tests_files: ", tests_files)

    print('========== 3.Initialized Environment...')
    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    env_set()

    print('========== 4.Build tasks and algorithm model...')
    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "batch_norm_decay": FLAGS.batch_norm_decay,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout
    }
    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': FLAGS.num_threads}),
        log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    deepfm = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir,
                                    params=model_params, config=config)

    if FLAGS.task_mode == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(train_files, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs))
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(valid_files, batch_size=FLAGS.batch_size, num_epochs=1),
            steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(deepfm, train_spec, eval_spec)
    elif FLAGS.task_mode == 'eval':
        deepfm.evaluate(input_fn=lambda: input_fn(valid_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_mode == 'infer':
        preds = deepfm.predict(
            input_fn=lambda: input_fn(tests_files, num_epochs=1, batch_size=FLAGS.batch_size),
            predict_keys="prob")
        with open(FLAGS.data_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_mode == 'export':
        feature_spec = {
            'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
            'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        deepfm.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
