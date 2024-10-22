#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
<<DCN: Deep & Cross Network for Ad Click Predictions>>
Implementation of DeepFM model with the following features：
#1 Input pipeline using Dataset API, Support parallel and prefetch
#2 Train pipeline using Custom Estimator by rewriting model_fn
#3 Support distributed training by TF_CONFIG
#4 Support export_model for TensorFlow Serving
########## TF Version: 1.8.0 ##########
"""

import os
import json
import glob
import random
import shutil
import tensorflow as tf
from datetime import date, timedelta

# =================== CMD Arguments for DCN model =================== #
flags = tf.app.flags
flags.DEFINE_integer("run_mode", 0, "{0-local, 1-single_distributed, 2-multi_distributed}")
flags.DEFINE_boolean("clr_mode", True, "Clear existed model or not")
flags.DEFINE_string("task_mode", "train", "{train, infer, eval, export}")
flags.DEFINE_string("ps_hosts", None, "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", None, "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "Job name: ps or worker")
flags.DEFINE_integer("task_index", None, "Index of task within the job")
flags.DEFINE_integer("num_threads", 4, "Number of threads")
flags.DEFINE_string("data_dir", "", "Data dir")
flags.DEFINE_string("model_dir", "", "Model check point dir")
flags.DEFINE_string("mark_dir", "", "Mark different model")
flags.DEFINE_string("servable_model_dir", "", "export servable model for TensorFlow Serving")
flags.DEFINE_integer("feature_size", 1842, "Number of features")
flags.DEFINE_integer("field_size", 39, "Number of fields")
flags.DEFINE_integer("embedding_size", 10, "Embedding size")
flags.DEFINE_integer("num_epochs", 20, "Number of epochs")
flags.DEFINE_integer("batch_size", 64, "Number of batch size")
flags.DEFINE_integer("log_steps", 5000, "Save summary every steps")
flags.DEFINE_string("loss", "log_loss", "{log_loss, square_loss}")
flags.DEFINE_string("optimizer", "Adam", "{Adam, Adagrad, Momentum, Ftrl, GD}")
flags.DEFINE_float("learning_rate", 0.0005, "Learning rate")
flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
flags.DEFINE_string("deep_layers", "256,128,64", "deep layers")
flags.DEFINE_integer("cross_layers", 3, "cross layers, polynomial degree")
flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
flags.DEFINE_boolean("batch_norm", False, "perform batch normalization (True or False)")
flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(decay=0.9)")
FLAGS = flags.FLAGS


# 0 1:0.1 2:0.003322 3:0.44 4:0.02 5:0.001594 6:0.016 7:0.02
# 8:0.04 9:0.008 10:0.166667 11:0.1 12:0 13:0.08
# 16:1 54:1 77:1 93:1 112:1 124:1 128:1 148:1 160:1 162:1 176:1 209:1 227:1
# 264:1 273:1 312:1 335:1 387:1 395:1 404:1 407:1 427:1 434:1 443:1 466:1 479:1
def input_fn(filenames, batch_size=64, num_epochs=1, perform_shuffle=False):
    print('Parsing ----------- ', filenames)

    def dataset_etl(line):
        feat_raw = tf.string_split([line], ' ')
        labels = tf.string_to_number(feat_raw.values[0], out_type=tf.float32)
        splits = tf.string_split(feat_raw.values[1:], ':')
        idx_val = tf.reshape(splits.values, splits.dense_shape)
        feat_idx, feat_val = tf.split(idx_val, num_or_size_splits=2, axis=1)    # 切割张量
        feat_idx = tf.string_to_number(feat_idx, out_type=tf.int32)             # [field_size, 1]
        feat_val = tf.string_to_number(feat_val, out_type=tf.float32)           # [field_size, 1]
        return {"feat_idx": feat_idx, "feat_val": feat_val}, labels

    # extract lines from input files[one filename or filename list] using the Dataset API,
    # multi-thread pre-process then prefetch some certain amount of data[6400]
    dataset = tf.data.TextLineDataset(filenames).map(dataset_etl, num_parallel_calls=4).prefetch(6400)

    # randomize the input data with a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()      # [batch_size, field_size, 1]

    return batch_features, batch_labels


def model_fn(features, labels, mode, params):

    # ----- hyper-parameters ----- #
    l2_reg = params["l2_reg"]
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    learning_rate = params["learning_rate"]
    deep_layers = list(map(int, params["deep_layers"].split(',')))
    cross_layers = params['cross_layers']
    dropout = list(map(float, params["dropout"].split(',')))

    # ----- initial weights ----- #
    # [numeric_feature, one-hot categorical_feature]统一做embedding
    embed_w = tf.get_variable(name='embed_w', shape=[feature_size, embedding_size],
                              initializer=tf.glorot_normal_initializer())
    cross_b = tf.get_variable(name='cross_b', shape=[cross_layers, field_size*embedding_size],
                              initializer=tf.glorot_uniform_initializer())
    cross_w = tf.get_variable(name='cross_w', shape=[cross_layers, field_size*embedding_size],
                              initializer=tf.glorot_uniform_initializer())

    # ----- reshape feature ----- #
    feat_idx = features['feat_idx']         # 非零特征位置[batch_size, field_size, 1]
    feat_idx = tf.reshape(feat_idx, shape=[-1, field_size])     # [Batch, Field]
    feat_val = features['feat_val']         # 非零特征的值[batch_size, field_size, 1]
    feat_val = tf.reshape(feat_val, shape=[-1, field_size])     # [Batch, Field]

    # ----- define f(x) ----- #
    with tf.variable_scope("Embed-layer"):
        embeddings = tf.nn.embedding_lookup(embed_w, feat_idx)              # [Batch, Field, K]
        feat_vals = tf.reshape(feat_val, shape=[-1, field_size, 1])         # [Batch, Field, 1]
        embeddings = tf.multiply(embeddings, feat_vals)                     # [Batch, Field, K]
        x0 = tf.reshape(embeddings, shape=[-1, field_size*embedding_size])  # [Batch, Field*K]

    with tf.variable_scope("Cross-layer"):
        xl = x0
        for l in range(cross_layers):
            wl = tf.reshape(cross_w[l], shape=[-1, 1])      # [Field*K,1]
            xlw = tf.matmul(xl, wl)                         # [Batch, 1]
            xl = x0 * xlw + cross_b[l]                      # [Batch, Field*K]

    with tf.variable_scope("Deep-layer"):
        if FLAGS.batch_norm:
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
            else:
                train_phase = False
        else:
            train_phase = True

        deep_inputs = x0        # [Batch, Field*K]
        for i in range(len(deep_layers)):
            deep_inputs = tf.contrib.layers.fully_connected(
                inputs=deep_inputs, num_outputs=deep_layers[i], scope='mlp_%d' % i,
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            if FLAGS.batch_norm:
                # <<https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md>>
                # Batch normalization after Relu
                deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' % i)
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

    with tf.variable_scope("DCN-out"):
        x_stack = tf.concat([xl, deep_inputs], 1)
        y_deep = tf.contrib.layers.fully_connected(
            inputs=x_stack, num_outputs=1, activation_fn=tf.identity,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='comb_layer')
        y_hat = tf.reshape(y_deep, shape=[-1])                  # [Batch]
        y_pred = tf.nn.sigmoid(y_hat)                           # [Batch]

    # ----- mode: predict/evaluate/train ----- #
    # predict: 不计算loss/metric; evaluate: 不进行梯度下降和参数更新
    # Provide an estimator spec for 'ModeKeys.PREDICT'
    predictions = {"prob": y_pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.estimator.export.PredictOutput(predictions)}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Provide an estimator spec for 'ModeKeys.EVAL'
    if FLAGS.loss == "log_loss":
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_hat)) \
               + l2_reg * tf.nn.l2_loss(embed_w) + l2_reg * tf.nn.l2_loss(cross_b) \
               + l2_reg * tf.nn.l2_loss(cross_w)
    else:
        loss = tf.reduce_mean(tf.square(labels-y_pred)) + l2_reg * tf.nn.l2_loss(embed_w) \
               + l2_reg * tf.nn.l2_loss(cross_b) + l2_reg * tf.nn.l2_loss(cross_w)
    eval_metric_ops = {"auc": tf.metrics.auc(labels, y_pred)}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    # Provide an estimator spec for 'ModeKeys.TRAIN'
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'Ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


# Implementation of Batch normalization, train/infer
def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


# Initialized Distributed Environment,初始化分布式环境
def distributed_env_set():
    if FLAGS.run_mode == 1:         # 单机分布式
        ps_hosts = FLAGS.ps_hosts.split(',')
        chief_hosts = FLAGS.worker_hosts.split(',')
        job_name = FLAGS.job_name
        task_index = FLAGS.task_index
        print('ps_host --------', ps_hosts)
        print('chief_hosts ----', chief_hosts)
        print('job_name -------', job_name)
        print('task_index -----', str(task_index))
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
        worker_hosts = worker_hosts[1:]     # the rest as worker
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


# print initial information of paras,打印初始化参数信息
def _print_init_info(train_files, valid_files, tests_files):
    print('task_mode --------- ', FLAGS.task_mode)
    print('data_dir ---------- ', FLAGS.data_dir)
    print('model_dir --------- ', FLAGS.model_dir)
    print('mark_dir ---------- ', FLAGS.mark_dir)
    print('feature_size ------ ', FLAGS.feature_size)
    print('field_size -------- ', FLAGS.field_size)
    print('embedding_size ---- ', FLAGS.embedding_size)
    print('num_epochs -------- ', FLAGS.num_epochs)
    print('batch_size -------- ', FLAGS.batch_size)
    print('loss -------------- ', FLAGS.loss)
    print('optimizer --------- ', FLAGS.optimizer)
    print('learning_rate ----- ', FLAGS.learning_rate)
    print('l2_reg ------------ ', FLAGS.l2_reg)
    print('deep_layers ------- ', FLAGS.deep_layers)
    print('dropout ----------- ', FLAGS.dropout)
    print('batch_norm -------- ', FLAGS.batch_norm)
    print('batch_norm_decay -- ', FLAGS.batch_norm_decay)
    print("train_files: ", train_files)
    print("valid_files: ", valid_files)
    print("tests_files: ", tests_files)


def main(_):
    print('==================== 1.Check Arguments and Print Init Info...')
    if FLAGS.mark_dir == "":    # 存储算法模型文件目录[标记不同时刻训练模型,程序执行日期前一天:20190327]
        FLAGS.mark_dir = 'ch04_DCN_' + (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.mark_dir
    if FLAGS.data_dir == "":    # windows环境测试[未指定data目录条件下]
        root_dir = os.path.abspath(os.path.dirname(os.getcwd()))
        FLAGS.data_dir = root_dir + '\\criteo_data_set\\'

    train_files = glob.glob("%s/train*set" % FLAGS.data_dir)    # 获取指定目录下train文件
    random.shuffle(train_files)
    valid_files = glob.glob("%s/valid*set" % FLAGS.data_dir)    # 获取指定目录下valid文件
    tests_files = glob.glob("%s/tests*set" % FLAGS.data_dir)    # 获取指定目录下tests文件
    _print_init_info(train_files, valid_files, tests_files)

    print('==================== 2.Clear Existed Model and Initialized Distributed Environment...')
    if FLAGS.clr_mode:        # 删除已存在的模型文件
        try:
            shutil.rmtree(FLAGS.model_dir)      # 递归删除目录下的目录及文件
        except Exception as e:
            print(e, "At clear_existed_model")
        else:
            print("Existed model cleared at %s folder" % FLAGS.model_dir)
    distributed_env_set()       # 分布式环境设置

    print('==================== 3.Build DCN model...')
    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
        "dropout": FLAGS.dropout,
        "deep_layers": FLAGS.deep_layers,
        "cross_layers": FLAGS.cross_layers,
        "batch_norm_decay": FLAGS.batch_norm_decay
    }
    session_config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': FLAGS.num_threads})
    config = tf.estimator.RunConfig().replace(session_config=session_config,
                                              save_summary_steps=FLAGS.log_steps,
                                              log_step_count_steps=FLAGS.log_steps)
    dcn = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir,
                                 params=model_params, config=config)

    print('==================== 4.Apply DCN model...')
    train_step = 179968*FLAGS.num_epochs/FLAGS.batch_size       # data_num * num_epochs / batch_size
    if FLAGS.task_mode == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(train_files, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs),
            max_steps=train_step)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(valid_files, batch_size=FLAGS.batch_size, num_epochs=1),
            steps=None, start_delay_secs=200, throttle_secs=300)
        tf.estimator.train_and_evaluate(dcn, train_spec, eval_spec)
    elif FLAGS.task_mode == 'eval':
        dcn.evaluate(input_fn=lambda: input_fn(valid_files, batch_size=FLAGS.batch_size, num_epochs=1))
    elif FLAGS.task_mode == 'infer':
        preds = dcn.predict(
            input_fn=lambda: input_fn(tests_files, batch_size=FLAGS.batch_size, num_epochs=1),
            predict_keys="prob")
        with open(FLAGS.data_dir+"/tests_pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_mode == 'export':
        feature_spec = {
            'feat_idx': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_idx'),
            'feat_val': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_val')}
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        dcn.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
