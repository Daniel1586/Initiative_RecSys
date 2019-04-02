#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
<<Wide&Deep: Wide & Deep Learning for Recommender Systems>>
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

# =================== CMD Arguments for W&D model =================== #
flags = tf.app.flags
flags.DEFINE_integer("run_mode", 0, "{0-local, 1-single_distributed, 2-multi_distributed}")
flags.DEFINE_boolean("clr_mode", True, "Clear existed model or not")
flags.DEFINE_string("task_mode", "train", "{train, infer, eval, export}")
flags.DEFINE_string("model_type", "wide", "model type {wide, deep, wide_deep}")
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
FLAGS = flags.FLAGS

# There are 13 integer features and 26 categorical features
C_COLUMNS = ['I' + str(i) for i in range(1, 14)]
D_COLUMNS = ['C' + str(i) for i in range(14, 40)]
LABEL_COLUMN = "is_click"
CSV_COLUMNS = [LABEL_COLUMN] + C_COLUMNS + D_COLUMNS
# Columns Defaults
CSV_COLUMN_DEFAULTS = [[0.0]]
C_COLUMN_DEFAULTS = [[0.0] for i in range(13)]
D_COLUMN_DEFAULTS = [[0] for i in range(26)]
CSV_COLUMN_DEFAULTS = CSV_COLUMN_DEFAULTS + C_COLUMN_DEFAULTS + D_COLUMN_DEFAULTS
print(CSV_COLUMN_DEFAULTS)


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


def build_feature():
    # numeric_feature
    # 1 { continuous base columns }
    deep_cbc = [tf.feature_column.numeric_column(colname) for colname in C_COLUMNS]

    # 2 { categorical base columns }
    deep_dbc = [tf.feature_column.categorical_column_with_identity(key=colname, num_buckets=10000, default_value=0) for
                colname in D_COLUMNS]

    # 3 { embedding columns }
    deep_emb = [tf.feature_column.embedding_column(c, dimension=FLAGS.embedding_size) for c in deep_dbc]

    # 3 { wide columns and deep columns }
    wide_columns = deep_cbc + deep_dbc
    deep_columns = deep_cbc + deep_emb

    return wide_columns, deep_columns


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
    print("train_files: ", train_files)
    print("valid_files: ", valid_files)
    print("tests_files: ", tests_files)


def main(_):
    print('==================== 1.Check Arguments and Print Init Info...')
    if FLAGS.mark_dir == "":    # 存储算法模型文件目录[标记不同时刻训练模型,程序执行日期前一天:20190327]
        FLAGS.mark_dir = 'ch05_W_D_' + (date.today() + timedelta(-1)).strftime('%Y%m%d')
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

    print('==================== 3.Build W&D model...')
    hidden_units = map(int, FLAGS.deep_layers.split(","))
    session_config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': FLAGS.num_threads})
    config = tf.estimator.RunConfig().replace(session_config=session_config,
                                              save_summary_steps=FLAGS.log_steps,
                                              log_step_count_steps=FLAGS.log_steps)

    if FLAGS.model_type == "wide":
        w_d = tf.estimator.LinearClassifier(model_dir=FLAGS.model_dir, feature_columns=0,
                                            config=config)
    elif FLAGS.model_type == "deep":
        w_d = tf.estimator.DNNClassifier(model_dir=FLAGS.model_dir, feature_columns=0,
                                         hidden_units=hidden_units, config=config)
    else:
        w_d = tf.estimator.DNNLinearCombinedClassifier(model_dir=FLAGS.model_dir,
                                                       linear_feature_columns=0,
                                                       dnn_feature_columns=0,
                                                       dnn_hidden_units=hidden_units,
                                                       config=config)

    print('==================== 4.Apply W&D model...')
    train_step = 179968*FLAGS.num_epochs/FLAGS.batch_size       # data_num * num_epochs / batch_size
    if FLAGS.task_mode == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(train_files, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs),
            max_steps=train_step)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(valid_files, batch_size=FLAGS.batch_size, num_epochs=1),
            steps=None, start_delay_secs=200, throttle_secs=300)
        tf.estimator.train_and_evaluate(w_d, train_spec, eval_spec)
    elif FLAGS.task_mode == 'eval':
        w_d.evaluate(input_fn=lambda: input_fn(valid_files, batch_size=FLAGS.batch_size, num_epochs=1))
    elif FLAGS.task_mode == 'infer':
        preds = w_d.predict(
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
        w_d.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
