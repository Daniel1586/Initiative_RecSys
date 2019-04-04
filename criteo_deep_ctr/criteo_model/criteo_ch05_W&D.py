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
flags.DEFINE_integer("embedding_size", 10, "Embedding size")
flags.DEFINE_integer("num_epochs", 20, "Number of epochs")
flags.DEFINE_integer("batch_size", 64, "Number of batch size")
flags.DEFINE_integer("log_steps", 5000, "Save summary every steps")
flags.DEFINE_string("deep_layers", "256,128,64", "deep layers")
FLAGS = flags.FLAGS

# There are 13 integer features and 26 categorical features
numeric_cols = ['I' + str(i) for i in range(1, 14)]
categoricals = ['C' + str(i) for i in range(14, 40)]
label_column = "clicked"
csv_columns = [label_column] + numeric_cols + categoricals
# Columns Defaults
csv_column_defaults = [[0.0]]
numeric_cols_default = [[0.0] for i in range(13)]
categoricals_default = [[0] for i in range(26)]
csv_column_defaults = csv_column_defaults + numeric_cols_default + categoricals_default
print(csv_column_defaults)


# There are 13 numeric features and 26 categorical features
# 0	1	1	5	0	1382	4	15	2	181	1	2	null	2
# 68fd1e64	80e26c9b	fb936136	7b4723c4	25c83c98	7e0ccccf	de7995b8
# 1f89b562	a73ee510	a8cd5504	b2cb9c98	37c9c164	2824a5f6	1adce6ef
# 8ba8b39a	891b62e7	e5ba7672	f54016b9	21ddcdc9	b1252a9d	07b5194c
# null      3a171ecb	c5c50484	e8b83407	9727dd16
def input_fn(filenames, batch_size=64, num_epochs=1, perform_shuffle=False):
    print('Parsing ----------- ', filenames)

    def datacsv_etl(line):
        columns = tf.decode_csv(line, record_defaults=csv_column_defaults)
        features = dict(zip(csv_columns, columns))
        labels = features.pop(label_column)
        return features, labels

    # extract lines from input files[one filename or filename list] using the Dataset API,
    # multi-thread pre-process then prefetch some certain amount of data[6400]
    dataset = tf.data.TextLineDataset(filenames).map(datacsv_etl, num_parallel_calls=4).prefetch(6400)

    # randomize the input data with a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()      # [batch_size, field_size, 1]

    return batch_features, batch_labels


def build_feature():
    # 1 numeric_feature columns
    deep_nfc = [tf.feature_column.numeric_column(col) for col in numeric_cols]

    # 2 categorical_feature columns
    deep_cfc = [tf.feature_column.categorical_column_with_identity(key=col, num_buckets=10000, default_value=0) for
                col in categoricals]

    # 3 embedding columns
    deep_emb = [tf.feature_column.embedding_column(c, dimension=FLAGS.embedding_size) for c in deep_cfc]

    # 4 wide columns and deep columns
    wide_columns = deep_nfc + deep_cfc
    deep_columns = deep_nfc + deep_emb

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
    print('model_type -------- ', FLAGS.model_type)
    print('data_dir ---------- ', FLAGS.data_dir)
    print('model_dir --------- ', FLAGS.model_dir)
    print('mark_dir ---------- ', FLAGS.mark_dir)
    print('embedding_size ---- ', FLAGS.embedding_size)
    print('num_epochs -------- ', FLAGS.num_epochs)
    print('batch_size -------- ', FLAGS.batch_size)
    print('deep_layers ------- ', FLAGS.deep_layers)
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

    train_files = glob.glob("%s/train*csv" % FLAGS.data_dir)    # 获取指定目录下train文件
    random.shuffle(train_files)
    valid_files = glob.glob("%s/valid*csv" % FLAGS.data_dir)    # 获取指定目录下valid文件
    tests_files = glob.glob("%s/tests*csv" % FLAGS.data_dir)    # 获取指定目录下tests文件
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
    wide_columns, deep_columns = build_feature()

    hidden_units = map(int, FLAGS.deep_layers.split(","))
    session_config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': FLAGS.num_threads})
    config = tf.estimator.RunConfig().replace(session_config=session_config,
                                              save_summary_steps=FLAGS.log_steps,
                                              log_step_count_steps=FLAGS.log_steps)
    if FLAGS.model_type == "wide":
        w_d = tf.estimator.LinearClassifier(model_dir=FLAGS.model_dir, feature_columns=wide_columns,
                                            config=config)
    elif FLAGS.model_type == "deep":
        w_d = tf.estimator.DNNClassifier(model_dir=FLAGS.model_dir, feature_columns=deep_columns,
                                         hidden_units=hidden_units, config=config)
    else:
        w_d = tf.estimator.DNNLinearCombinedClassifier(model_dir=FLAGS.model_dir,
                                                       linear_feature_columns=wide_columns,
                                                       dnn_feature_columns=deep_columns,
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
                fo.write("%f\n" % (prob['prob'][1]))
    elif FLAGS.task_mode == 'export':
        if FLAGS.model_type == "wide":
            feature_columns = wide_columns
        elif FLAGS.model_type == "deep":
            feature_columns = deep_columns
        elif FLAGS.model_type == "wide_deep":
            feature_columns = wide_columns + deep_columns
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        w_d.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
