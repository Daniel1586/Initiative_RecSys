#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Implementation of CTR model with the following features：
#1 Input pipeline using Dataset API, Support parallel and prefetch.
#2 Train pipeline using Custom Estimator by rewriting model_fn.
#3 Support distributed training by TF_CONFIG.
#4 Support export_model for TensorFlow Serving.
############### TF Version: 1.13.1/Python Version: 3.7 ###############
"""

import os
import json
import glob
import random
import shutil
import tensorflow as tf
from datetime import date, timedelta
from tensorflow_estimator import estimator
from ctr_model import lr, fm

# =================== CMD Arguments for CTR model =================== #
flags = tf.app.flags
flags.DEFINE_integer("run_mode", 0, "{0-local, 1-single_distributed, 2-multi_distributed}")
flags.DEFINE_string("ps_hosts", None, "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("wk_hosts", None, "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "Job name: ps or worker")
flags.DEFINE_integer("task_id", None, "Index of task within the job")
flags.DEFINE_integer("num_thread", 4, "Number of threads")
# global parameters--全局参数设置
flags.DEFINE_string("algorithm", "FM", "{LR, FM, ., .}")
flags.DEFINE_string("task_mode", "train", "{train, eval, infer, export}")
flags.DEFINE_string("input_dir", "", "Input data dir")
flags.DEFINE_string("model_dir", "", "Model check point file dir")
flags.DEFINE_string("serve_dir", "", "Export servable model for TensorFlow Serving")
flags.DEFINE_string("clear_mod", "True", "{True, False},Clear existed model or not")
flags.DEFINE_integer("log_steps", 2000, "Save summary every steps")
# model parameters--模型参数设置
flags.DEFINE_integer("samples_size", 269738, "Number of train samples")
flags.DEFINE_integer("feature_size", 2829, "Number of features[numeric + one-hot categorical_feature]")
flags.DEFINE_integer("field_size", 39, "Number of fields")
flags.DEFINE_integer("embed_size", 16, "Embedding size[length of hidden vector of xi/xj]")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
flags.DEFINE_integer("batch_size", 256, "Number of batch size")
flags.DEFINE_string("loss_mode", "log_loss", "{log_loss, square_loss}")
flags.DEFINE_string("optimizer", "Adam", "{Adam, Adagrad, Momentum, Ftrl, GD}")
flags.DEFINE_float("learning_rate", 0.0005, "Learning rate")
flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularization")
FLAGS = flags.FLAGS


# 0 1:0.1 2:0.003322 3:0.44 4:0.02 5:0.001594 6:0.016 7:0.02
# 8:0.04 9:0.008 10:0.166667 11:0.1 12:0 13:0.08
# 16:1 54:1 77:1 93:1 112:1 124:1 128:1 148:1 160:1 162:1 176:1 209:1 227:1
# 264:1 273:1 312:1 335:1 387:1 395:1 404:1 407:1 427:1 434:1 443:1 466:1 479:1
def input_fn(filenames, batch_size=64, num_epochs=1, perform_shuffle=True):
    print("Parsing ----------- ", filenames)

    def dataset_etl(line):
        feat_raw = tf.string_split([line], " ")
        labels = tf.string_to_number(feat_raw.values[0], out_type=tf.float32)
        splits = tf.string_split(feat_raw.values[1:], ":")
        idx_val = tf.reshape(splits.values, splits.dense_shape)
        feat_idx, feat_val = tf.split(idx_val, num_or_size_splits=2, axis=1)    # 切割张量
        feat_idx = tf.string_to_number(feat_idx, out_type=tf.int32)             # [field_size, 1]
        feat_val = tf.string_to_number(feat_val, out_type=tf.float32)           # [field_size, 1]
        return {"feat_idx": feat_idx, "feat_val": feat_val}, labels

    # extract lines from input files[filename or filename list] using the Dataset API,
    # multi-thread pre-process then prefetch some certain amount of data[6400]
    dataset = tf.data.TextLineDataset(filenames).map(dataset_etl, num_parallel_calls=4).prefetch(9600)

    # randomize the input data with a window of 512 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=1024)

    # epochs from blending together
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()      # [batch_size, field_size, 1]

    return batch_features, batch_labels


# Initialized Distributed Environment,初始化分布式环境
def distr_env_set():
    if FLAGS.run_mode == 1:         # 单机分布式
        ps_hosts = FLAGS.ps_hosts.split(',')
        cf_hosts = FLAGS.wk_hosts.split(',')
        job_name = FLAGS.job_name
        task_idx = FLAGS.task_id
        print("ps_host --------", ps_hosts)
        print("chief_hosts ----", cf_hosts)
        print("job_name -------", job_name)
        print("task_index -----", str(task_idx))
        # 无worker参数
        tf_config = {
            "cluster": {"chief": cf_hosts, "ps": ps_hosts},
            "task": {"type": job_name, "index": task_idx}
        }
        print(json.dumps(tf_config))
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
    elif FLAGS.run_mode == 2:      # 集群分布式
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.wk_hosts.split(',')
        cf_hosts = worker_hosts[0:1]    # get first worker as chief
        wk_hosts = worker_hosts[1:]     # the rest as worker
        task_idx = FLAGS.task_id
        job_name = FLAGS.job_name
        print("ps_host --------", ps_hosts)
        print("chief_hosts ----", cf_hosts)
        print("worker_host ----", wk_hosts)
        print("job_name -------", job_name)
        print("task_index -----", str(task_idx))
        # use #worker=0 as chief
        if job_name == "worker" and task_idx == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        if job_name == "worker" and task_idx == 1:
            job_name = "evaluator"
            task_idx = 0
        # the others as worker
        if job_name == "worker" and task_idx > 1:
            task_idx -= 2

        tf_config = {
            "cluster": {"chief": cf_hosts, "worker": wk_hosts, "ps": ps_hosts},
            "task": {"type": job_name, "index": task_idx}
        }
        print(json.dumps(tf_config))
        os.environ["TF_CONFIG"] = json.dumps(tf_config)


def main(_):
    print("==================== 1.Check Args and Initialized Distributed Env...")
    if FLAGS.model_dir == "":       # 算法模型checkpoint文件
        FLAGS.model_dir = (date.today() + timedelta(-1)).strftime("%Y%m%d") + "_Ckt_" + FLAGS.algorithm
    if FLAGS.serve_dir == "":       # 算法模型输出pb文件
        FLAGS.serve_dir = (date.today() + timedelta(-1)).strftime("%Y%m%d") + "_Exp_" + FLAGS.algorithm
    if FLAGS.input_dir == "":       # windows环境测试
        FLAGS.input_dir = os.path.dirname(os.getcwd()) + "\\data" + "\\data_set_criteo\\"

    train_files = glob.glob("%s/train*set" % FLAGS.input_dir)       # 获取指定目录下train文件
    valid_files = glob.glob("%s/valid*set" % FLAGS.input_dir)       # 获取指定目录下valid文件
    tests_files = glob.glob("%s/tests*set" % FLAGS.input_dir)       # 获取指定目录下tests文件
    random.shuffle(train_files)                                     # 打散train文件

    if FLAGS.clear_mod == "True" and FLAGS.task_mode == "train":    # 删除已存在的模型文件
        try:
            shutil.rmtree(FLAGS.model_dir)      # 递归删除目录下的目录及文件
        except Exception as e:
            print(e, "At clear_existed_model")
        else:
            print("Existed model cleared at %s folder" % FLAGS.model_dir)
    distr_env_set()       # 分布式环境设置

    print("==================== 2.Set model params and Build CTR model...")
    model_params = {
        "feature_size": FLAGS.feature_size,
        "field_size": FLAGS.field_size,
        "embed_size": FLAGS.embed_size,
        "loss_mode": FLAGS.loss_mode,
        "optimizer": FLAGS.optimizer,
        "learning_rate": FLAGS.learning_rate,
        "l2_reg_lambda": FLAGS.l2_reg_lambda
    }
    if FLAGS.algorithm == "LR":
        model_fn = lr
    elif FLAGS.algorithm == "FM":
        model_fn = fm
    else:
        model_fn = None
        print("Invalid algorithm, not supported!")

    batch_num = int(FLAGS.samples_size/FLAGS.batch_size)
    train_step = batch_num * FLAGS.num_epochs       # data_num * num_epochs / batch_size
    session_config = tf.ConfigProto(device_count={"GPU": 1, "CPU": FLAGS.num_thread})
    config = estimator.RunConfig(session_config=session_config,
                                 save_checkpoints_steps=batch_num,
                                 save_summary_steps=FLAGS.log_steps,
                                 log_step_count_steps=FLAGS.log_steps)
    ctr = estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir,
                              params=model_params, config=config)

    print("==================== 3.Apply CTR model to diff tasks...")
    if FLAGS.task_mode == "train":
        train_spec = estimator.TrainSpec(
            input_fn=lambda: input_fn(train_files, FLAGS.batch_size, FLAGS.num_epochs, True),
            max_steps=train_step)
        eval_spec = estimator.EvalSpec(
            input_fn=lambda: input_fn(valid_files, FLAGS.batch_size, 1, False), steps=None,
            start_delay_secs=50, throttle_secs=20)
        estimator.train_and_evaluate(ctr, train_spec, eval_spec)
    elif FLAGS.task_mode == "eval":
        ctr.evaluate(input_fn=lambda: input_fn(valid_files, FLAGS.batch_size, 1, False))
    elif FLAGS.task_mode == "infer":
        preds = ctr.predict(
            input_fn=lambda: input_fn(tests_files, FLAGS.batch_size, 1, False), predict_keys="prob")
        with open(FLAGS.input_dir+"/pred_tests.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_mode == "export":
        feature_spec = {
            "feat_idx": tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name="feat_idx"),
            "feat_val": tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name="feat_val")}
        serving_input_receiver_fn = estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        ctr.export_savedmodel(FLAGS.serve_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
