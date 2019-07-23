#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from dataReader import FeatureDictionary, DataParser
from matplotlib import pyplot as plt

import config
from DeepCrossNet import DeepCrossNet


def _load_data():
    dftrain = pd.read_csv(config.TRAIN_FILE)    # 595212*59,有‘target’项
    dftests = pd.read_csv(config.TEST_FILE)     # 892816*58,无‘target’项

    def preprocess(df):
        cols_ = [c for c in df.columns if c not in ['id', 'target']]
        # 计算每个样本缺失值个数
        df["missed_feature_num"] = np.sum((df[cols_] == -1).values, axis=1)
        df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
        return df

    dftrain = preprocess(dftrain)
    dftests = preprocess(dftests)

    cols = [c for c in dftrain.columns if c not in ['id', 'target']]
    cols = [c for c in cols if c not in config.IGNORE_COLS]

    _x_train = dftrain[cols].values
    _y_train = dftrain['target'].values
    _x_tests = dftests[cols].values
    _id_test = dftests['id'].values

    return dftrain, dftests, _x_train, _y_train, _x_tests, _id_test


def run_base_model_dcn(dftrain, dftests, folds, dcn_para):
    # dataset数据集处理,得到特征的有效key值和one-hot之后的特征个数
    fd = FeatureDictionary(dftrain=dftrain, dftests=dftests,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS,
                           cate_cols=config.CATEGORICAL_COLS)

    data_parser = DataParser(feat_dict=fd)
    # cat_xi_train: categorical非零特征位置;cat_xv_train:categorical非零特征的值
    # num_xv_train: numeric特征值(train);num_xv_tests:numeric特征值(test)
    # _y_train: 训练数据label;_id_test: 测试数据id
    cat_xi_train, cat_xv_train, num_xv_train, _y_train = data_parser.parse(df=dftrain, has_label=True)
    cat_xi_tests, cat_xv_tests, num_xv_tests, _id_test = data_parser.parse(df=dftests)

    dcn_para['cate_feature_size'] = fd.feature_dim
    dcn_para['field_size'] = len(cat_xi_train[0])
    dcn_para['numeric_feature_size'] = len(config.NUMERIC_COLS)

    y_tests_meta = np.zeros((dftests.shape[0], 1), dtype=float)

    _get = lambda x, l: [x[i1] for i1 in l]
    gini_results_epoch_train = np.zeros((len(folds), dcn_para['epoch']), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dcn_para['epoch']), dtype=float)

    # 遍历K折分层采样的train data
    for i, (train_idx, valid_idx) in enumerate(folds):
        # 计算每次使用的train/valid
        xi_train_cat = _get(cat_xi_train, train_idx)
        xv_train_cat = _get(cat_xv_train, train_idx)
        xv_train_num = _get(num_xv_train, train_idx)
        y_train_ = _get(_y_train, train_idx)
        xi_valid_cat = _get(cat_xi_train, valid_idx)
        xv_valid_cat = _get(cat_xv_train, valid_idx)
        xv_valid_num = _get(num_xv_train, valid_idx)
        y_valid_ = _get(_y_train, valid_idx)

        # 模型建立/训练/预测
        dcn = DeepCrossNet(**dcn_para)
        dcn.fit(xi_train_cat, xv_train_cat, xv_train_num, y_train_,
                xi_valid_cat, xv_valid_cat, xv_valid_num, y_valid_)

        y_tests_meta += dcn.predict(cat_xi_tests, cat_xv_tests, num_xv_tests)
        gini_results_epoch_train[i] = dcn.train_result
        gini_results_epoch_valid[i] = dcn.valid_result

    y_tests_meta /= float(len(folds))

    # save result
    clf_str = "DeepCrossNet"
    print("%s: %d Epoch" % (clf_str, dcn_para['epoch']))
    filename = "%s_%dEpoch.csv" % (clf_str, dcn_para['epoch'])
    _make_submission(_id_test, y_tests_meta, filename)

    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_tests_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d" % (i+1))
        legends.append("valid-%d" % (i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("%s" % model_name)
    plt.legend(legends)
    plt.savefig("fig/%s.png" % model_name)
    plt.close()


# =============== DeepCrossNet模型参数设置 =============== #
dcn_params = {
    "embedding_size": 8,
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layer_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "cross_layer_num": 3,
    "random_seed": config.RANDOM_SEED
}

print('========== 1.Loading dataset...')
dfTrain, dfTests, x_train, y_train, x_test, ids_test = _load_data()

print('========== 2.Stratifying dataset...')
# 对train data进行分层采样,确保训练集各类别样本比例与原始数据集相同
kfolds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                              random_state=config.RANDOM_SEED).split(x_train, y_train))

print('========== 3.Running base model DCN...')
y_test_dcn = run_base_model_dcn(dfTrain, dfTests, kfolds, dcn_params)
