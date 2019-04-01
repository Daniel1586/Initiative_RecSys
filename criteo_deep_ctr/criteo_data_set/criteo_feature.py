#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Preprocess Criteo dataset. This dataset was used for the Display Advertising
Challenge (https://www.kaggle.com/c/criteo-display-ad-challenge).
----数据解压: train.csv=45840617条样本, test.csv=6042135条样本
----这里train只取20W 条数据, test只取4W 条数据测试
This code is referenced from PaddlePaddle models.
(https://github.com/PaddlePaddle/models/blob/develop/legacy/deep_fm/preprocess.py)
--For numeric features, clipped and normalized.
--For categorical features, removed long-tailed data appearing less than 200 times.
########## TF Version: 1.8.0 ##########
"""

import os
import sys
import random
import argparse
import collections

# There are 13 numeric features and 26 categorical features
# 数值特征I1-I13(整数), 离散特征C1-C26
numeric_features = range(1, 14)
categorical_features = range(14, 40)

# Clip numeric features. The clip point for each numeric feature
# is derived from the 95% quantile of the total values in each feature
# 数值特征的阈值,若数值特征超过阈值,则该特征值置为阈值[剔除异常值]
numeric_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorical_feature, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                # 遍历离散特征,统计不同离散特征值出现次数
                for i in range(0, self.num_feature):
                    if features[categorical_feature[i]] != '':
                        self.dicts[i][features[categorical_feature[i]]] += 1

        for j in range(0, self.num_feature):
            # 剔除频次小于cutoff的离散特征,剩下特征按频次从大到小排序
            temp_list = filter(lambda x: x[1] >= cutoff, self.dicts[j].items())
            sort_list = sorted(temp_list, key=lambda x: (-x[1], x[0]))
            # 符合条件的离散特征, 编号1:len()-1, 不符合条件的特征编号为0
            vocabs, _ = list(zip(*sort_list))
            tran_dict = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[j] = tran_dict
            self.dicts[j]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return map(len, self.dicts)


class NumericFeatureGenerator:
    """
    Normalize the numeric features to [0, 1] by min-max normalization
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature
        self.max = [-sys.maxsize] * num_feature

    def build(self, datafile, numeric_feature):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    val = features[numeric_feature[i]]
                    if val != '':
                        val = int(val)
                        if val > numeric_clip[i]:
                            val = numeric_clip[i]
                        self.min[i] = min(self.min[i], val)
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])


def preprocess(datadir, outdir):
    """
    All the 13 numeric(integer) features are normalized to [0,1] and these
    numeric features are combined into one vector with dimension 13.
    Each of the 26 categorical features are one-hot encoded and all the one-hot
    vectors are combined into one sparse binary vector.
    """

    print('========== 1.Preprocess numeric and categorical features...')
    dists = NumericFeatureGenerator(len(numeric_features))
    dists.build(FLAGS.input_dir + 'train.txt', numeric_features)

    dicts = CategoryDictGenerator(len(categorical_features))
    dicts.build(FLAGS.input_dir + 'train.txt', categorical_features, cutoff=FLAGS.cutoff)

    print('========== 2.Generate index of feature embedding ...')
    # 生成数值特征编号: I1-I13
    output = open(FLAGS.output_dir + 'feature_embedding', 'w')
    for i in numeric_features:
        output.write("{0} {1}\n".format('I'+str(i), i))

    dict_sizes = list(dicts.dicts_sizes())
    categorical_feature_offset = [dists.num_feature]
    # 生成离散特征编号: C1|xxxx XX (不同离散特征第一个特征编号的特征统一为<unk>)
    for i in range(1, len(categorical_features)+1):
        offset = categorical_feature_offset[i - 1] + dict_sizes[i - 1]
        categorical_feature_offset.append(offset)
        for key, val in dicts.dicts[i-1].items():
            output.write("{0} {1}\n".format('C'+str(i)+'|'+key, categorical_feature_offset[i - 1]+val+1))

    random.seed(0)

    # 90% data are used for training, and 10% data are used for validation.
    print('========== 3.Generate train/valid/test dataset ...')
    with open(FLAGS.output_dir + 'train.set', 'w') as out_train:
        with open(FLAGS.output_dir + 'valid.set', 'w') as out_valid:
            with open(FLAGS.input_dir + 'train.txt', 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split('\t')

                    feat_vals = []
                    # numeric features normalized to [0,1]
                    for i in range(0, len(numeric_features)):
                        val = dists.gen(i, features[numeric_features[i]])
                        feat_vals.append(str(numeric_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                    # categorical features one-hot embedding
                    for i in range(0, len(categorical_features)):
                        val = dicts.gen(i, features[categorical_features[i]]) + categorical_feature_offset[i] + 1
                        feat_vals.append(str(val) + ':1')

                    label = features[0]
                    if random.randint(0, 9999) % 10 != 0:
                        out_train.write("{0} {1}\n".format(label, ' '.join(feat_vals)))
                    else:
                        out_valid.write("{0} {1}\n".format(label, ' '.join(feat_vals)))

    with open(FLAGS.output_dir + 'tests.set', 'w') as out_test:
        with open(FLAGS.input_dir + 'test.txt', 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                feat_vals = []
                for i in range(0, len(numeric_features)):
                    val = dists.gen(i, features[numeric_features[i] - 1])
                    feat_vals.append(str(numeric_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                for i in range(0, len(categorical_features)):
                    val = dicts.gen(i, features[categorical_features[i] - 1]) + categorical_feature_offset[i] + 1
                    feat_vals.append(str(val) + ':1')

                label = 0   # test fake label
                out_test.write("{0} {1}\n".format(label, ' '.join(feat_vals)))


if __name__ == "__main__":
    run_mode = 0        # 0: windows环境测试
    if run_mode == 0:
        root_dir = os.path.abspath(os.path.dirname(os.getcwd()))
        data_dir = root_dir + '\\criteo_data_raw\\'
        outs_dir = root_dir + '\\criteo_data_set\\'
    else:
        data_dir = ''
        outs_dir = ''

    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2, help="threads num")
    parser.add_argument("--input_dir", type=str, default=data_dir, help="input data dir")
    parser.add_argument("--output_dir", type=str, default=outs_dir, help="feature map output dir")
    parser.add_argument("--cutoff", type=int, default=200, help="cutoff long-tailed categorical values")

    FLAGS, unparsed = parser.parse_known_args()
    print('threads -------------- ', FLAGS.threads)
    print('input_dir ------------ ', FLAGS.input_dir)
    print('output_dir ----------- ', FLAGS.output_dir)
    print('cutoff --------------- ', FLAGS.cutoff)

    preprocess(FLAGS.input_dir, FLAGS.output_dir)
