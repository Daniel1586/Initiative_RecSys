#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Preprocess Criteo dataset. This dataset was used for the Display Advertising
Challenge (https://www.kaggle.com/c/criteo-display-ad-challenge).
----数据解压: train.csv=45840617条样本, test.csv=6042135条样本
----这里train只取20W 条数据, test只取4W 条数据测试
----train.txt/test.txt-->train.csv/valid.csv/tests.csv
This code is referenced from PaddlePaddle models.
(https://github.com/PaddlePaddle/models/blob/develop/legacy/deep_fm/preprocess.py)
--For numeric features, clipped and normalized.
--For categorical features, removed long-tailed data appearing less than 200 times.
########## TF Version: 1.8.0 ##########
"""

import os
import random
import argparse

# There are 13 numeric features and 26 categorical features
# 数值特征I1-I13(整数), 离散特征C1-C26
col_features = range(1, 40)


def preprocess(datadir, outdir):

    # 90% data are used for training, and 10% data are used for validation.
    print('========== 1.Generate train/valid/test dataset ...')
    with open(outdir + 'train.csv', 'w') as out_train:
        with open(outdir + 'valid.csv', 'w') as out_valid:
            with open(datadir + 'train.txt', 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split('\t')
                    for i in col_features:
                        if features[i] == '':
                            features[i] = str(0)

                    label = features[0]
                    if random.randint(0, 9999) % 10 != 0:
                        out_train.write("{0},{1}\n".format(label, ','.join(features[1:])))
                    else:
                        out_valid.write("{0},{1}\n".format(label, ','.join(features[1:])))

    with open(outdir + 'tests.csv', 'w') as out_test:
        with open(datadir + 'test.txt', 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in col_features:
                    if features[i-1] == '':
                        features[i-1] = str(0)

                label = 0       # test fake label
                out_test.write("{0},{1}\n".format(label, ','.join(features)))


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

    FLAGS, unparsed = parser.parse_known_args()
    print('threads -------------- ', FLAGS.threads)
    print('input_dir ------------ ', FLAGS.input_dir)
    print('output_dir ----------- ', FLAGS.output_dir)

    preprocess(FLAGS.input_dir, FLAGS.output_dir)
