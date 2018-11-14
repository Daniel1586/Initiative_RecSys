#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd


class FeatureDictionary(object):
    def __init__(self, trainfile=None, testsfile=None,
                 dftrain=None, dftests=None, numeric_cols=None, ignore_cols=None):
        assert not ((trainfile is None) and (dftrain is None)), "trainfile or dftrain at least one is set"
        assert not ((trainfile is not None) and (dftrain is not None)), "trainfile or dftrain only one can be set"
        assert not ((testsfile is None) and (dftests is None)), "testsfile or dftests at least one is set"
        assert not ((testsfile is not None) and (dftests is not None)), "testsfile or dftests only one can be set"
        if numeric_cols is None:
            numeric_cols = []
        if ignore_cols is None:
            ignore_cols = []

        self.trainfile = trainfile
        self.testsfile = testsfile
        self.dftrain = dftrain
        self.dftests = dftests
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.feature_dict = {}
        self.feature_dim = 0
        self.gen_feature_dict()

    def gen_feature_dict(self):
        if self.dftrain is None:
            dftrain = pd.read_csv(self.trainfile)
        else:
            dftrain = self.dftrain

        if self.dftests is None:
            dftests = pd.read_csv(self.testsfile)
        else:
            dftests = self.dftests

        df = pd.concat([dftrain, dftests], sort=False)

        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                # numeric特征
                self.feature_dict[col] = tc
                tc += 1
            else:
                # categorical特征,需要进行one-hot编码
                us = df[col].unique()
                self.feature_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
                tc += len(us)
        self.feature_dim = tc


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"

        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)

        if has_label:
            # train data删除columns=['id','target'],inplace=True表示在原始数据操作
            y = dfi['target'].values.tolist()
            dfi.drop(['id', 'target'], axis=1, inplace=True)
        else:
            # test data删除columns=['id'],inplace=True表示在原始数据操作
            ids = dfi['id'].values.tolist()
            dfi.drop(['id'], axis=1, inplace=True)
        # dfi for feature index,特征的索引位置
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feature_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feature_dict[col])
                dfv[col] = 1.

        xi = dfi.values.tolist()
        xv = dfv.values.tolist()
        if has_label:
            return xi, xv, y
        else:
            return xi, xv, ids
