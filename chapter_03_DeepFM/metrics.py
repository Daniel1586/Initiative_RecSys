#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def gini(actual, pred):
    assert (len(actual) == len(pred))
    tmp = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    # lexsort排字典序函数,返回索引:优先使用后面排序(从小到大)
    # 按pred从大到小排列
    tmp = tmp[np.lexsort((tmp[:, 2], -1 * tmp[:, 1]))]
    # 总的真实值
    total_losses = tmp[:, 0].sum()
    # 总的真实值累计求和/总的真实值
    ginisum = tmp[:, 0].cumsum().sum() / total_losses

    ginisum -= (len(actual) + 1) / 2.

    return ginisum / len(actual)


def gini_norm(actual, pred):
    # Gini,基尼系数
    gini_coeff = gini(actual, pred)
    # Max Gini,最大可能基尼系数
    gini_coeff_max = gini(actual, actual)
    # Normalized Gini
    gini_normalized = gini_coeff/gini_coeff_max

    return gini_normalized
