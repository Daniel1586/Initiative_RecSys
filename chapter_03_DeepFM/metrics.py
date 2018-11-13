#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def gini(actual, pred):
    assert (len(actual) == len(pred))
    tmp = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    tmp = tmp[np.lexsort((tmp[:, 2], -1 * tmp[:, 1]))]
    total_losses = tmp[:, 0].sum()

    ginisum = tmp[:, 0].cumsum().sum() / total_losses
    ginisum -= (len(actual) + 1) / 2.

    return ginisum / len(actual)


def gini_norm(actual, pred):
    return gini(actual, pred) / gini(actual, actual)
