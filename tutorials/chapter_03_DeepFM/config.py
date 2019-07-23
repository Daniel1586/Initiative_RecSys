#!/usr/bin/python
# -*- coding: utf-8 -*-

# Kaggle比赛项目: Porto Seguro’s Safe Driver Prediction
# 根据汽车保单持有人数据建立ML模型,预测该持有人是否会在次年提出索赔
# train: 595212*59,有‘target’项;特征列共57列
# test : 892816*58,无‘target’项;特征列共57列
# 第4层词缀标注特征变量类型: cat为多分类变量,bin为二分类变量,无词缀则属于连续或顺序变量
# 第2层词缀标注特征变量名称: ind为与司机个人相关特征,reg为与地区相关特征,
#                         car是汽车相关特征,calc通过计算或估计得到的特征

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
SUB_DIR = "output"

NUM_SPLITS = 3
RANDOM_SEED = 2017

# 多分类变量
CATEGORICAL_COLS = [
    # 'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat',
    # 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',
    # 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
    # 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
    # 'ps_car_10_cat', 'ps_car_11_cat',
]

# 数值变量+特征工程
NUMERIC_COLS = [
    # # binary
    # "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin",
    # "ps_ind_09_bin", "ps_ind_10_bin", "ps_ind_11_bin",
    # "ps_ind_12_bin", "ps_ind_13_bin", "ps_ind_16_bin",
    # "ps_ind_17_bin", "ps_ind_18_bin",
    # "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    # "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin",

    # numeric
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",

    # feature engineering
    "missed_feature_num", "ps_car_13_x_ps_reg_03",
]

# 忽略不计特征
IGNORE_COLS = [
    "id", "target",
    "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    "ps_calc_13", "ps_calc_14",
    "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"
]
