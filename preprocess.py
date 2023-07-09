"""
Author: dkl, lhx
Description: 因子预处理代码，包含：
    * 去极值: del_outlier
    * 标准化: standardize
    * 中性化: neutralize
Date: 2023-07-07 11:45:56
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import utils


# 去极值
def del_outlier(factor_df, factor_name, method="mad", n=3):
    """
    Description
    ----------
    对每期因子进行去极值

    Parameters
    ----------
    factor_df: pandas.DataFrame. 因子数据,格式为trade_date,stock_code,factor
    factor_name: str. 因子名称
    method: str. 去极值方式,为'mad'或'sigma',默认为mad
    n: float.去极值的n值.默认取值为3

    Return
    ----------
    pandas.DataFrame.
    去极值后的因子数据, 格式为trade_date,stock_code,factor
    """
    utils._check_sub_columns(factor_df, [factor_name])
    factor_df = factor_df.copy()
    if method == "mad":
        g = factor_df.groupby("trade_date", group_keys=False)
        factor_df = g.apply(_single_mad_del, factor_name, n)
    elif method == "sigma":
        g = factor_df.groupby("trade_date", group_keys=False)
        factor_df = g.apply(_single_sigma_del, factor_name, n)
    if method not in ["mad", "sigma"]:
        raise ValueError("method must be mad or sigma")
    return factor_df


def _single_mad_del(factor_df, factor_name, n):
    """
    Description
    ----------
    单期MAD法去极值

    Parameters
    ----------
    factor_df: pandas.DataFrame. 因子值数据
    factor_name: str.因子名称
    n: float. 去极值的n

    Return
    ----------
    去极值后的因子数据
    """
    # 找出当期factor和factor_median的偏差bias_sr
    factor_median = factor_df[factor_name].median()
    bias_sr = abs(factor_df[factor_name] - factor_median)
    # 找到bias_sr的中位数new_median
    new_median = bias_sr.median()
    # 找到上下界
    dt_up = factor_median + n * new_median
    dt_down = factor_median - n * new_median

    # 超出上下界的值，赋值为上下界
    factor_df[factor_name] = factor_df[factor_name].clip(dt_down, dt_up, axis=0)
    return factor_df


def _single_sigma_del(factor_df, factor_name, n):
    """
    Description
    ----------
    单期Sigma法去极值

    Parameters
    ----------
    factor_df: pandas.DataFrame. 因子值数据
    factor_name: str. 因子名称
    n: float. 去极值的n

    Return
    ----------
    去极值后的因子数据
    """
    factor_mean = factor_df[factor_name].mean()
    factor_std = factor_df[factor_name].std()
    dt_up = factor_mean + n * factor_std
    dt_down = factor_mean - n * factor_std
    factor_df[factor_name] = factor_df[factor_name].clip(
        dt_down, dt_up, axis=0
    )  # 超出上下限的值，赋值为上下限
    return factor_df


# 标准化代码
def standardize(factor_df, factor_name, method="rank"):
    """
    Description
    ----------
    标准化

    Parameters
    ----------
    factor: pandas.DataFrame,因子值,格式为trade_date,stock_code,factor
    factor_name: str.因子名称
    method: str.中性化方式，可选为'rank'（排序标准化）或者'zscore'（Z-score标准化），默认为rank

    Return
    ----------
    pandas.DataFrame.
    标准化后的因子数据, 格式为trade_date,stock_code,factor
    """
    utils._check_sub_columns(factor_df, [factor_name])
    if method == "zscore":
        g = factor_df.groupby("trade_date", group_keys=False)
        factor_df = g.apply(_single_zscore_standardize, factor_name)
    elif method == "rank":
        g = factor_df.groupby("trade_date", group_keys=False)
        factor_df = g.apply(_single_rank_standardize, factor_name)
    else:
        raise ValueError("method must be rank or zscore")
    return factor_df


def _single_rank_standardize(factor_df, factor_name):
    """
    Description
    ----------
    单期因子数据排序标准化

    Parameters
    ----------
    factor: pandas.DataFrame,因子值,格式为trade_date,stock_code,factor
    factor_name: str.因子名称

    Return
    ----------
    pandas.DataFrame.排序标准化后的因子数据
    """
    factor_df[factor_name] = factor_df[factor_name].rank()
    return _single_zscore_standardize(factor_df, factor_name)


def _single_zscore_standardize(factor_df, factor_name):
    """
    Description
    ----------
    单期因子数据zscore标准化

    Parameters
    ----------
    factor: pandas.DataFrame,因子值,格式为trade_date,stock_code,factor
    factor_name: str.因子名称

    Return
    ----------
    pandas.DataFrame.zscore标准化后的因子数据
    """
    factor_mean = factor_df[factor_name].mean()
    factor_std = factor_df[factor_name].std()
    factor_df[factor_name] = (factor_df[factor_name] - factor_mean) / factor_std
    return factor_df


# 中性化代码
def neutralize(factor_df, factor_name, mktmv_df=None, industry_df=None):
    """
    Description
    ----------
    中性化

    Parameters
    ----------
    factor_df: pandas.DataFrame.
        因子值, 格式为trade_date,stock_code,factor
    mktmv_df: pandas.DataFrame.
        股票流通市值,格式为trade_date,stock_code,mktmv.
        默认为None即不进行市值中性化
    industry_df: pandas.DataFrame, 股票所属行业, 格式为trade_date,stock_code,ind_code.默认为None即不进行行业中性化

    Return
    ----------
    pandas.DataFrame.
    中性化后的因子数据, 格式为trade_date,stock_code,factor
    """
    neu_factor = factor_df.copy()
    if mktmv_df is not None:
        neu_factor = mktmv_neutralize(neu_factor, factor_name, mktmv_df)
    if industry_df is not None:
        neu_factor = ind_neutralize(neu_factor, factor_name, industry_df)
    return neu_factor


# 市值中性化
def mktmv_neutralize(factor_df, factor_name, mktmv_df):
    """
    Description
    ----------
    市值中性化

    Parameters
    ----------
    factor_df: pandas.DataFrame, 格式为trade_date, stock_code, factor
    factor_name: str.因子名称
    mktmv_df: pandas.DataFrame,股票流通市值,格式为trade_date,stock_code,mktmv.

    Return
    ----------
    pandas.DataFrame.中性化后的因子值
    """
    # 检查输入数据
    utils._check_sub_columns(mktmv_df, ["mktmv"])
    utils._check_sub_columns(factor_df, [factor_name])
    # 合并两个数据，groupby做回归
    df = pd.merge(factor_df, mktmv_df, on=["trade_date", "stock_code"])
    g = df.groupby("trade_date", group_keys=False)
    df = g.apply(_mktmv_reg, factor_name)
    df = df.drop(columns=["mktmv"])
    return df


def _mktmv_reg(df, factor_name):
    """
    Description
    ----------
    对单期因子进行市值中性化

    Parameters
    ----------
    df:pandas.DataFrame, 格式为trade_date, stock_code, factor, mktmv
    factor_name: str.因子名称

    Return
    ----------
    pandas.DataFrame.中性化后的因子值
    """
    x = df["mktmv"].values.reshape(-1, 1)
    y = df[factor_name]
    lr = LinearRegression()
    lr.fit(x, y)  # 拟合
    y_predict = lr.predict(x)  # 预测
    df[factor_name] = y - y_predict
    return df


# 行业中性化
def ind_neutralize(factor_df, factor_name, industry_df):
    """
    Description
    ----------
    对每期因子进行行业中性化
    方法: 先用pd.get_dummies生成行业虚拟变量, 然后用带截距项回归得到残差作为因子

    Parameters
    ----------
    factor_df: pandas.DataFrame,因子值,格式为trade_date,stock_code,factor
    factor_name: str. 因子名称
    industry_df: pandas.DataFrame, 股票所属行业, 格式为trade_date,stock_code,ind_code

    Return
    ----------
    pandas.DataFrame.行业中性化后的因子数据
    """
    # 检查输入数据
    utils._check_sub_columns(factor_df, [factor_name])
    utils._check_sub_columns(industry_df, ["ind_code"])
    # 生成虚拟变量，拼接形成新的df
    ind_dummies = pd.get_dummies(industry_df["ind_code"], drop_first=True, prefix="ind")
    # 格式为 trade_date,stock_code,dummies_ind_code
    ind_new = pd.concat([industry_df.drop(columns=["ind_code"]), ind_dummies], axis=1)
    # 拼接两个表格
    df = pd.merge(factor_df, ind_new, on=["trade_date", "stock_code"])
    g = df.groupby("trade_date", group_keys=False)
    df = g.apply(_single_ind_neutralize, factor_name)
    df = df[["trade_date", "stock_code", factor_name]].copy()
    return df


def _single_ind_neutralize(df, factor_name):
    """
    Description
    ----------
    对单期因子进行行业中性化

    Parameters
    ----------
    df: pandas.DataFrame, 因子值和行业的df, 格式为trade_date,stock_code,'factor_name',dummy_ind_code
    factor_name: str. 因子名称

    Return
    ----------
    pandas.DataFrame.行业中性化后的因子数据
    """
    x = df.iloc[:, 3:]
    y = df[factor_name]
    # 计算回归残差
    lr = LinearRegression()
    lr.fit(x, y)
    y_predict = lr.predict(x)
    df[factor_name] = y - y_predict
    return df
