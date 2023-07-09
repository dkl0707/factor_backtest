"""
Author: dkl
Description: 因子分析模块, 包含
    * 计算因子IC序列: get_factor_ic
    * Newywest-T计算: newy_west_test
    * 分析因子IC: analysis_factor_ic
    * 计算因子收益率经风险调整后的Alpha和t值: risk_adj_alpha
    * Fama-Macbeth回归: fama_macbeth_reg
Date: 2023-07-05 19:12:16
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels import FamaMacBeth
import utils


def get_factor_ic(factor_df, ret_df, factor_name):
    """
    Description
    ----------
    计算因子IC序列

    Parameters
    ----------
    factor_df: pandas.DataFrame. 未提前的因子数据.

    Return
    ----------
    pandas.DataFrame.
    """

    def calc_corr_func(df):
        return np.corrcoef(df[factor_name], df["ret"])[0, 1]

    prev_factor_df = utils.get_previous_factor(factor_df)
    df = pd.merge(prev_factor_df, ret_df, on=["trade_date", "stock_code"])
    ic_df = df.groupby(["trade_date"], group_keys=False).apply(calc_corr_func)
    ic_df = ic_df.reset_index()
    ic_df.columns = ["trade_date", "IC"]
    return ic_df


def newy_west_test(arr, factor_name="factor", max_lags=None):
    """
    Description
    ----------
    计算收益率均值并输出Newywest-t统计量和p值

    Parameters
    ----------
    arr: array_like. 收益率序列
    factor_name: str. 因子名称, 默认为'factor'
    max_lags: int. 滞后阶数. 默认为None, 即int(4*(T/100)**(2/9))

    Return
    ----------
    Dict.
    输出示例为{"ret_mean(%)":10%, "t-value": 2.30, "p-value": 0.02, "p-star": **}
    """
    arr = np.array(arr).reshape(-1)
    if max_lags is None:
        T = len(arr)
        max_lags = int(4 * (T / 100) ** (2 / 9))
    mean_test = {"factor_name": [factor_name]}
    model = sm.OLS(arr, np.ones(len(arr)))
    result = model.fit(missing="drop", cov_type="HAC", cov_kwds={"maxlags": max_lags})
    # 平均收益率
    ret_mean = list(result.params)[0] * 100
    # t值
    t_value = list(result.tvalues)[0]
    # p值
    p_value = list(result.pvalues)[0]
    # p值对应的星号
    p_star = _get_p_star(p_value)
    # 平均收益率
    mean_test["ret_mean(%)"] = [round(ret_mean, 3)]
    # t值
    mean_test["t-value"] = [round(t_value, 3)]
    # p值
    mean_test["p-value"] = [round(p_value, 3)]
    mean_test["p-star"] = [p_star]
    return mean_test


def _get_p_star(x):
    dct = {
        "": [0.1, 1],
        "*": [0.05, 0.1],
        "**": [0.01, 0.05],
        "***": [-0.01, 0.01],
    }
    for key, value in dct.items():
        lower, upper = value[0], value[1]
        if (lower < x) and (x <= upper):
            return key
    raise ValueError("x is invalid.")


def analysis_factor_ic(factor_df, ret_df, factor_name):
    """
    Description
    ----------
    分析因子IC

    Parameters
    ----------
    factor_df: pandas.DataFrame.
        未提前的因子数据，格式为trade_date, stock_code, factor_name
    ret_df: pandas.DataFrame.
        收益率数据，格式为trade_date, stock_code, ret
    factor_name: str.
        因子名称, 默认为'factor'

    Return
    ----------
    tuple. 第一个为Dict，格式为dct = {
        "因子名称": [factor_name],
        "IC均值": [ic_mean],
        "IC标准差": [ic_std],
        "IR比率": [ir_ratio],
        "IC>0的比例(%)": [ic0_ratio],
        "IC>0.02的比例(%)": [ic002_ratio],
    }
    第二个为因子的IC时序图和累计图
    """
    ic_df = get_factor_ic(factor_df, ret_df, factor_name)
    ic_mean = ic_df["IC"].mean()
    ic_std = ic_df["IC"].std()
    ir_ratio = ic_mean / ic_std
    ic0_ratio = 100 * len(ic_df.loc[ic_df["IC"] > 0, :]) / len(ic_df)
    ic002_ratio = 100 * len(ic_df.loc[ic_df["IC"] > 0.02, :]) / len(ic_df)
    dct = {
        "因子名称": [factor_name],
        "IC均值": [ic_mean],
        "IC标准差": [ic_std],
        "IR比率": [ir_ratio],
        "IC>0的比例(%)": [ic0_ratio],
        "IC>0.02的比例(%)": [ic002_ratio],
    }
    ic_df["trade_date"] = pd.to_datetime(ic_df["trade_date"])
    plot_params_dct = {
        "x1": ic_df["trade_date"],
        "y1": ic_df["IC"],
        "x2": ic_df["trade_date"],
        "y2": ic_df["IC"].cumsum(),
        "label1": "因子IC",
        "label2": "因子IC累计值",
        "xlabel": "日期",
        "ylabel1": "因子IC",
        "ylabel2": "因子IC累计值",
        "fig_title": f"因子{factor_name}的IC分析",
    }
    fig = utils.plot_bar_line(**plot_params_dct)
    return dct, fig


def risk_adj_alpha(factor_ret, risk_factor_ret, max_lags=None):
    """
    Description
    ----------
    计算因子收益率经风险调整后的Alpha和t值

    Parameters
    ----------
    factor_ret: pandas.DataFrame. 待检测因子收益率序列
    risk_factor_ret: pandas.DataFrame. 风险因子收益率矩阵
    max_lags: int. 滞后阶数. 默认为None, 即int(4*(T/100)**(2/9))

    Return
    ----------
    tuple. 为(alpha, alphat)
    """
    risk_factor_name_lst = risk_factor_ret.drop(columns=["trade_date"]).columns.tolist()
    factor_name = factor_ret.drop(columns=["trade_date"]).columns[0]
    df = pd.merge(factor_ret, risk_factor_ret, on="trade_date")
    risk_factor_ret = df[risk_factor_name_lst].values
    factor_ret = df[factor_name].values
    if max_lags is None:
        T = len(factor_ret)
        max_lags = int(4 * (T / 100) ** (2 / 9))
    # 加入常数项，回归
    X = sm.add_constant(risk_factor_ret)
    model = sm.OLS(factor_ret, X)
    results = model.fit(missing="drop", cov_type="HAC", cov_kwds={"maxlags": max_lags})
    # 获取系数
    coefficients = results.params
    # 风险调整的alpha
    alpha = coefficients[0]
    # 计算Newey-West调整的t值
    cov_matrix = results.cov_params()
    alphat = alpha / np.sqrt(np.diag(cov_matrix)[0])
    return alpha, alphat


def fama_macbeth_reg(ret, factor_df, factor_name_lst):
    """
    Description
    ----------
    Fama-Macbeth回归
    返回平均观测值数量, 系数对应的Newey-West t值,估计参数和R-square

    Parameters
    ----------
    ret: pandas.DataFrame.
        股票收益率数据, 格式为trade_date, stock_code, ret
    factor_df: pandas.DataFrame.
        因子数据, 格式为trade_date, stock_code, factor_name
    factor_name_lst: List[str].
        因子变量名列表。输入格式为列表[factor_name1, factor_name2, …, factor_namem]

    Return
    ----------
    Dict. 输出示例为: {
        "factor_name": factor_name_lst,
        "beta": list(fama_macbeth.params[1:]),
        "t-value": list(fama_macbeth.tstats[1:]),
        "R-square": fama_macbeth.rsquared,
        "Average-Obs": fama_macbeth.time_info[0],
    }
    """
    utils._check_columns(ret, ["trade_date", "stock_code", "ret"])
    utils._check_columns(factor_df, ["trade_date", "stock_code"] + factor_name_lst)
    # 将因子数据提前一期
    prev_factor_df = utils.get_previous_factor(factor_df)
    # 合并数据
    regdf = pd.merge(ret, prev_factor_df, on=["trade_date", "stock_code"])
    regdf["trade_date"] = pd.to_datetime(regdf["trade_date"])
    T = len(list(set(regdf["trade_date"])))
    regdf = regdf.sort_index(level=["stock_code", "trade_date"])
    regdf = regdf.set_index(["stock_code", "trade_date"])
    formula = "ret ~ 1 + " + " + ".join(factor_name_lst)
    model = FamaMacBeth.from_formula(formula, data=regdf)
    bandw = 4 * (T / 100) ** (2 / 9)
    fama_macbeth = model.fit(cov_type="kernel", debiased=False, bandwidth=bandw)
    res_dct = {
        "factor_name": factor_name_lst,
        "beta": list(fama_macbeth.params[1:]),
        "t-value": list(fama_macbeth.tstats[1:]),
        "R-square": fama_macbeth.rsquared,
        "Average-Obs": fama_macbeth.time_info[0],
    }
    return res_dct
