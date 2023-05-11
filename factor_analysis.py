'''
Author: dkl
Description: 因子分析模块, 包含
    * 计算因子IC序列: get_factor_ic
    * 计算各组收益率的回测指标: get_group_ret_analysis
    * Newywest-T计算: newy_west_test
Date: 2023-05-09 09:12:16
'''
import numpy as np
import pandas as pd
import statsmodels.api as sm
import utils
import backtest


def get_factor_ic(factor_df, ret_df, factor_name):
    '''
    Description
    ----------
    计算因子IC序列

    Parameters
    ----------
    factor_df: pandas.DataFrame. 未提前的因子数据.

    Return
    ----------
    pandas.DataFrame.
    '''

    def calc_corr_func(df):
        return np.corrcoef(df[factor_name], df['ret'])[0, 1]

    prev_factor_df = utils.get_previous_factor(factor_df)
    df = pd.merge(prev_factor_df, ret_df, on=['trade_date', 'stock_code'])
    ic_s = df.groupby(['trade_date'], group_keys=False).apply(calc_corr_func)
    ic_df = pd.DataFrame(ic_s, columns=['ic'])
    return ic_df


def get_group_ret_analysis(group_ret, rf=0, period='DAILY'):
    '''
    Description
    ----------
    计算各组收益率的回测指标

    Parameters
    ----------
    group_ret: pandas.DataFrame.
        各组收益率, 每列为各组收益率的时间序列
    rf: float. 无风险收益率, 默认为0
    period: str. 指定数据频率
        有DAILY, WEEKLY, MONTHLY三种, 默认为DAILY

    Return
    ----------
    pandas.DataFrame.
    '''
    analysis_df = pd.DataFrame()
    for col in group_ret.columns:
        res_df = backtest.get_backtest_result(group_ret[col], period=period).T
        analysis_df = pd.concat([analysis_df, res_df], axis=1)
    analysis_df.columns = group_ret.columns
    return analysis_df


# Newywest-t统计量计算
def newy_west_test(arr, factor_name='factor'):
    '''
    Description
    ----------
    计算收益率均值并输出t统计量和p值

    Parameters
    ----------
    arr: array_like. 收益率序列
    factor_name: 因子名称, 默认为'factor'

    Return
    ----------
    Dict.
    输出示例为{"ret_mean(%)":10%, "t-value": 2.30, "p-value": 0.02, "p-star": **}
    '''
    arr = np.array(arr).reshape(-1)
    mean_test = {'factor_name': [factor_name]}
    model = sm.OLS(arr, np.ones(len(arr)))
    result = model.fit(missing='drop', cov_type='HAC', cov_kwds={'maxlags': 5})
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
    mean_test['t-value'] = [round(t_value, 3)]
    # p值
    mean_test["p-value"] = [round(p_value, 3)]
    mean_test['p-star'] = [p_star]
    return mean_test


def _get_p_star(x):
    dct = {
        '': [0.1, 1],
        '*': [0.05, 0.1],
        '**': [0.01, 0.05],
        '***': [-0.01, 0.01],
    }
    for key, value in dct.items():
        lower, upper = value[0], value[1]
        if (lower < x) and (x <= upper):
            return key
    raise ValueError('x is invalid.')
