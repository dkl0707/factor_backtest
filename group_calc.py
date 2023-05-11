'''
Author: dkl, ssj
Description: 分组计算, 包含
    * 计算股票分组收益率: get_group_ret
    * 对股票分组: get_stock_group
Date: 2023-04-28 21:31:08
'''
import numpy as np
import pandas as pd
import utils


# ################################################
# 分组模块
def get_group_ret(factor_df, ret_df, factor_name, n_groups, mktmv_df=None):
    '''
    Description
    ----------
    计算分组的收益率

    Parameters
    ----------
    factor_df: pandas.DataFrame.
        未提前的因子数据，格式为trade_date, stock_code, factor_name
    ret_df: pandas.DataFrame.
        收益率数据，格式为trade_date, stock_code, ret
    factor_name: str.
        因子名称
    n_groups: int.
        分组数量.
    mktmv_df: pandas.DataFrame. 默认为None，即等权
        市值数据.格式为trade_date, stock_code, mktmv

    Return
    ----------
    pandas.DataFrame.
    索引为trade_date, 列名为group0到group{n_group-1},和factor_ret
    '''
    factor_col_lst = ['trade_date', 'stock_code', factor_name]
    ret_col_lst = ['trade_date', 'stock_code', 'ret']
    utils._check_columns(factor_df, factor_col_lst)
    utils._check_columns(ret_df, ret_col_lst)
    # 对因子数据提前一期
    prev_factor_df = utils.get_previous_factor(factor_df)
    # 拼接
    df = pd.merge(prev_factor_df, ret_df, on=['trade_date', 'stock_code'])
    if mktmv_df is not None:
        utils._check_columns(mktmv_df, ['trade_date', 'stock_code', 'mktmv'])
    else:
        mktmv_df = df[['trade_date', 'stock_code']].copy()
        mktmv_df['mktmv'] = 1

    def get_group_weight_ret(df):
        df = df.copy()
        df['weight'] = df['mktmv'] / df['mktmv'].sum()
        return np.sum(df['weight'] * df['ret'])

    # 计算分组
    # 此时, df格式为trade_date, stock_code, factor_name
    df = df.copy()
    df = get_stock_group(df, factor_name, n_groups)
    df = pd.merge(df, mktmv_df, on=['trade_date', 'stock_code'])
    # 此时, df格式为trade_date, stock_code, factor_name, mktmv
    # 分组计算收益率
    group_name = factor_name + '_group'
    g = df.groupby(['trade_date', group_name])
    stacked_group_ret = g.apply(get_group_weight_ret)
    stacked_group_ret = stacked_group_ret.reset_index()
    stacked_group_ret.columns = ['trade_date', group_name, 'ret']
    # 反堆栈
    group_ret = utils.unstackdf(stacked_group_ret, code_name=group_name)
    # 计算多空收益率
    factor_ret = _get_factor_ret(group_ret, n_groups)
    group_ret['factor_ret'] = factor_ret
    return group_ret


def get_stock_group(factor_df, factor_name, n_groups):
    '''
    Description
    ----------
    通过因子值，对股票分成n_groups组
    组名从小到大为group0到group{n_groups-1}

    Parameters
    ----------
    factor_df: pandas.DataFrame.
        因子数据，格式为trade_date, stock_code, factor_name
    factor_name: str.
        因子名称
    n_groups: int.
        分组数量

    Return
    ----------
    pandas.DataFrame.
    格式为trade_date, stock_code, factor_name, factor_name_group
    '''
    col_lst = ['trade_date', 'stock_code', factor_name]
    utils._check_sub_columns(factor_df, col_lst)
    factor_df = factor_df.copy()
    # 对factor_df中的factor_name进行分组，并进行标记
    g = factor_df.groupby('trade_date', group_keys=False)
    factor_df = g.apply(_get_single_period_group,
                        factor_name=factor_name,
                        n_groups=n_groups)
    return factor_df


def _get_single_period_group(df, factor_name, n_groups):
    # 将df中的factor_name进行分组
    # 组名从小到大为group0到group{n_groups-1}
    df = df.copy()
    group_name = factor_name + '_group'
    labels = ['group' + str(i) for i in range(n_groups)]
    df[group_name] = pd.cut(df[factor_name].rank(),
                            bins=n_groups,
                            labels=labels)
    return df


# 多空因子收益
def _get_factor_ret(group_ret, n_groups):
    long_group_name = 'group' + str(n_groups - 1)
    short_group_name = 'group0'
    long_group_ret = group_ret[long_group_name]
    short_group_ret = group_ret[short_group_name]
    factor_ret = long_group_ret - short_group_ret
    return factor_ret
