'''
Author: dkl
Description: 常用的辅助函数, 包含:
   * 获取上期因子值: get_previous_factor
   * 数据堆栈: stackdf
   * 数据反堆栈: unstackdf
   * 获取交易日历中历史最近的日期: get_last_date
   * 获取交易日历中未来最近的日期: get_next_date
Date: 2022-10-04 09:57:32
'''
import numpy as np
import pandas as pd


# 因子处理部分
def get_previous_factor(factor_df):
    '''
    Description
    ----------
    获取上期因子值

    Parameters
    ----------
    factor_df: pandas.DataFrame. 输入因子数据,必须含有trade_date

    Return
    ----------
    pandas.DataFrame. 上期因子值
    '''
    _check_sub_columns(factor_df, ['trade_date'])
    factor_df = factor_df.copy()
    # 将日期往前挪一期，建立本期交易日和上期交易日的映射DataFrame
    this_td_lst = factor_df['trade_date'].drop_duplicates().tolist()
    last_td_lst = [np.nan] + this_td_lst[:-1]
    td_df = pd.DataFrame({
        'this_trade_date': this_td_lst,
        'last_trade_date': last_td_lst
    })
    # 与原来的收益率数据进行合并
    factor_df = pd.merge(td_df,
                         factor_df,
                         left_on='last_trade_date',
                         right_on='trade_date')
    # 将上期交易日修改为本期交易日，这样每个交易日对应的是下期的收益率
    factor_df = factor_df.drop(columns=['last_trade_date', 'trade_date'])
    factor_df = factor_df.rename(columns={'this_trade_date': 'trade_date'})
    # 去除空值
    factor_df = factor_df.dropna().reset_index(drop=True)
    return factor_df


# 堆栈和反堆栈部分
def stackdf(df, var_name, date_name='trade_date', code_name='stock_code'):
    '''
    Description
    ----------
    对输入数据进行堆栈，每行为截面数据，每列为时间序列数据

    Parameters
    ----------
    df: pandas.DataFrame.
        输出数据为堆栈后的数据
    date_name: str. 日期名称, 默认为trade_date
    code_name: str. 代码名称, 默认为stock_code

    Return
    ----------
    pandas.DataFrame.
    堆栈后的数据,列为trade_date, stock_code和var_name
    '''
    df = df.copy()
    df = df.stack().reset_index()
    df.columns = ['trade_date', 'stock_code', var_name]
    return df


def unstackdf(df, date_name='trade_date', code_name='stock_code'):
    '''
    Description
    ----------
    反堆栈函数

    Parameters
    ----------
    df: pandas.DataFrame.
        输入列必须为三列且必须有date_name和code_name
    date_name: str. 日期名称, 默认为trade_date
    code_name: str. 代码名称, 默认为stock_code

    Return
    ----------
    pandas.DataFrame. 反堆栈后的数据
    '''
    _check_sub_columns(df, [date_name, code_name])
    if not (len(df.columns) == 3):
        error_message = 'length of df.columns must be 3'
        raise ValueError(error_message)
    df = df.copy()
    df = df.set_index([date_name, code_name]).unstack()
    df.columns = df.columns.get_level_values(1).tolist()
    df.index = df.index.tolist()
    return df


# 检查df的列的部分
def _check_sub_columns(df, var_lst):
    '''
    Description
    ----------
    检查var_lst是否是df.columns的列的子集（不考虑排序）

    Parameters
    ----------
    df: pandas.DataFrame. 输入数据
    var_lst: List[str]. 变量名列表

    Return
    ----------
    Bool
    '''
    if not set(var_lst).issubset(df.columns):
        var_name = ','.join(var_lst)
        raise ValueError(f'{var_name} must be in the columns of df.')


def _check_columns(df, var_lst):
    '''
    Description
    ----------
    检查var_lst是否是df.columns的列（不考虑顺序）

    Parameters
    ----------
    df: pandas.DataFrame. 输入数据
    var_lst: List[str]. 变量名列表

    Return
    ----------
    Bool
    '''
    lst1 = list(var_lst)
    lst2 = df.columns.tolist()
    if not sorted(lst1) == sorted(lst2):
        var_str = ', '.join(var_lst)
        err = 'The columns of df must be var_lst:{}'.format(var_str)
        raise ValueError(err)


# 日期部分
def get_last_date(date, trade_date_lst):
    '''
    Description
    ----------
    获取交易日历中历史最近的日期

    Parameters
    ----------
    date: str. 所选日期
    trade_date_lst: List[str]. 交易日历列表

    Return
    ----------
    str. 交易日历中未来最近的日期
    '''
    # 如果输入为空，返回为空
    if date is np.nan:
        return date
    if date < trade_date_lst[0]:
        raise ValueError('date must be smaller than trade_date_lst[0]')
    # 找未来最近的月频交易日
    for i in range(len(trade_date_lst) - 1):
        if (trade_date_lst[i] <= date) and (date < trade_date_lst[i + 1]):
            return trade_date_lst[i]


def get_next_date(date, trade_date_lst):
    '''
    Description
    ----------
    获取交易日历中未来最近的日期

    Parameters
    ----------
    date: str. 所选日期
    trade_date_lst: List[str]. 交易日历列表

    Return
    ----------
    str. 交易日历中未来最近的日期
    '''
    # 如果输入为空，返回为空
    if date is np.nan:
        return date
    # 如果比提取的交易日历中的第一个交易日来的小，返回他
    if date < trade_date_lst[0]:
        return trade_date_lst[0]
    # 找未来最近的月频交易日
    for i in range(len(trade_date_lst) - 1):
        if (trade_date_lst[i] < date) and (date <= trade_date_lst[i + 1]):
            return trade_date_lst[i + 1]
