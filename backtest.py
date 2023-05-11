'''
Author: dkl
Description:
   回测模块，包含:
   * 净值曲线计算：net_value
   * 历史最大净值计算：previous_peak
   * 回撤序列计算：drawdown
   * 计算最大回撤：max_drawdown
   * 年化收益率计算：annualized_return
   * 年化波动率计算：annualized_volatility
   * 年化夏普比率计算：annualized_sharpe
Date: 2022-12-20 08:57:59
'''
import numpy as np
import pandas as pd

# 根据数据频率映射到相应的年化因子
mapping_dct = {'DAILY': 252, 'WEEKLY': 52, 'MONTHLY': 12}


def _convert_returns_type(returns):
    try:
        returns = np.asarray(returns)
    except Exception:
        raise ValueError('returns is not array_like.')
    return returns


def _check_period(period):
    period_lst = list(mapping_dct.keys())
    if period not in period_lst:
        period_str = ','.join(period_lst)
        raise ValueError(f'period must be in {period_str}')
    return


def net_value(returns):
    '''
    Description
    ----------
    计算净值曲线

    Parameters
    ----------
    returns: array_like. 收益率序列

    Return
    ----------
    numpy.ndarray. 净值曲线
    '''
    returns = _convert_returns_type(returns)
    return np.cumprod(1 + returns)


def previous_peak(returns):
    '''
    Description
    ----------
    计算历史最大净值

    Parameters
    ----------
    returns: array_like. 收益率序列

    Return
    ----------
    numpy.ndarray. 历史最大净值
    '''
    returns = _convert_returns_type(returns)
    nv = net_value(returns)
    return np.maximum.accumulate(nv)


def drawdown(returns):
    '''
    Description
    ----------
    计算回撤序列，回撤=净值现值/历史最大净值-1

    Parameters
    ----------
    returns: array_like. 收益率序列

    Return
    ----------
    numpy.ndarray. 单位为%
    '''
    returns = _convert_returns_type(returns)
    nv = net_value(returns)
    previous_peaks = previous_peak(returns)
    dd = (nv / previous_peaks - 1) * 100
    return dd


def max_drawdown(returns):
    '''
    Description
    ----------
    计算最大回撤

    Parameters
    ----------
    returns: array_like. 收益率序列

    Return
    ----------
    float. 最大回撤. 单位为%
    '''
    returns = _convert_returns_type(returns)
    dd = drawdown(returns)
    # 注意上述drawdown单位已为%
    mdd = np.min(dd)
    return mdd


def annualized_return(returns, period='DAILY'):
    '''
    Description
    ----------
    计算年化收益率

    Parameters
    ----------
    returns: array_like. 收益率序列
    period: str. 计算周期, 必须为DAILY, WEEKLY, MONTHLY中的一种。默认'DAILY'

    Return
    ----------
    float. 年化收益率.单位为%
    '''
    _check_period(period)
    returns = _convert_returns_type(returns)
    ann_factor = mapping_dct[period]
    # 交易日数量
    n = returns.shape[0]
    # 计算最后的净值
    nv = net_value(returns)
    final_value = nv[-1]
    # 年化收益率
    ann_ret = 100 * (final_value**(ann_factor / n) - 1)
    return ann_ret


def annualized_volatility(returns, period='DAILY'):
    '''
    Description
    ----------
    计算年化波动率

    Parameters
    ----------
    returns: array_like. 收益率序列
    period: str. 计算周期, 必须为DAILY, WEEKLY, MONTHLY中的一种。默认'DAILY'

    Return
    ----------
    float. 年化波动率.单位为%
    '''
    _check_period(period)
    returns = _convert_returns_type(returns)
    ann_factor = mapping_dct[period]
    ann_vol = 100 * np.std(returns) * np.sqrt(ann_factor)
    return ann_vol


def annualized_sharpe(returns, rf=0, period='DAILY'):
    '''
    Description
    ----------
    计算年化夏普比率.年化夏普=(年化收益率-无风险收益率)/年化波动率

    Parameters
    ----------
    returns: array_like. 收益率序列
    rf: float. 无风险收益率, 单位为绝对值, 默认为0
    period: str. 计算周期, 必须为DAILY, WEEKLY, MONTHLY中的一种。默认'DAILY'

    Return
    ----------
    float. 年化夏普比率.
    '''
    _check_period(period)
    returns = _convert_returns_type(returns)
    ann_ret = annualized_return(returns, period)
    ann_vol = annualized_volatility(returns, period)
    ann_sr = (ann_ret / 100 - rf) / (ann_vol / 100)
    return ann_sr


# 回测模块
def get_backtest_result(returns, rf=0, period='DAILY'):
    '''
    Description
    ----------
    输出回测指标. 包含年化收益率、年化波动率、夏普比率、最大回撤

    Parameters
    ----------
    returns: array_like. 收益率序列
    rf: float. 无风险收益率, 单位为绝对值, 默认为0
    period: str. 计算周期, 必须为DAILY, WEEKLY, MONTHLY中的一种。默认'DAILY'

    Return
    ----------
    Dict. 输出格式为{
        'ann_ret': [ann_ret],
        'ann_vol': [ann_vol],
        'ann_sp': [ann_sp],
        'mdd': [mdd],
    }
    '''
    ann_ret = annualized_return(returns, period)
    ann_vol = annualized_volatility(returns, period)
    ann_sp = annualized_sharpe(returns, rf, period)
    mdd = max_drawdown(returns)
    dct = {
        'ann_ret': [ann_ret],
        'ann_vol': [ann_vol],
        'ann_sp': [ann_sp],
        'mdd': [mdd],
    }
    return pd.DataFrame(dct)
