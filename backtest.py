"""
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
   * 计算年化超额收益率: er_annual_return
   * 计算超额收益的年化波动率: er_annual_volatility
   * 计算信息比率: information_ratio
   * 计算超额收益的最大回撤: er_max_drawdown
   * 计算策略相对于基准的胜率: winrate
   * 输出回测指标: get_backtest_result
Date: 2023-07-07 08:57:59
"""
import numpy as np


# 根据数据频率映射到相应的年化因子
mapping_dct = {"DAILY": 252, "WEEKLY": 52, "MONTHLY": 12}


def _convert_returns_type(returns):
    try:
        returns = np.asarray(returns)
    except Exception:
        raise ValueError("returns is not array_like.")
    return returns


def _check_period(period):
    period_lst = list(mapping_dct.keys())
    if period not in period_lst:
        period_str = ",".join(period_lst)
        raise ValueError(f"period must be in {period_str}")
    return


def net_value(returns):
    """
    Description
    ----------
    计算净值曲线

    Parameters
    ----------
    returns: array_like. 收益率序列

    Return
    ----------
    numpy.ndarray. 净值曲线
    """
    returns = _convert_returns_type(returns)
    return np.cumprod(1 + returns)


def previous_peak(returns):
    """
    Description
    ----------
    计算历史最大净值

    Parameters
    ----------
    returns: array_like. 收益率序列

    Return
    ----------
    numpy.ndarray. 历史最大净值
    """
    returns = _convert_returns_type(returns)
    nv = net_value(returns)
    return np.maximum.accumulate(nv)


def drawdown(returns):
    """
    Description
    ----------
    计算回撤序列，回撤=净值现值/历史最大净值-1

    Parameters
    ----------
    returns: array_like. 收益率序列

    Return
    ----------
    numpy.ndarray. 回撤序列, 单位为%
    """
    returns = _convert_returns_type(returns)
    nv = net_value(returns)
    previous_peaks = previous_peak(returns)
    dd = (nv / previous_peaks - 1) * 100
    return dd


def max_drawdown(returns):
    """
    Description
    ----------
    计算最大回撤

    Parameters
    ----------
    returns: array_like. 收益率序列

    Return
    ----------
    float. 最大回撤. 单位为%
    """
    returns = _convert_returns_type(returns)
    dd = drawdown(returns)
    # 注意上述drawdown单位已为%
    mdd = np.min(dd)
    return mdd


def annualized_return(returns, period="DAILY"):
    """
    Description
    ----------
    计算年化收益率, 即净值**(1/(T/ann_factor))-1

    Parameters
    ----------
    returns: array_like. 收益率序列
    period: str. 计算周期, 必须为DAILY, WEEKLY, MONTHLY中的一种。默认'DAILY'

    Return
    ----------
    float. 年化收益率.单位为%
    """
    _check_period(period)
    returns = _convert_returns_type(returns)
    ann_factor = mapping_dct[period]
    # 交易日数量
    n = returns.shape[0]
    # 计算最后的净值
    nv = net_value(returns)
    final_value = nv[-1]
    # 年化收益率
    ann_ret = 100 * (final_value ** (ann_factor / n) - 1)
    return ann_ret


def annualized_volatility(returns, period="DAILY"):
    """
    Description
    ----------
    计算年化波动率，即标准差*sqrt(ann_factor)

    Parameters
    ----------
    returns: array_like. 收益率序列
    period: str. 计算周期, 必须为DAILY, WEEKLY, MONTHLY中的一种。默认'DAILY'

    Return
    ----------
    float. 年化波动率.单位为%
    """
    _check_period(period)
    returns = _convert_returns_type(returns)
    ann_factor = mapping_dct[period]
    ann_vol = 100 * np.std(returns) * np.sqrt(ann_factor)
    return ann_vol


def annualized_sharpe(returns, rf=0, period="DAILY"):
    """
    Description
    ----------
    计算年化夏普比率.年化夏普=(年化收益率-无风险收益率)/年化波动率

    Parameters
    ----------
    returns: array_like. 收益率序列
    rf: float. 无风险收益率, 默认为0
    period: str. 计算周期, 必须为DAILY, WEEKLY, MONTHLY中的一种。默认'DAILY'

    Return
    ----------
    float. 年化夏普比率.
    """
    _check_period(period)
    returns = _convert_returns_type(returns)
    ann_ret = annualized_return(returns, period)
    ann_vol = annualized_volatility(returns, period)
    if ann_vol < 1e-10:
        return 0
    return (ann_ret / 100 - rf) / (ann_vol / 100)


# 和基准比较模块
def _compare_length(returns, benchmark_returns):
    if len(returns) != len(benchmark_returns):
        message = "Length of returns must be equal to length of benchmark_returns."
        raise ValueError(message)


def er_annual_return(returns, benchmark_returns, period="DAILY"):
    """
    Description
    ----------
    计算年化超额收益率, 即(1+年化策略收益率)/(1+年化基准收益率)-1

    Parameters
    ----------
    returns: array_like. 收益率序列
    period: str. 计算周期, 必须为DAILY, WEEKLY, MONTHLY中的一种。默认'DAILY'

    Return
    ----------
    float. 年化超额收益率.单位为%
    """
    _check_period(period)
    returns = _convert_returns_type(returns)
    benchmark_returns = _convert_returns_type(benchmark_returns)
    _compare_length(returns, benchmark_returns)
    str_ann_ret = annualized_return(returns, period) / 100
    benchmark_ann_ret = annualized_return(benchmark_returns, period) / 100
    er_ann_ret = 100 * ((1 + str_ann_ret) / (1 + benchmark_ann_ret) - 1)
    return er_ann_ret


def er_annual_volatility(returns, benchmark_returns, period="DAILY"):
    """
    Description
    ----------
    计算超额收益的年化波动率

    Parameters
    ----------
    returns: array_like. 收益率序列
    benchmark_returns: array_like. 基准收益率序列
    period: str. 计算周期, 必须为DAILY, WEEKLY, MONTHLY中的一种。默认'DAILY'

    Return
    ----------
    float. 超额收益的年化波动率.单位为%
    """
    _check_period(period)
    returns = _convert_returns_type(returns)
    benchmark_returns = _convert_returns_type(benchmark_returns)
    _compare_length(returns, benchmark_returns)
    er_ret = returns - benchmark_returns
    er_ann_vol = annualized_volatility(er_ret, period)
    return er_ann_vol


def information_ratio(returns, benchmark_returns, period="DAILY"):
    """
    Description
    ----------
    计算信息比率, 即超额年化收益率/超额年化夏普比率

    Parameters
    ----------
    returns: array_like. 收益率序列
    benchmark_returns: array_like. 基准收益率序列
    period: str. 计算周期, 必须为DAILY, WEEKLY, MONTHLY中的一种。默认'DAILY'

    Return
    ----------
    float. 信息比率
    """
    _check_period(period)
    returns = _convert_returns_type(returns)
    benchmark_returns = _convert_returns_type(benchmark_returns)
    _compare_length(returns, benchmark_returns)
    er_ann_ret = er_annual_return(returns, benchmark_returns, period=period)
    er_ann_vol = er_annual_volatility(returns, benchmark_returns, period=period)
    if er_ann_vol < 1e-10:
        return 0
    return er_ann_ret / er_ann_vol


def er_max_drawdown(returns, benchmark_returns):
    """
    Description
    ----------
    计算超额收益的最大回撤

    Parameters
    ----------
    returns: array_like. 收益率序列
    benchmark_returns: array_like. 基准收益率序列

    Return
    ----------
    float. 超额收益的最大回撤.单位为%
    """
    returns = _convert_returns_type(returns)
    benchmark_returns = _convert_returns_type(benchmark_returns)
    _compare_length(returns, benchmark_returns)
    # 计算策略和基准的净值,再计算超额收益的净值
    str_nv = np.cumprod(1 + returns)
    benchmark_nv = np.cumprod(1 + benchmark_returns)
    er_nv = str_nv / benchmark_nv
    # 历史最大净值
    er_prev_peaks = np.maximum.accumulate(er_nv)
    # 回撤
    er_dd = 100 * (er_nv / er_prev_peaks - 1)
    # 注意上述drawdown单位已为%
    er_mdd = np.min(er_dd)
    return er_mdd


def winrate(returns, benchmark_returns):
    """
    Description
    ----------
    计算策略相对于基准的胜率

    Parameters
    ----------
    returns: array_like. 收益率序列
    benchmark_returns: array_like. 基准收益率序列

    Return
    ----------
    float. 策略相对于基准的胜率.单位为%
    """
    returns = _convert_returns_type(returns)
    benchmark_returns = _convert_returns_type(benchmark_returns)
    _compare_length(returns, benchmark_returns)
    er_ret = returns - benchmark_returns
    rate = 100 * np.sum(np.where(er_ret > 0, 1, 0)) / len(er_ret)
    return rate


# 总结输出模块
def get_backtest_result(returns, rf=0, benchmark_returns=None, period="DAILY"):
    """
    Description
    ----------
    输出回测指标. 包含年化收益率、年化波动率、夏普比率、最大回撤等

    Parameters
    ----------
    returns: array_like. 收益率序列
    rf: float. 无风险收益率, 单位为绝对值, 默认为0
    benchmark_returns: array_like. 基准收益率序列, 默认为None，即不计算相关指标
    period: str. 计算周期, 必须为DAILY, WEEKLY, MONTHLY中的一种。默认'DAILY'

    Return
    ----------
    Dict. 输出格式为{
        '年化收益率(%)': [ann_ret],
        '年化波动率(%)': [ann_vol],
        '夏普比率': [ann_sp],
        '最大回撤(%)': [mdd],
    }
    若benchmark不为None，则会额外输出:超额年化收益率(%),
    超额年化波动率(%). 信息比率, 相对基准胜率(%), 超额收益最大回撤(%)
    """
    ann_ret = annualized_return(returns, period)
    ann_vol = annualized_volatility(returns, period)
    ann_sr = annualized_sharpe(returns, rf, period)
    mdd = max_drawdown(returns)
    dct = {
        "年化收益率(%)": [ann_ret],
        "年化波动率(%)": [ann_vol],
        "夏普比率": [ann_sr],
        "最大回撤(%)": [mdd],
    }
    er_dct = dict()
    if benchmark_returns is not None:
        er_ann_ret = er_annual_return(returns, benchmark_returns, period=period)
        er_ann_vol = er_annual_volatility(returns, benchmark_returns, period=period)
        str_ir = information_ratio(returns, benchmark_returns, period)
        str_winrate = winrate(returns, benchmark_returns)
        er_mdd = er_max_drawdown(returns, benchmark_returns)
        er_dct = {
            "超额年化收益率(%)": [er_ann_ret],
            "超额年化波动率(%)": [er_ann_vol],
            "信息比率": [str_ir],
            "相对基准胜率(%)": [str_winrate],
            "超额收益最大回撤(%)": [er_mdd],
        }
    dct.update(er_dct)
    return dct
