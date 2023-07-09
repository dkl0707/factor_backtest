"""
Author: dkl, ssj
Description: 分组计算, 包含
    * 对股票分组: get_stock_group
    * 计算股票分组收益率: get_group_ret
    * 计算各组收益率的回测指标: get_group_ret_backtest
    * 单分组下的因子分析: analysis_group_ret
    * 获取序贯排序双分组情况: get_double_sort_group
    * 计算序贯排序双分组的收益率: get_doublesort_group_ret
    * 计算序贯排序双分组收益率均值: double_sort_mean
    * 计算序贯排序双分组的回测指标: double_sort_backtest
Date: 2023-07-05 21:31:08
"""
import numpy as np
import pandas as pd
import factor_analysis
import backtest
import utils


# ################################################
# 分组模块
def get_stock_group(factor_df, factor_name, n_groups):
    """
    Description
    ----------
    通过因子值，对股票分成n_groups组
    组名从小到大为Group0到Group{n_groups-1}

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
    """
    col_lst = ["trade_date", "stock_code", factor_name]
    utils._check_sub_columns(factor_df, col_lst)
    factor_df = factor_df.copy()
    # 对factor_df中的factor_name进行分组，并进行标记
    g = factor_df.groupby("trade_date", group_keys=False)
    factor_df = g.apply(
        _get_single_period_group, factor_name=factor_name, n_groups=n_groups
    )
    return factor_df


def _get_single_period_group(df, factor_name, n_groups):
    # 将df中的factor_name进行分组
    # 组名从小到大为group0到group{n_groups-1}
    df = df.copy()
    group_name = factor_name + "_group"
    labels = ["Group" + str(i) for i in range(n_groups)]
    df[group_name] = pd.cut(df[factor_name].rank(), bins=n_groups, labels=labels)
    return df


def get_group_ret(factor_df, ret_df, factor_name, n_groups, mktmv_df=None):
    """
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
    索引为trade_date, 列名为group0到group{n_group-1},和H-L
    """
    factor_col_lst = ["trade_date", "stock_code", factor_name]
    ret_col_lst = ["trade_date", "stock_code", "ret"]
    utils._check_columns(factor_df, factor_col_lst)
    utils._check_columns(ret_df, ret_col_lst)
    # 对因子数据提前一期
    prev_factor_df = utils.get_previous_factor(factor_df)
    # 拼接
    df = pd.merge(prev_factor_df, ret_df, on=["trade_date", "stock_code"])
    if mktmv_df is not None:
        utils._check_columns(mktmv_df, ["trade_date", "stock_code", "mktmv"])
        mktmv_df = utils.get_previous_factor(mktmv_df)
    else:
        mktmv_df = df[["trade_date", "stock_code"]].copy()
        mktmv_df["mktmv"] = 1

    def get_group_weight_ret(df):
        df = df.copy()
        df["weight"] = df["mktmv"] / df["mktmv"].sum()
        return np.sum(df["weight"] * df["ret"])

    # 计算分组
    # 此时, df格式为trade_date, stock_code, factor_name
    df = df.copy()
    df = get_stock_group(df, factor_name, n_groups)
    df = pd.merge(df, mktmv_df, on=["trade_date", "stock_code"])
    # 此时, df格式为trade_date, stock_code, factor_name, mktmv
    # 分组计算收益率
    group_name = factor_name + "_group"
    g = df.groupby(["trade_date", group_name])
    stacked_group_ret = g.apply(get_group_weight_ret)
    stacked_group_ret = stacked_group_ret.reset_index()
    stacked_group_ret.columns = ["trade_date", group_name, "ret"]
    # 反堆栈
    group_ret = utils.unstackdf(stacked_group_ret, code_name=group_name)
    # 计算多空收益率
    factor_ret = _get_factor_ret(group_ret, n_groups)
    group_ret["H-L"] = factor_ret
    return group_ret


def _get_factor_ret(group_ret, n_groups):
    # 多空因子收益
    long_group_name = "Group" + str(n_groups - 1)
    short_group_name = "Group0"
    long_group_ret = group_ret[long_group_name]
    short_group_ret = group_ret[short_group_name]
    factor_ret = long_group_ret - short_group_ret
    return factor_ret


def get_group_ret_backtest(group_ret, rf=0, benchmark=None, period="DAILY"):
    """
    Description
    ----------
    计算各组收益率的回测指标

    Parameters
    ----------
    group_ret: pandas.DataFrame.
        各组收益率, 每列为各组收益率的时间序列
    rf: float.
        无风险收益率, 默认为0
    benchmark: pandas. DataFrame.
        基准收益率数据，格式为trade_date, ret
    period: str. 指定数据频率
        有DAILY, WEEKLY, MONTHLY三种, 默认为DAILY

    Return
    ----------
    pandas.DataFrame. 列名为分组名称
    行名为年化收益率(%), 年化波动率(%), 夏普比率, 最大回撤(%)
    若benchmark不为None，则会额外输出: 超额年化收益率(%),
    超额年化波动率(%). 信息比率, 相对基准胜率(%), 超额收益最大回撤(%)
    """
    if benchmark is not None:
        utils._check_columns(benchmark, ["trade_date", "ret"])
        benchmark = benchmark.set_index("trade_date")
        s1 = set(benchmark.index.tolist())
        s2 = set(group_ret.index.tolist())
        common_time_lst = sorted(list(s1.intersection(s2)))
        group_ret = group_ret.loc[common_time_lst].copy()
        benchmark = benchmark.loc[common_time_lst].copy()
        benchmark_ret = benchmark["ret"]
        benchmark.columns = ["benchmark"]
        group_ret = pd.concat([group_ret, benchmark], axis=1)
    else:
        benchmark_ret = None
    backtest_df = pd.DataFrame()
    for col in group_ret.columns:
        res_dct = backtest.get_backtest_result(
            group_ret[col], rf=rf, benchmark_returns=benchmark_ret, period=period
        )
        res_df = pd.DataFrame(res_dct).T
        backtest_df = pd.concat([backtest_df, res_df], axis=1)
    backtest_df.columns = group_ret.columns
    return backtest_df


def analysis_group_ret(
    factor_df,
    ret_df,
    factor_name,
    n_groups,
    mktmv_df=None,
    rf=0,
    benchmark=None,
    period="DAILY",
):
    '''
    Description
    ----------
    单分组下的因子分析

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
    mktmv_df: pandas.DataFrame.
        市值数据.格式为trade_date, stock_code, mktmv. 默认为None，即等权
    rf: float.
        无风险收益率, 默认为0
    benchmark: pandas. DataFrame.
        基准收益率数据，格式为trade_date, ret
    period: str. 指定数据频率
        有DAILY, WEEKLY, MONTHLY三种, 默认为DAILY

    Return
    ----------
    tuple. 第一个为分组回测指标，格式为pandas.DataFrame
    第二个为分组净值曲线图，第三个为因子多空净值曲线图
    '''
    group_ret = get_group_ret(factor_df, ret_df, factor_name, n_groups, mktmv_df)
    # 回测指标计算
    backtest_df = get_group_ret_backtest(group_ret, rf, benchmark, period)
    time_idx = pd.to_datetime(group_ret.index)
    factor_ret = group_ret["H-L"]
    weight_name = "等权" if mktmv_df is None else "市值加权"
    # 分组净值曲线
    group_ret = group_ret.drop(columns=["H-L"])
    plot_params_dct1 = {
        "x_lst": [time_idx] * n_groups,
        "y_lst": [(group_ret[col] + 1).cumprod() for col in group_ret.columns],
        "label_lst": group_ret.columns.tolist(),
        "xlabel": "日期",
        "ylabel": "净值",
        "fig_title": f"因子{factor_name}的分组净值曲线({weight_name})",
    }
    fig1 = utils.plot_multi_line(**plot_params_dct1)

    # 因子多空净值曲线
    factor_cumret = (1 + factor_ret).cumprod() - 1
    plot_params_dct2 = {
        "x1": time_idx,
        "y1": 100 * np.array(factor_ret).reshape(-1),
        "x2": time_idx,
        "y2": 100 * np.array(factor_cumret).reshape(-1),
        "label1": "因子多空收益率(%)",
        "label2": "因子累计多空收益率(%)",
        "xlabel": "日期",
        "ylabel1": "因子多空收益率(%)",
        "ylabel2": "因子累计收益率(%)",
        "fig_title": f"因子{factor_name}的多空收益曲线图({weight_name})",
    }
    fig2 = utils.plot_bar_line(**plot_params_dct2)
    return backtest_df, fig1, fig2


# 序贯排序双分组部分
def get_double_sort_group(
    factor1_df,
    factor2_df,
    factor1_name,
    factor2_name,
    n_groups1,
    n_groups2,
):
    """
    Description
    ----------
    获取序贯排序双分组情况

    Parameters
    ----------
    factor1_df: pandas.DataFrame.
        未提前的因子数据，格式为trade_date, stock_code, factor1_name
    factor2_df: pandas.DataFrame.
        未提前的因子数据，格式为trade_date, stock_code, factor2_name
    factor1_name: str.
        因子1名称
    factor2_name: str.
        因子2名称
    n_groups1: int.
        对因子1分组数量.
    n_groups2: int.
        对因子2分组数量.

    Return
    ----------
    pandas.DataFrame.
    列名为trade_date, stock_code, factor1_name, factor2_name, group1_name, group2_name, ret
    """
    utils._check_columns(factor1_df, ["trade_date", "stock_code", factor1_name])
    utils._check_columns(factor2_df, ["trade_date", "stock_code", factor2_name])
    factor_df = pd.merge(factor1_df, factor2_df, on=["trade_date", "stock_code"])
    # 对factor_df中的factor_name进行分组，并进行标记
    g1 = factor_df.groupby("trade_date", group_keys=False)
    factor_df = g1.apply(
        _get_single_period_group, factor_name=factor1_name, n_groups=n_groups1
    )
    # 再按照trade_date, factor1_group进行分组
    g2 = factor_df.groupby("trade_date", group_keys=False)
    factor_df = g2.apply(
        _get_single_period_group, factor_name=factor2_name, n_groups=n_groups2
    )
    return factor_df


def get_double_sort_group_ret(
    factor1_df,
    factor2_df,
    ret_df,
    factor1_name,
    factor2_name,
    n_groups1,
    n_groups2,
    mktmv_df=None,
):
    """
    Description
    ----------
    计算序贯排序双分组的收益率

    Parameters
    ----------
    factor1_df: pandas.DataFrame.
        未提前的因子数据，格式为trade_date, stock_code, factor1_name
    factor2_df: pandas.DataFrame.
        未提前的因子数据，格式为trade_date, stock_code, factor2_name
    ret_df: pandas.DataFrame.
        收益率数据，格式为trade_date, stock_code, ret
    factor1_name: str.
        因子1名称
    factor2_name: str.
        因子2名称
    n_groups1: int.
        对因子1分组数量.
    n_groups2: int.
        对因子2分组数量.
    mktmv_df: pandas.DataFrame. 默认为None，即等权
        未提前的市值数据.格式为trade_date, stock_code, mktmv

    Return
    ----------
    pandas.DataFrame.
    列名为trade_date, group1_name, group2_name, ret
    """
    utils._check_columns(ret_df, ["trade_date", "stock_code", "ret"])
    # 获取双分组结果，并提前一期
    factor_df = get_double_sort_group(
        factor1_df,
        factor2_df,
        factor1_name,
        factor2_name,
        n_groups1,
        n_groups2,
    )
    factor_df = utils.get_previous_factor(factor_df)

    # 计算分组收益
    df = pd.merge(factor_df, ret_df, on=["trade_date", "stock_code"])
    if mktmv_df is not None:
        utils._check_columns(mktmv_df, ["trade_date", "stock_code", "mktmv"])
        mktmv_df = utils.get_previous_factor(mktmv_df)
    else:
        mktmv_df = df[["trade_date", "stock_code"]].copy()
        mktmv_df["mktmv"] = 1

    def get_group_weight_ret(df):
        df = df.copy()
        df["weight"] = df["mktmv"] / df["mktmv"].sum()
        return np.sum(df["weight"] * df["ret"])

    # 计算分组
    # 此时, df格式为trade_date, stock_code, factor_name
    df = pd.merge(df, mktmv_df, on=["trade_date", "stock_code"])
    # 此时, df格式为trade_date, stock_code, factor_name, mktmv

    # 分组计算收益率
    group1_name = factor1_name + "_group"
    group2_name = factor2_name + "_group"
    g = df.groupby(["trade_date", group1_name, group2_name])
    group_ret = g.apply(get_group_weight_ret)
    group_ret = group_ret.reset_index()
    group_ret.columns = ["trade_date", group1_name, group2_name, "ret"]

    # 计算多空收益率
    for i in range(n_groups1):
        group1_idx = "Group" + str(i)
        factor_ret = pd.DataFrame()
        factor_ret["trade_date"] = sorted(list(set(group_ret["trade_date"])))
        factor_ret[group1_name] = group1_idx
        factor_ret[group2_name] = "H-L"
        long_group2_idx = "Group" + str(n_groups2 - 1)
        short_group2_idx = "Group0"
        cond1 = group_ret[group1_name] == group1_idx
        cond2_long = group_ret[group2_name] == long_group2_idx
        cond2_short = group_ret[group2_name] == short_group2_idx
        long_ret = group_ret.loc[cond1 & cond2_long, "ret"].values
        short_ret = group_ret.loc[cond1 & cond2_short, "ret"].values
        factor_ret["ret"] = long_ret - short_ret
        group_ret = pd.concat([group_ret, factor_ret])
    group_ret = group_ret.sort_values(["trade_date", group1_name, group2_name])
    group_ret = group_ret.reset_index(drop=True)
    return group_ret


def double_sort_mean(group_ret, factor1_name, factor2_name):
    """
    Description
    ----------
    计算序贯排序双分组收益率均值

    Parameters
    ----------
    group_ret: pandas.DataFrame.
        各组收益率, 格式为trade_date, group1_name, group2_name, ret
    factor1_name: str.
        因子1名称
    factor2_name: str.
        因子2名称

    Return
    ----------
    pandas.DataFrame.
    列名为Group0, Group1, …, Groupm, H-L
    索引第一层为Group0,…,Groupn
    第二层为ret_mean(%), tvalue
    """
    group1_name = factor1_name + "_group"
    group2_name = factor2_name + "_group"
    n_groups1 = len(group_ret.drop_duplicates([group1_name]))
    n_groups2 = len(group_ret.drop_duplicates([group2_name])) - 1
    ret_mean_arr = np.zeros((n_groups1, n_groups2 + 1))
    ret_t_arr = np.zeros((n_groups1, n_groups2 + 1))
    for i in range(n_groups1):
        for j in range(n_groups2 + 1):
            cond1 = group_ret[group1_name] == "Group" + str(i)
            if j < n_groups2:
                cond2 = group_ret[group2_name] == "Group" + str(j)
            else:
                cond2 = group_ret[group2_name] == "H-L"
            ret_arr = group_ret.loc[cond1 & cond2, "ret"].values
            test_dct = factor_analysis.newy_west_test(ret_arr)
            ret_mean_arr[i, j] = test_dct["ret_mean(%)"][0]
            ret_t_arr[i, j] = test_dct["t-value"][0]

    # 将t值放到均值下面调整输出结果
    res_arr = np.zeros((2 * n_groups1, n_groups2 + 1))
    for i in range(n_groups1):
        mean_idx = 2 * i
        t_idx = 2 * i - 1
        res_arr[mean_idx, :] = ret_mean_arr[i, :]
        res_arr[t_idx, :] = ret_t_arr[i, :]
    # 设置列名
    col_lst = ["Group" + str(i) for i in range(n_groups2)] + ["H-L"]
    # 建立索引
    idx1_lst = ["Group" + str(i) for i in range(n_groups1)]
    idx2_lst = ["ret_mean(%)", "t-value"]
    res_idx = pd.MultiIndex.from_product([idx1_lst, idx2_lst])
    res_df = pd.DataFrame(res_arr, columns=col_lst, index=res_idx)
    return res_df


def double_sort_backtest(
    group_ret, factor1_name, factor2_name, rf=0, benchmark=None, period="DAILY"
):
    """
    Description
    ----------
    计算序贯排序双分组的回测指标

    Parameters
    ----------
    group_ret: pandas.DataFrame.
        各组收益率, 格式为trade_date, group1_name, group2_name, ret
    factor1_name: str.
        因子1名称
    factor2_name: str.
        因子2名称
    rf: float.
        无风险收益率, 默认为0
    benchmark: pandas. DataFrame.
        基准收益率数据，格式为trade_date, ret
    period: str. 指定数据频率
        有DAILY, WEEKLY, MONTHLY三种, 默认为DAILY

    Return
    ----------
    pandas.DataFrame.
    列名为Group0, …, Groupm, H-L
    索引第一层为Group0, …, Groupn,
    第二层为年化收益率(%), 年化波动率(%), 夏普比率, 最大回撤(%)
    若benchmark不为None，则第二层会额外输出:超额年化收益率(%),
    超额年化波动率(%). 信息比率, 相对基准胜率(%), 超额收益最大回撤(%)
    """
    group1_name = factor1_name + "_group"
    group2_name = factor2_name + "_group"
    n_groups1 = len(group_ret.drop_duplicates([group1_name]))
    n_groups2 = len(group_ret.drop_duplicates([group2_name])) - 1
    if benchmark is not None:
        utils._check_columns(benchmark, ["trade_date", "ret"])
        # 选出共同的trade_date
        s1 = set(benchmark["trade_date"].tolist())
        s2 = set(group_ret["trade_date"].tolist())
        common_time_lst = sorted(list(s1.intersection(s2)))
        cond1 = group_ret["trade_date"].isin(common_time_lst)
        cond2 = benchmark["trade_date"].isin(common_time_lst)
        group_ret = group_ret.loc[cond1, :].copy()
        benchmark = benchmark.loc[cond2, :].copy()
        # 排序重设索引
        group_ret = group_ret.sort_values(["trade_date", group1_name, group2_name])
        benchmark = benchmark.sort_values("trade_date")
        group_ret = group_ret.reset_index(drop=True)
        benchmark = benchmark.reset_index(drop=True)
        benchmark_ret = benchmark["ret"]
        ind_n = 9
    else:
        benchmark_ret = None
        ind_n = 4

    res_arr = np.zeros((n_groups1 * ind_n, n_groups2 + 1))
    for i in range(n_groups1):
        for j in range(n_groups2 + 1):
            group1_idx = "Group" + str(i)
            if j == n_groups2:
                group2_idx = "H-L"
            else:
                group2_idx = "Group" + str(j)
            cond1 = group_ret[group1_name] == group1_idx
            cond2 = group_ret[group2_name] == group2_idx
            ret_arr = group_ret.loc[cond1 & cond2, "ret"].values
            temp_res_dct = backtest.get_backtest_result(
                ret_arr, rf=rf, benchmark_returns=benchmark_ret, period=period
            )
            temp_res_arr = np.array(list(temp_res_dct.values())).reshape(-1)
            res_arr[ind_n * i : ind_n * (i + 1), j] = temp_res_arr

    # 序贯排序回测结果整理
    col_lst = ["Group" + str(i) for i in range(n_groups2)] + ["H-L"]
    idx1_lst = ["Group" + str(i) for i in range(n_groups1)]
    idx2_lst = list(temp_res_dct.keys())
    res_idx = pd.MultiIndex.from_product([idx1_lst, idx2_lst])
    res_df = pd.DataFrame(res_arr, columns=col_lst, index=res_idx)
    return res_df
