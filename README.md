<!--
 * @Author: dkl
 * @Description: 因子回测框架V2说明
 * @Date: 2023-07-09 08:36:44
-->
# 因子回测框架说明
本框架包含了常见的因子处理和回测函数，并给出了示例。

## 更新
2023-07-09: 修正了市值因子没提前一期的bug，增加了单分组回测结果可视化、序贯排序等函数

## 文件说明
1. `data`: 数据文件夹
2. `example.ipynb`：示例文件
3. `preprocess.py`: 数据预处理文件
4. `group_calc.py`: 分组回测文件
5. `factor_analysis.py`: 因子分析模块
6. `backtest.py`: 回测模块
7. `utils.py`: 工具性函数

## 函数文档说明
### `preprocess.py` 
1. del_outlier
```
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
```

2. standardize
```
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
```

3. neutralize
```
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
```

### `group_calc.py`
1. get_stock_group
```
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
```

2. get_group_ret
```
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
```

3. get_group_ret_backtest
```
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
```

4. analysis_group_ret
```
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
```


5. get_double_sort_group
```
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
```

6. get_double_sort_group_ret
```
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
```

7.  double_sort_mean
```
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
```

8. double_sort_backtest
```
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
```

### `factor_analysis.py`
1. get_factor_ic
```
Description
----------
计算因子IC序列

Parameters
----------
factor_df: pandas.DataFrame. 未提前的因子数据.

Return
----------
pandas.DataFrame.
```

2. newy_west_test:
```
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
```

3. analysis_factor_ic
```
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
```


4. risk_adj_alpha
```
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
```


5. fama_macbeth_reg
```
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
Dict.输出示例为: {
        "factor_name": factor_name_lst,
        "beta": list(fama_macbeth.params[1:]),
        "t-value": list(fama_macbeth.tstats[1:]),
        "R-square": fama_macbeth.rsquared,
        "Average-Obs": fama_macbeth.time_info[0],
    }
```


### `backtest.py`
1. net_value
```
Description
----------
计算净值曲线

Parameters
----------
returns: array_like. 收益率序列

Return
----------
numpy.ndarray. 净值曲线
```

2. previous_peak
```
Description
----------
计算历史最大净值

Parameters
----------
returns: array_like. 收益率序列

Return
----------
numpy.ndarray. 历史最大净值
```


3. drawdown
```
Description
----------
计算回撤序列，回撤=净值现值/历史最大净值-1

Parameters
----------
returns: array_like. 收益率序列

Return
----------
numpy.ndarray. 单位为%
```


4. max_drawdown
```
Description
----------
计算最大回撤

Parameters
----------
returns: array_like. 收益率序列

Return
----------
float. 最大回撤. 单位为%
```


5. annualized_return
```
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
```


6. annualized_volatility
```
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
```


7. annualized_sharpe
```
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
```


8. er_annual_return
```
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
```

9. er_annual_volatility
```
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
```

10. information_ratio
```
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
```

11. er_max_drawdown
```
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
```

12. winrate
```
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
```

13. get_backtest_result
```
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
```


### `utils.py`
1. get_previous_factor
```
Description
----------
获取上期因子值

Parameters
----------
factor_df: pandas.DataFrame. 输入因子数据,必须含有trade_date

Return
----------
pandas.DataFrame. 上期因子值
```


2. stackdf
```
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
```


3. unstackdf
```
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
```


4. get_last_date
```
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
```


5. get_next_date
```
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
```

6. plot_bar_line
```
Description
----------
绘制双坐标的柱状图和线形图

Parameters
----------
x1: array_like, 柱状图横坐标
y1: array_like, 柱状图纵坐标
x2: array_like, 线形图横坐标
y2: array_like, 线形图纵坐标
xlabel: str. 横坐标标签
ylabel1: str. 柱状图纵坐标标签
ylabel2: str. 柱状图纵坐标标签
fig_title: str. 图片标题

Return
----------
figure.
```

7. plot_multi_line
```
Description
----------
绘制多根折线图

Parameters
----------
x_lst: array_like, 折线图横坐标列表
y_lst: array_like, 折线图纵坐标列表
label_lst: array_like, 折线图标签列表
xlabel: str. 折线图横坐标标签
ylabel: str. 折线图纵坐标标签
fig_title: str. 图片标题

Return
----------
figure.
```