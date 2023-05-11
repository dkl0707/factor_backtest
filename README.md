# 因子回测框架说明
本框架包含了常见的因子处理和回测函数，并给出了示例。
## 文件说明
1. `data`: 数据文件夹
2. `example.ipynb`：示例文件
3. `preprocess.py`: 数据预处理文件
4. `group_calc.py`: 分组回测文件
5. `factor_analysis.py`: 因子分析模块
6. `backtest.py`: 回测模块
7. `utils.py`: 工具性函数

## 函数说明
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
pandas.DataFrame.去极值后的因子面板
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
pandas.DataFrame.标准化后的因子面板
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
pandas.DataFrame.中性化后的因子面板
```
### `group_calc.py`
1. get_group_ret
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

2. get_stock_group
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
2. get_group_ret_analysis
```
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
```
3. newy_west_test(arr, factor_name='factor'):
```
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
````
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


8. get_backtest_result
```
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
