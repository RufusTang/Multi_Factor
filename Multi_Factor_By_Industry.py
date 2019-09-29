"""
- 选股因子：6个已知方向的因子
- 数据处理：处理缺失值
- 选股权重：
- 因子升序从小到大分10组，第几组为所在组得分
- 因子降序从大到小分10组，第几组为所在组得分
- 选股范围：
  - 选股的指数、模块：沪深300
  
- 3、调仓周期：
- 调仓：每月进行一次调仓选出20个排名靠前的股票
- 交易规则：卖出已持有的股票
- 买入新的股票池当中的股票

"""

# 导入函数库
from jqdata import *
from jqfactor import *
import datetime as dt
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import LinearRegression


# 初始化函数，设定基准等等
def initialize(context):

    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    log.set_level('order', 'error')


    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_slip_fee(context)
    
    
    # 参数设置
    # 持股数量，最终按照筛选选取排名前几位的股票作为候选股票
    g.stocks_num  = 2
    
    # 分组数，打分的时候用，不同分组有不同的得分
    g.groups_num  = 10
    
    # 因子包括：market_cap，pe_ratio，pb_ratio，return_on_invested_capital，inc_revenue，inc_profit_before_tax
    # g.factor_pool = ['market_cap','pe_ratio_lyr','pb_ratio','inc_return','gross_profit_margin','inc_net_profit_year_on_year','inc_total_revenue_annual']
    # g.factor_pool = ['market_cap','pe_ratio_lyr','pb_ratio','inc_return','inc_net_profit_year_on_year','inc_total_revenue_annual']
    # g.factor_pool = ['pe_ratio_lyr','inc_net_profit_year_on_year','inc_return','market_cap','inc_total_revenue_annual','pb_ratio']
    g.factor_pool = ['market_cap','pe_ratio_lyr','pb_ratio','inc_return','inc_net_profit_year_on_year','inc_total_revenue_annual']
    # g.factor_pool = ['pe_ratio_lyr','inc_return','market_cap','inc_total_revenue_annual','inc_net_profit_year_on_year','pb_ratio']
    
    # 是否需要取倒数
    # 判断逻辑按照越大越好的原则来设置
    # 如果实际是越小越好的因子就取倒数，一方面规避负值的影响，另一方面统一标准
    g.backward_pool = [1,1,1,0,0,1]
    
    # 设置选取因子的数量，作为回测框架时调用
    g.factor_num = 4
    
    
    # 买卖的全局变量数组，用于传递筛选结果
    # 记得在每次操作完成后要清零，避免不必要的差错
    g.tobuy_list = []
    
    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
    # 开盘前运行
    # 进行股票的挑选
    run_monthly(before_market_open, monthday = 1, time='before_open', reference_security='000300.XSHG')
    # 开盘时运行
    run_monthly(market_open, monthday = 1, time='open', reference_security='000300.XSHG')


    # 进行股票的挑选
    # run_weekly(before_market_open, 1, time='before_open', reference_security='000300.XSHG')
    # 开盘时运行
    # run_weekly(market_open, 1, time='open', reference_security='000300.XSHG')

# 根据不同的时间段设置滑点与手续费
def set_slip_fee(context):
    # 将滑点设置为0
    set_slippage(FixedSlippage(0)) 
    
    # 根据不同的时间段设置手续费
    dt=context.current_dt

    if dt>datetime.datetime(2013,1, 1):
        set_commission(PerTrade(buy_cost=0.0003, 
                                sell_cost=0.0013, 
                                min_cost=5)) 

    elif dt>datetime.datetime(2011,1, 1):
        set_commission(PerTrade(buy_cost=0.001, 
                                sell_cost=0.002, 
                                min_cost=5))

    elif dt>datetime.datetime(2009,1, 1):
        set_commission(PerTrade(buy_cost=0.002, 
                                sell_cost=0.003, 
                                min_cost=5))

    else:
        set_commission(PerTrade(buy_cost=0.003, 
                                sell_cost=0.004, 
                                min_cost=5))



#设置可行股票池，剔除(金融类、)st、停牌股票，输入日期
def get_feasible_stocks(context):
    
    # 其实这个是比较关键的一步
    # 选择合理的股票Universe
    # 下一步在此步基础上操作
    s = get_index_stocks('000300.XSHG', date=context.current_dt)
    #print '输入股票个数为：%s'%len(s)
    all_stocks = s
    #得到是否停牌信息的dataframe，停牌得1，未停牌得0
    suspended_info_df = get_price(list(all_stocks), end_date = context.current_dt, count = 1, frequency = 'daily', fields = 'paused')['paused'].T
    #过滤未停牌股票 返回dataframe
    suspended_index = suspended_info_df.iloc[:,0] == 1
    #得到当日停牌股票的代码list:
    suspended_stocks = suspended_info_df[suspended_index].index.tolist()

    #剔除停牌股票
    for stock in suspended_stocks:
        if stock in all_stocks:
            all_stocks.remove(stock)

    return all_stocks   


## 开盘前运行函数
def before_market_open(context):
    # 初始化
    g.tobuy_list = []

    # 获得
    g.tobuy_list = get_candidate_list(context)
    
def get_candidate_list(context):
    
    # 最终返回的数据
    candidate_list = []
    
    
    industry_list = []
    
    industry_list = get_industries(name='sw_l1', date=context.current_dt - dt.timedelta(days=1)).index

    for i in range(0,len(industry_list)):
        # 获取行业编码列表
        industry_i =  industry_list[i]
        # print "行业：%s"%str(industry_i)
        
        # 根据行业编码提取相应的股票
        sec_pool = []
        sec_pool = get_industry_stocks(industry_i,date=context.current_dt - dt.timedelta(days=1))
        # print "股票池：%s"%str(sec_pool)
        
        
        Secs_Chosen = []
        Secs_Chosen = Sec_Choose(context,sec_pool)
        # print "选出股票：%s"%str(Secs_Chosen)
        
        candidate_list =  candidate_list + Secs_Chosen


    return list(set(candidate_list))
                

def Sec_Choose(context,sec_pool):
    ret_list = []
    
    # 1、获取因子数据
    # 因子包括：market_cap，pe_ratio，pb_ratio，return_on_invested_capital，inc_revenue，inc_profit_before_tax
    #取沪深300作为股票池
    feasible_stocks = sec_pool

    # 记录相应score的pandas数组
    # 最终对score的pandas 数组进行排序
    factor_data_pd = pd.DataFrame(index = feasible_stocks)
    score_pd = pd.DataFrame(index = feasible_stocks)

    # 1、获取因子数据
    # 开始对因子库、group库进行循环
    # 循环的结果放到score_pd中
    # 看看选取多少个因子值
    for i in range(0, g.factor_num):
    # for i in range(0,1):

        # 赋初始值
        # 是否进行因子回归，是否进行倒置，之前设置中已经设置
        factor_i = g.factor_pool[i]
        backward_i = g.backward_pool[i]

        # 重置
        factor = pd.DataFrame()

        # 提取市值因子
        # 确定参考的因子为市值因子
        factor_refer_name = 'market_cap'
        
        # 如果是市值因子，则不用中性化，直接进入标准化
        if factor_i == factor_refer_name:
            # 获取因子值
            factor = get_factor(context,factor_i,context.current_dt - dt.timedelta(days=1),backward_i,sec_pool)
            #去极值
            factor = mad(factor)

        # 如果不是市值因子，则进行中性化，中性化后进行标准化
        else:
            # 获取因子值
            factor = get_factor(context,factor_i,context.current_dt - dt.timedelta(days=1),backward_i,sec_pool)
            #去极值
            factor = mad(factor)


            # 提取参考的市值因子
            factor_refer = get_factor(context,factor_refer_name,context.current_dt - dt.timedelta(days=1),1,sec_pool)
            
            # 进行相应的中性化操作
            factor = neutral(factor,factor_refer)
            
        #标准化        
        factor = stand(factor)

        # 进行数据组合
        factor_data_pd = pd.concat([factor_data_pd,factor], axis=1)

    # 填充值，因为逻辑是越大越好，所以选择-1进行填充
    factor_data_pd = factor_data_pd.fillna(-1)
    
    
    # 2、进行打分
    # factor_data_pd已经被g.factor_num控制了因子数
    for col in factor_data_pd.columns:
        # 按因子大小从大到小降序排列
        # sorted_factor = pd.DataFrame(factor_data_pd.sort_values(by=col, ascending=False)[col])
        # 框架调用职能用python2的代码，所以用sort
        sorted_factor = pd.DataFrame(factor_data_pd.sort(col, ascending=False)[col])

        # 新增加“_score”列
        # 打分赋初值
        sorted_factor[col+"_score"] = 0

        # 每组股票的数量
        # 整数取整
        stock_count = len(sorted_factor) // g.groups_num

        # 开始打分
        # 按照下面的打分规则：分值越小，相应的因子值越大
        # 结论：分值越小，越厉害
        for i in range(g.groups_num):
            if i == g.groups_num - 1:
                sorted_factor[col+"_score"][i*stock_count:] = i+1
            else:
                sorted_factor[col+"_score"][i*stock_count:(i+1)*stock_count] = i+1    

        # 合并所有因子的分数
        factor_data_pd = pd.concat([factor_data_pd, sorted_factor[col+"_score"]], axis=1)
    
    # 分数加和
    # 分数越小越好
    # 默认是升序排列，所以挑选分数最小的几只股票
    # sum_score = factor_data_pd.iloc[:, len(g.factor_pool):].sum(1).sort_values()
    # 代码版本兼容问题


    # pthon3 版本
    # sum_score = factor_data_pd.iloc[:, g.factor_num:].sum(1).sort_values()
    
    # python2 版本
    factor_data_pd['total_score'] = factor_data_pd.iloc[:, g.factor_num:].sum(1)
    sum_score = factor_data_pd.sort('total_score', ascending=True)['total_score']

    # 选股结果
    ret_list = list(sum_score[:g.stocks_num].index)    
    
    return ret_list


def mad(factor):
    # 3倍中位数去极值
    # 确定输入变量
    factor_index = factor.index
    factor_columns = factor.columns[0]
    
    
    # 求出因子值的中位数
    med = np.median(factor)

    # 求出因子值与中位数的差值，进行绝对值
    mad = np.median(abs(factor - med))

    # 定义几倍的中位数上下限
    high = med + (3 * 1.4826 * mad)
    low = med - (3 * 1.4826 * mad)

    # 替换上下限以外的值
    factor = np.where(factor > high, high, factor)
    factor = np.where(factor < low, low, factor)
    
    # 因为np处理后会将数据由[x]变为[[x]]的数据，所以进行处理
    ret = pd.DataFrame([d[0] for d in factor],index = factor_index, columns = [factor_columns])
    return ret


def stand(factor):
    mean = factor.mean()
    std = factor.std()
    return (factor - mean)/std
    
# factor1：主参数，一般为需要被中性化的参数
# factor2：被动参数，一般作为参考想象，一般是市值作为中性化的参考参数
def neutral(factor1,factor2):
    factor1 = factor1.fillna(0)
    factor2 = factor2.fillna(0)
    
    # 确定输入变量
    factor_index = factor1.index
    factor_columns = factor1.columns[0]

    # 根据LinearRegression的要求需要将参数做reshape的处理
    # python 3 在这里有一错误，所以要增加values的设置
    x = factor2.values.reshape(-1,1)
    
    y = factor1.values.reshape(-1,1)
    try:
        # 建立回归方程并预测
        lr = LinearRegression()
        lr.fit(x, y)
        y_predict = lr.predict(x)
    except:
        y_predict = y
        
    # 去除线性的关系，留下误差作为该因子的值
    res = y - y_predict    
    
    ret = pd.DataFrame([d[0] for d in res],index = factor_index, columns = [factor_columns])
    
    return ret    
        

def get_factor(context,factor_name,date,if_backward,sec_pool):
    #获取五张财务基础所有指标名称
    val = get_fundamentals(query(valuation).limit(1)).columns.tolist()
    bal = get_fundamentals(query(balance).limit(1)).columns.tolist()
    cf = get_fundamentals(query(cash_flow).limit(1)).columns.tolist()
    inc = get_fundamentals(query(income).limit(1)).columns.tolist()
    ind = get_fundamentals(query(indicator).limit(1)).columns.tolist()


    # 获取相应的指数代码，可以独立使用
    # stock = get_index_stocks('000300.XSHG', date)

    # 获取相应的可行的代码，不可以独立使用，只能在回测环境中使用
    # stock = get_feasible_stocks(context)
    stock = sec_pool
    if factor_name in val:
        q = query(valuation).filter(valuation.code.in_(stock))
        df = get_fundamentals(q, date)
        
    elif factor_name in bal:
        q = query(balance).filter(balance.code.in_(stock))
        df = get_fundamentals(q, date)
        
    elif factor_name in cf:
        q = query(cash_flow).filter(cash_flow.code.in_(stock))
        df = get_fundamentals(q, date)


    elif factor_name in inc:
        q = query(income).filter(income.code.in_(stock))
        df = get_fundamentals(q, date)
    
    elif factor_name in ind:
        q = query(indicator).filter(indicator.code.in_(stock))
        df = get_fundamentals(q, date)
        

    ret_pd = pd.DataFrame()

    if if_backward:
        ret_pd[factor_name] = np.array(1/df[factor_name])
        ret_pd['code'] = np.array(df['code'])
        ret_pd = ret_pd.set_index('code')
    else:
        ret_pd[factor_name] = np.array(df[factor_name])
        ret_pd['code'] = np.array(df['code'])
        ret_pd = ret_pd.set_index('code')

    return ret_pd



## 开盘时运行函数
def market_open(context):
    #调仓，先卖出股票
    for stock in context.portfolio.long_positions:
        if stock not in g.tobuy_list:
            order_target_value(stock, 0)

    #再买入新股票
    total_value = context.portfolio.total_value # 获取总资产
    for i in range(len(g.tobuy_list)):
        value = total_value / len(g.tobuy_list) # 确定每个标的的权重
        order_target_value(g.tobuy_list[i], value) # 调整标的至目标权重
    
    g.tobuy_list = []

