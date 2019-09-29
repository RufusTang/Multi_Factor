"""
- ѡ�����ӣ�6����֪���������
- ���ݴ�������ȱʧֵ
- ѡ��Ȩ�أ�
- ���������С�����10�飬�ڼ���Ϊ������÷�
- ���ӽ���Ӵ�С��10�飬�ڼ���Ϊ������÷�
- ѡ�ɷ�Χ��
  - ѡ�ɵ�ָ����ģ�飺����300
  
- 3���������ڣ�
- ���֣�ÿ�½���һ�ε���ѡ��20��������ǰ�Ĺ�Ʊ
- ���׹��������ѳ��еĹ�Ʊ
- �����µĹ�Ʊ�ص��еĹ�Ʊ

"""

# ���뺯����
from jqdata import *
from jqfactor import *
import datetime as dt
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import LinearRegression


# ��ʼ���������趨��׼�ȵ�
def initialize(context):

    # �趨����300��Ϊ��׼
    set_benchmark('000300.XSHG')
    # ������̬��Ȩģʽ(��ʵ�۸�)
    set_option('use_real_price', True)
    log.set_level('order', 'error')


    ### ��Ʊ����趨 ###
    # ��Ʊ��ÿ�ʽ���ʱ���������ǣ�����ʱӶ�����֮��������ʱӶ�����֮����ǧ��֮һӡ��˰, ÿ�ʽ���Ӷ����Ϳ�5��Ǯ
    set_slip_fee(context)
    
    
    # ��������
    # �ֹ����������հ���ɸѡѡȡ����ǰ��λ�Ĺ�Ʊ��Ϊ��ѡ��Ʊ
    g.stocks_num  = 2
    
    # ����������ֵ�ʱ���ã���ͬ�����в�ͬ�ĵ÷�
    g.groups_num  = 10
    
    # ���Ӱ�����market_cap��pe_ratio��pb_ratio��return_on_invested_capital��inc_revenue��inc_profit_before_tax
    # g.factor_pool = ['market_cap','pe_ratio_lyr','pb_ratio','inc_return','gross_profit_margin','inc_net_profit_year_on_year','inc_total_revenue_annual']
    # g.factor_pool = ['market_cap','pe_ratio_lyr','pb_ratio','inc_return','inc_net_profit_year_on_year','inc_total_revenue_annual']
    # g.factor_pool = ['pe_ratio_lyr','inc_net_profit_year_on_year','inc_return','market_cap','inc_total_revenue_annual','pb_ratio']
    g.factor_pool = ['market_cap','pe_ratio_lyr','pb_ratio','inc_return','inc_net_profit_year_on_year','inc_total_revenue_annual']
    # g.factor_pool = ['pe_ratio_lyr','inc_return','market_cap','inc_total_revenue_annual','inc_net_profit_year_on_year','pb_ratio']
    
    # �Ƿ���Ҫȡ����
    # �ж��߼�����Խ��Խ�õ�ԭ��������
    # ���ʵ����ԽСԽ�õ����Ӿ�ȡ������һ�����ܸ�ֵ��Ӱ�죬��һ����ͳһ��׼
    g.backward_pool = [1,1,1,0,0,1]
    
    # ����ѡȡ���ӵ���������Ϊ�ز���ʱ����
    g.factor_num = 4
    
    
    # ������ȫ�ֱ������飬���ڴ���ɸѡ���
    # �ǵ���ÿ�β�����ɺ�Ҫ���㣬���ⲻ��Ҫ�Ĳ��
    g.tobuy_list = []
    
    ## ���к�����reference_securityΪ����ʱ��Ĳο���ģ�����ı��ֻ���������֣���˴���'000300.XSHG'��'510300.XSHG'��һ���ģ�
    # ����ǰ����
    # ���й�Ʊ����ѡ
    run_monthly(before_market_open, monthday = 1, time='before_open', reference_security='000300.XSHG')
    # ����ʱ����
    run_monthly(market_open, monthday = 1, time='open', reference_security='000300.XSHG')


    # ���й�Ʊ����ѡ
    # run_weekly(before_market_open, 1, time='before_open', reference_security='000300.XSHG')
    # ����ʱ����
    # run_weekly(market_open, 1, time='open', reference_security='000300.XSHG')

# ���ݲ�ͬ��ʱ������û�����������
def set_slip_fee(context):
    # ����������Ϊ0
    set_slippage(FixedSlippage(0)) 
    
    # ���ݲ�ͬ��ʱ�������������
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



#���ÿ��й�Ʊ�أ��޳�(�����ࡢ)st��ͣ�ƹ�Ʊ����������
def get_feasible_stocks(context):
    
    # ��ʵ����ǱȽϹؼ���һ��
    # ѡ�����Ĺ�ƱUniverse
    # ��һ���ڴ˲������ϲ���
    s = get_index_stocks('000300.XSHG', date=context.current_dt)
    #print '�����Ʊ����Ϊ��%s'%len(s)
    all_stocks = s
    #�õ��Ƿ�ͣ����Ϣ��dataframe��ͣ�Ƶ�1��δͣ�Ƶ�0
    suspended_info_df = get_price(list(all_stocks), end_date = context.current_dt, count = 1, frequency = 'daily', fields = 'paused')['paused'].T
    #����δͣ�ƹ�Ʊ ����dataframe
    suspended_index = suspended_info_df.iloc[:,0] == 1
    #�õ�����ͣ�ƹ�Ʊ�Ĵ���list:
    suspended_stocks = suspended_info_df[suspended_index].index.tolist()

    #�޳�ͣ�ƹ�Ʊ
    for stock in suspended_stocks:
        if stock in all_stocks:
            all_stocks.remove(stock)

    return all_stocks   


## ����ǰ���к���
def before_market_open(context):
    # ��ʼ��
    g.tobuy_list = []

    # ���
    g.tobuy_list = get_candidate_list(context)
    
def get_candidate_list(context):
    
    # ���շ��ص�����
    candidate_list = []
    
    
    industry_list = []
    
    industry_list = get_industries(name='sw_l1', date=context.current_dt - dt.timedelta(days=1)).index

    for i in range(0,len(industry_list)):
        # ��ȡ��ҵ�����б�
        industry_i =  industry_list[i]
        # print "��ҵ��%s"%str(industry_i)
        
        # ������ҵ������ȡ��Ӧ�Ĺ�Ʊ
        sec_pool = []
        sec_pool = get_industry_stocks(industry_i,date=context.current_dt - dt.timedelta(days=1))
        # print "��Ʊ�أ�%s"%str(sec_pool)
        
        
        Secs_Chosen = []
        Secs_Chosen = Sec_Choose(context,sec_pool)
        # print "ѡ����Ʊ��%s"%str(Secs_Chosen)
        
        candidate_list =  candidate_list + Secs_Chosen


    return list(set(candidate_list))
                

def Sec_Choose(context,sec_pool):
    ret_list = []
    
    # 1����ȡ��������
    # ���Ӱ�����market_cap��pe_ratio��pb_ratio��return_on_invested_capital��inc_revenue��inc_profit_before_tax
    #ȡ����300��Ϊ��Ʊ��
    feasible_stocks = sec_pool

    # ��¼��Ӧscore��pandas����
    # ���ն�score��pandas �����������
    factor_data_pd = pd.DataFrame(index = feasible_stocks)
    score_pd = pd.DataFrame(index = feasible_stocks)

    # 1����ȡ��������
    # ��ʼ�����ӿ⡢group�����ѭ��
    # ѭ���Ľ���ŵ�score_pd��
    # ����ѡȡ���ٸ�����ֵ
    for i in range(0, g.factor_num):
    # for i in range(0,1):

        # ����ʼֵ
        # �Ƿ�������ӻع飬�Ƿ���е��ã�֮ǰ�������Ѿ�����
        factor_i = g.factor_pool[i]
        backward_i = g.backward_pool[i]

        # ����
        factor = pd.DataFrame()

        # ��ȡ��ֵ����
        # ȷ���ο�������Ϊ��ֵ����
        factor_refer_name = 'market_cap'
        
        # �������ֵ���ӣ��������Ի���ֱ�ӽ����׼��
        if factor_i == factor_refer_name:
            # ��ȡ����ֵ
            factor = get_factor(context,factor_i,context.current_dt - dt.timedelta(days=1),backward_i,sec_pool)
            #ȥ��ֵ
            factor = mad(factor)

        # ���������ֵ���ӣ���������Ի������Ի�����б�׼��
        else:
            # ��ȡ����ֵ
            factor = get_factor(context,factor_i,context.current_dt - dt.timedelta(days=1),backward_i,sec_pool)
            #ȥ��ֵ
            factor = mad(factor)


            # ��ȡ�ο�����ֵ����
            factor_refer = get_factor(context,factor_refer_name,context.current_dt - dt.timedelta(days=1),1,sec_pool)
            
            # ������Ӧ�����Ի�����
            factor = neutral(factor,factor_refer)
            
        #��׼��        
        factor = stand(factor)

        # �����������
        factor_data_pd = pd.concat([factor_data_pd,factor], axis=1)

    # ���ֵ����Ϊ�߼���Խ��Խ�ã�����ѡ��-1�������
    factor_data_pd = factor_data_pd.fillna(-1)
    
    
    # 2�����д��
    # factor_data_pd�Ѿ���g.factor_num������������
    for col in factor_data_pd.columns:
        # �����Ӵ�С�Ӵ�С��������
        # sorted_factor = pd.DataFrame(factor_data_pd.sort_values(by=col, ascending=False)[col])
        # ��ܵ���ְ����python2�Ĵ��룬������sort
        sorted_factor = pd.DataFrame(factor_data_pd.sort(col, ascending=False)[col])

        # �����ӡ�_score����
        # ��ָ���ֵ
        sorted_factor[col+"_score"] = 0

        # ÿ���Ʊ������
        # ����ȡ��
        stock_count = len(sorted_factor) // g.groups_num

        # ��ʼ���
        # ��������Ĵ�ֹ��򣺷�ֵԽС����Ӧ������ֵԽ��
        # ���ۣ���ֵԽС��Խ����
        for i in range(g.groups_num):
            if i == g.groups_num - 1:
                sorted_factor[col+"_score"][i*stock_count:] = i+1
            else:
                sorted_factor[col+"_score"][i*stock_count:(i+1)*stock_count] = i+1    

        # �ϲ��������ӵķ���
        factor_data_pd = pd.concat([factor_data_pd, sorted_factor[col+"_score"]], axis=1)
    
    # �����Ӻ�
    # ����ԽСԽ��
    # Ĭ�����������У�������ѡ������С�ļ�ֻ��Ʊ
    # sum_score = factor_data_pd.iloc[:, len(g.factor_pool):].sum(1).sort_values()
    # ����汾��������


    # pthon3 �汾
    # sum_score = factor_data_pd.iloc[:, g.factor_num:].sum(1).sort_values()
    
    # python2 �汾
    factor_data_pd['total_score'] = factor_data_pd.iloc[:, g.factor_num:].sum(1)
    sum_score = factor_data_pd.sort('total_score', ascending=True)['total_score']

    # ѡ�ɽ��
    ret_list = list(sum_score[:g.stocks_num].index)    
    
    return ret_list


def mad(factor):
    # 3����λ��ȥ��ֵ
    # ȷ���������
    factor_index = factor.index
    factor_columns = factor.columns[0]
    
    
    # �������ֵ����λ��
    med = np.median(factor)

    # �������ֵ����λ���Ĳ�ֵ�����о���ֵ
    mad = np.median(abs(factor - med))

    # ���弸������λ��������
    high = med + (3 * 1.4826 * mad)
    low = med - (3 * 1.4826 * mad)

    # �滻�����������ֵ
    factor = np.where(factor > high, high, factor)
    factor = np.where(factor < low, low, factor)
    
    # ��Ϊnp�����Ὣ������[x]��Ϊ[[x]]�����ݣ����Խ��д���
    ret = pd.DataFrame([d[0] for d in factor],index = factor_index, columns = [factor_columns])
    return ret


def stand(factor):
    mean = factor.mean()
    std = factor.std()
    return (factor - mean)/std
    
# factor1����������һ��Ϊ��Ҫ�����Ի��Ĳ���
# factor2������������һ����Ϊ�ο�����һ������ֵ��Ϊ���Ի��Ĳο�����
def neutral(factor1,factor2):
    factor1 = factor1.fillna(0)
    factor2 = factor2.fillna(0)
    
    # ȷ���������
    factor_index = factor1.index
    factor_columns = factor1.columns[0]

    # ����LinearRegression��Ҫ����Ҫ��������reshape�Ĵ���
    # python 3 ��������һ��������Ҫ����values������
    x = factor2.values.reshape(-1,1)
    
    y = factor1.values.reshape(-1,1)
    try:
        # �����ع鷽�̲�Ԥ��
        lr = LinearRegression()
        lr.fit(x, y)
        y_predict = lr.predict(x)
    except:
        y_predict = y
        
    # ȥ�����ԵĹ�ϵ�����������Ϊ�����ӵ�ֵ
    res = y - y_predict    
    
    ret = pd.DataFrame([d[0] for d in res],index = factor_index, columns = [factor_columns])
    
    return ret    
        

def get_factor(context,factor_name,date,if_backward,sec_pool):
    #��ȡ���Ų����������ָ������
    val = get_fundamentals(query(valuation).limit(1)).columns.tolist()
    bal = get_fundamentals(query(balance).limit(1)).columns.tolist()
    cf = get_fundamentals(query(cash_flow).limit(1)).columns.tolist()
    inc = get_fundamentals(query(income).limit(1)).columns.tolist()
    ind = get_fundamentals(query(indicator).limit(1)).columns.tolist()


    # ��ȡ��Ӧ��ָ�����룬���Զ���ʹ��
    # stock = get_index_stocks('000300.XSHG', date)

    # ��ȡ��Ӧ�Ŀ��еĴ��룬�����Զ���ʹ�ã�ֻ���ڻز⻷����ʹ��
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



## ����ʱ���к���
def market_open(context):
    #���֣���������Ʊ
    for stock in context.portfolio.long_positions:
        if stock not in g.tobuy_list:
            order_target_value(stock, 0)

    #�������¹�Ʊ
    total_value = context.portfolio.total_value # ��ȡ���ʲ�
    for i in range(len(g.tobuy_list)):
        value = total_value / len(g.tobuy_list) # ȷ��ÿ����ĵ�Ȩ��
        order_target_value(g.tobuy_list[i], value) # ���������Ŀ��Ȩ��
    
    g.tobuy_list = []

