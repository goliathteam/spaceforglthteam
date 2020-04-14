#GICS_SUB_INDUSTRY_NAME
import bql
from IPython.display import display, clear_output
from functools import reduce
from itertools import chain
import numpy as np
import ipywidgets
import pandas as pd
import scipy.stats as st
import datetime
from dateutil.relativedelta import relativedelta

bq = bql.Service()
f = bq.func
d = bq.data

classification = d.GICS_SECTOR_NAME()
show_transparency = False
FACTORS_WINSORIZE = ['FCF/Net_Income','net_debt_to_equity', 'working_cap_to_total_assets']

#*************************************************************
#
#     Get Universe
#
#*************************************************************

def get_univ(idx, as_of_date):
    """ we get the members of 1 or more indices, exclude those with no sector
    classification. And then get the unique Ids and finally sort the Ids. """
    orig_univ = bq.univ.members(idx, dates = as_of_date)
    cond = classification != np.nan
    cond_1 = d.ID() !='BION SE Equity'
    cond_2 = classification != 'Financials'
    all_cond = f.and_(cond, cond_1)
    all_cond = f.and_(all_cond, cond_2)
    univ = bq.univ.filter(orig_univ, all_cond)
    
    unique_univ = get_unique_ids(univ)
    return unique_univ

def get_unique_ids(univ):
    """ Different index might have the same stock but on a different listing
    with this function we get the fundamental ticker, and then the unique tickers
    """
    # request fundamental id
    req = bql.Request(univ, {'f_id': d.fundamental_ticker()})
    res = bq.execute(req)
    fundamental_id = res[0].df()
    
    # unique values
    fundamental_id.f_id = fundamental_id.f_id.apply(get_eqy_ticker)
    unique_tickers = fundamental_id.f_id.tolist()
    unique_tickers = list(set(unique_tickers))
    
    if '  Equity' in unique_tickers: unique_tickers.remove('  Equity')
        
    return unique_tickers

def get_eqy_ticker(ticker):
    ticker_list = ticker.strip().split(' ')

    if len(ticker_list) == 2:
        ticker = '{} {} Equity'.format(ticker_list[0][:-2], ticker_list[0][-2:])

    return ticker


#*************************************************************
#
#     Get all SCORES
#
#*************************************************************

def get_scores_all(univ, as_of_dt_val, fx):  
    flds = get_req_flds(as_of_dt_val)
    requests = [bql.Request(univ, {k:flds[k]},  with_params = {'currency':fx}) for k in flds.keys()]
    responses = list(bq.execute_many(requests))
    return responses

def get_req_flds(dt):
    
    params = get_data_obj_params(dt)
    
    quality_fields = get_scores_quality(dt, params)
    momentum_fields = get_scores_momentum(dt, params)
    risk_fields = get_scores_risk(dt, params)
    value_fields = get_scores_value(dt, params)
    ref_fields = get_ref_fields()
    
    all_flds = dict(chain(quality_fields.items(), 
                          momentum_fields.items(), 
                          risk_fields.items(), 
                          value_fields.items(), 
                          ref_fields.items() ))
    return all_flds

def get_ref_fields():
    return ({'country':d.COUNTRY_FULL_NAME(),
             'sector':classification,
             'name':d.name() })

def dt_str_to_datetime(dt_str):
    s = list(dt_str.split('-'))
    a = [int(x) for x in s]
    dt = datetime.datetime(a[0], a[1], a[2])
    return dt

def get_data_obj_params(as_of_dt_val):
    fy1_param = {'FPO':'1', 
                 'FPT':'A', 
                 'AS_OF_DATE': as_of_dt_val, 
                 'FILL' : 'prev'}
    
    hist_5y_param = {'FPO':f.range(-4,0), 
                     'FPT':'A', 
                     'AS_OF_DATE':as_of_dt_val, 
                     'FILL' : 'prev'}
    
    last_pt_param = {'FPT':'LTM', 
                     'AS_OF_DATE':as_of_dt_val, 
                     'FILL' : 'prev'}
    
    aod_yr_str = as_of_dt_val[:4]
    dt = dt_str_to_datetime(as_of_dt_val)
    aod_minus_1y = dt - relativedelta(years = 1)
    est_1y_ago_params = {'FPO': '1', 
                         'FPT' : 'A', 
                         'FPR' : aod_yr_str, 
                         'FILL' : 'prev', 
                         'AS_OF_DATE' : aod_minus_1y}
    
    return {'fy1':fy1_param,
            'hist_5y':hist_5y_param,
            'last_pt':last_pt_param,
            'est_1y_ago':est_1y_ago_params}

#*************************************************************
#
#     BQL response to DF
#
#*************************************************************

def bq_res_array_to_pd(res_array):
    a = res_array[0][0].df().reset_index()
    r_all = pd.DataFrame({'ID':a['ID']})
    for res in res_array:
        for r in res:
            r_n = r.df().reset_index()
            r_all = r_all.merge(r_n, on = 'ID')
    return r_all

def rem_unnecesary_cols(df):
    df = df.loc[:,~df.columns.duplicated()]
    df = df[df.columns.drop(list(df.filter(regex='REVISION_DATE')))]
    df = df[df.columns.drop(list(df.filter(regex='PERIOD_END_DATE')))]
    df = df[df.columns.drop(list(df.filter(regex='AS_OF_DATE_')))]
    df = df[df.columns.drop(list(df.filter(regex='ID_DATE')))]
    df = df[df.columns.drop(list(df.filter(regex='CURRENCY')))]
    return df
   

#****************************************************************
#
#    Quality scores
#
#****************************************************************    

def get_scores_quality(as_of_dt_val, params):
    fy1_param = params['fy1'] 
    hist_5y_param = params['hist_5y'] 
    last_pt_param = params['last_pt']
    flds  = {}
    
    # flds['roce_fy1'] = d.RETURN_COM_EQY( **fy1_param )
    # Hay estimaciones para lo de arriba pero son pocas y malas, sera major 
    # usar la estimacion del net income dividido por el shareholders equity del ultimo period. 
    net_income = d.net_income( ** fy1_param)
    sh_equity = d.TOTAL_EQUITY( **last_pt_param)
    flds['roce_fy1'] = net_income / sh_equity
    
    flds['roce_avg_5y'] = f.AVG(d.RETURN_COM_EQY( **hist_5y_param ))
    flds['roce_std'] = f.std(d.RETURN_COM_EQY( **hist_5y_param ))
    
    flds['fcf_fy1'] = d.cf_free_cash_flow( **fy1_param )
    flds['fcf_avg_5y'] = f.AVG(d.cf_free_cash_flow( **hist_5y_param ))
    flds['fcf_std'] = f.std(d.cf_free_cash_flow( **hist_5y_param ))
    
    # CALIDAD - FCF/Net Income
    fcf_last = d.cf_free_cash_flow( **last_pt_param )
    net_income_last = d.NET_INCOME( **last_pt_param )
    flds['FCF/Net_Income'] = fcf_last / net_income_last    
    return flds


#****************************************************************
#
#    Momentum scores
#
#****************************************************************

def get_scores_momentum(as_of_dt_val, params):
    flds  = {}
    fy1_param = params['fy1'] 
    hist_5y_param = params['hist_5y'] 
    last_pt_param = params['last_pt']
    est_1y_ago_params = params['est_1y_ago']

    flds['sales_growth'] = d.SALES_GROWTH( **last_pt_param )
    flds['eps_growth'] = d.EPS_GROWTH( **last_pt_param )
    
    # MOMENTUM - EPS revisions 
    eps_param = {'fpt':'A','fpo':'1f','AS_OF_DATE':as_of_dt_val}
    is_eps = d.is_eps( **eps_param )
    # In future distributions, we will move estimate revision functions to bq.func
    try:
        up = d.contributor_revisions(is_eps, 'NETUP', '100D')
        down = d.contributor_revisions(is_eps, 'NETDN', '100D')
        count = d.contributor_count(is_eps)
    except AttributeError:
        up = f.contributor_revisions(is_eps, 'NETUP', '100D')
        down = f.contributor_revisions(is_eps, 'NETDN', '100D')
        count = f.contributor_count(is_eps)
    flds['eps_revisions'] = (up - down) / count
        
    # MOMENTUM - ROCE n > ROCE n-1 
    # (current year end RETURN_COM_EQY est / estimate one year ago current year end RETURN_COM_EQY estimate)
    # 23 Jul 18 Inaki change to act instead of est
    curr_y_roce_act = d.RETURN_COM_EQY( dates = as_of_dt_val, fpo = '0', fpt = 'A' )
    curr_y_roce_act_1y_ago = d.RETURN_COM_EQY( dates = as_of_dt_val, fpo = '-1', fpt = 'A' )
    flds['roce_act_chg'] = curr_y_roce_act / curr_y_roce_act_1y_ago
    
    # MOMENTUM - Leverage n <  Leverage n-1 
    # (current year end net debt est / estimate one year ago current year end net debt estimate)
    curr_y_net_debt_est = d.NET_DEBT( **fy1_param )
    curr_y_net_debt_est_1y_ago = d.NET_DEBT( **est_1y_ago_params )
    flds['net_debt_est_chg'] = curr_y_net_debt_est / curr_y_net_debt_est_1y_ago
    
    # MOMENTUM - Price performance vs sector
    dt_rng = f.range('-1y', as_of_dt_val)
    px_series = d.PX_LAST(dates = dt_rng, fill = 'prev', ca_adj = 'full')
    px_1y = f.first(px_series)
    px_td = f.last(px_series)
    px_perf_1y = (px_td - px_1y) / px_1y
    perf_sector_avg = f.groupavg(px_perf_1y, classification)
    #flds['px_perf_vs_sector'] =  (px_perf_1y - perf_sector_avg) / perf_sector_avg
    # 17 May 18 - change in model suggested by Inaki
    flds['px_perf_vs_sector'] =  (px_perf_1y - perf_sector_avg) 
    # 23 Jul 18 - change in model suggested by Inaki
    #flds['px_perf_vs_sector'] = px_perf_1y  
    
    return flds


#****************************************************************
#
#    Risk scores
#
#****************************************************************

def get_scores_risk(as_of_dt_val, params):
    flds  = {}
    fy1_param = params['fy1'] 
    hist_5y_param = params['hist_5y'] 
    last_pt_param = params['last_pt']
    
    last_y_param = {'FPT':'A', 
                    'AS_OF_DATE':as_of_dt_val, 
                    'FILL' : 'prev'}
    
    curr_y_net_debt_est = d.NET_DEBT( **fy1_param )
    ebitda_act = d.ebitda( **last_y_param )
    flds['net_debt_to_ebitda_est'] = curr_y_net_debt_est / ebitda_act
    
    # RISK - Net Debt / EBITDA
    net_int_exp = d.IS_NET_INTEREST_EXPENSE( **last_y_param )
    ebit = d.EBIT( **last_y_param )
    flds['interest_coverage_ratio'] = net_int_exp / ebit
    
    # RISK - Interest / EBIT (últimos 12 meses)
    oper_cf = d.CF_CASH_FROM_OPER( **last_y_param )
    flds['interest_to_cash_from_ops'] = net_int_exp / oper_cf
    
    # RISK - Net Debt / Equity
    total_eqy = d.TOTAL_EQUITY( **last_pt_param )
    flds['net_debt_to_equity'] = curr_y_net_debt_est / total_eqy
    
    # RISK - Working Capital / total Assets (últimos 12 meses)
    working_capital = d.WORKING_CAPITAL( **last_pt_param )
    total_assets = d.BS_TOT_ASSET( **last_pt_param )
    flds['working_cap_to_total_assets'] = curr_y_net_debt_est / total_eqy
    
    # RISK - Analysts recommedations - buy recommendations % of all rec
    rec_buy_num = d.TOT_BUY_REC(DATES = as_of_dt_val)
    rec_all_num = d.TOT_ANALYST_REC(DATES = as_of_dt_val)
    flds['buy_rec_percent_all_rec'] = rec_buy_num / rec_all_num
    
    return flds

#****************************************************************
#
#    Value scores
#
#****************************************************************

def get_scores_value(as_of_dt_val, params):
    flds  = {}
    fy1_param = params['fy1'] 
    hist_5y_param = params['hist_5y'] 
    last_pt_param = params['last_pt']

    # VALUE - ev_ebitda (FY1) Y
    flds['ev_ebitda_est'] = d.ev_to_ebitda( **fy1_param )
    
    # VALUE - EV/EBITDA (vs historia) N
    ev_ebitda_5y_avg = f.AVG(d.ev_to_ebitda( **hist_5y_param ))
    flds['ev_ebitda_vs_hist'] = d.ev_to_ebitda( **fy1_param ) / ev_ebitda_5y_avg
    
    # VALUE - FCF/EV yield (FY1) Y
    fcf_fy1 = d.cf_free_cash_flow( **fy1_param )
    ev_est = d.ENTERPRISE_VALUE( **fy1_param )
    flds['fcf_to_ev'] = fcf_fy1 / ev_est
    
    # VALUE - P/BV (FY1) Y
    flds['px_to_book_value_est'] = d.PX_TO_BOOK_RATIO( **fy1_param )
    
    # VALUE - P/BV (vs historia) Y
    pb = d.PX_TO_BOOK_RATIO( **fy1_param )
    pb_5y_avg = f.AVG(d.PX_TO_BOOK_RATIO( **hist_5y_param ))    
    flds['px_to_book_value_est_vs_hist'] = pb / pb_5y_avg
    
    # VALUE - PER (FY1) Y
    flds['per_est'] = d.PE_RATIO( **fy1_param )
    
    # VALUE - PER (vs historia) Y
    pe = d.PE_RATIO( **fy1_param ) 
    pe_5y_avg = f.AVG(d.PE_RATIO( **hist_5y_param ))
    flds['per_vs_hist'] = pe / pe_5y_avg

    return flds


#****************************************************************
#
#    Main output
#
#****************************************************************

def get_data(dt, fx, idx, transparency):
    global show_transparency
    show_transparency = transparency
    univ = get_univ(idx, dt)
    preliminary_data = get_scores_all(univ, dt, fx)
    df = bq_res_array_to_pd(preliminary_data)
    df = rem_unnecesary_cols(df)
    calculate_scores(df)
    return df

def cust_winsorize(s, num_std):
    """
    scipy.stats.mstats import winsorize cant clean based on 
    standard deviation, and replaces nan with values 
    
    Takes pandas series, and a number of standard deviations. Then
    calculates mean +- std dev * num std dev. And winsorizes on max/min 
    resulting values after removing values above /below mean +- std dev * num std dev
    """
    # get current values for +- num std dev
    avg = s.mean()
    std = s.std()
    hi_cut_off = avg + num_std * std
    lo_cut_off = avg - num_std * std
    
    # eliminate values greater than std dev
    s_clean = np.where(s >= hi_cut_off, np.nan, s)
    s_clean = np.where(s_clean <= lo_cut_off, np.nan, s_clean)
    
    # find resulting min and max after previous step
    s_clean_min = np.nanmin(s_clean)
    s_clean_max = np.nanmax(s_clean)
    
    # winsorize
    res_df = np.where(s > hi_cut_off, s_clean_max, s)
    res_df = np.where(res_df <=  lo_cut_off, s_clean_min, res_df)
    return pd.Series(res_df)


def to_score(all_df, c_name, higher_better):
    global FACTORS_WINSORIZE
    
    col = all_df[c_name]
    sector = all_df['sector']
    
    # no infinite values
    col = col.replace('inf', np.nan)
    col = col.replace('-inf', np.nan)
    
    # winsorize 
    if c_name in FACTORS_WINSORIZE:
        col = cust_winsorize(col, 1.5)
        all_df[c_name +'_winsorize'] = col 
    
    # shift base to abs(min)
    min_val = abs(col.min())
    all_df[c_name +'_adj'] = col + min_val 
    
    # get sector average
    sector_avg = c_name + '_sector_avg' 
    all_df[sector_avg] = all_df[c_name +'_adj'].groupby(sector).transform(np.mean)
    
    # sum sector averages
    sector_avgs = all_df[['sector', sector_avg]]
    sector_avgs = sector_avgs.drop_duplicates('sector')
    sum_sector_avg = sector_avgs[sector_avg].sum()
    
    all_df[sector_avg + '_sum'] = sum_sector_avg
    all_df[sector_avg + '_weight'] = all_df[sector_avg] / all_df[sector_avg + '_sum'] 
    
    
    # layering
    layer = c_name + '_layer'
    all_df[layer] = np.where(np.isnan(all_df[c_name +'_adj']), 
                            all_df[sector_avg + '_weight'],
                            all_df[c_name +'_adj'] / all_df[sector_avg] * all_df[sector_avg + '_weight'])
    
    # layering_min 
    min_layer_val = abs(all_df[layer].min())
    all_df[c_name + '_layer_min'] = min_layer_val
    
    # log
    all_df[c_name + '_log'] = np.log(all_df[layer]  + all_df[c_name + '_layer_min'] + 0.000001 )
    
    # zscore
    z_score_sign = -1
    if higher_better == True:
        z_score_sign = 1
    
    log = c_name + '_log'
    all_df[log +'_avg'] = all_df[log].mean()
    all_df[log + '_std'] = all_df[log].std()
    all_df[c_name + '_zscore'] = (( all_df[log] - all_df[log +'_avg']) / all_df[log + '_std'] ) * z_score_sign
    
    # to probability
    np.warnings.filterwarnings('ignore')
    zscore_col = all_df[c_name + '_zscore']
    all_df[c_name + '_score'] = np.where(np.isnan(zscore_col), np.nan, st.norm.cdf(zscore_col) * 100)
             
def is_fld_higher_better(fld):
    lower_better_fld = ['roce_std', 
                        'fcf_std', 
                        'net_debt_to_ebitda_est',
                        'interest_coverage_ratio',
                        'interest_to_cash_from_ops',
                        'net_debt_to_equity',
                        'working_cap_to_total_assets',
                        'ev_ebitda_est',
                        'ev_ebitda_vs_hist',
                        'ev_ebitda_vs_sector_est',
                        'fcf_to_ev',
                        'px_to_book_value_est',
                        'px_to_book_value_est_vs_hist',
                        'px_to_book_value_vs_sector_est',
                        'per_est',
                        'per_vs_hist',
                        'per_vs_sector_est']
    if fld in lower_better_fld:
        return False
    else:
        return True
    
def calculate_scores(data):
    data = factor_to_score(data)
    aggregate_scores(data)
    
    if show_transparency == False:
        return data[['ID', 'name', 'country', 'sector', 
                     'total_score', 'calidad', 'momentum',
                     'riesgo', 'valoracion' ]]
    else:
        return data
    
def factor_to_score(data):
    score_list = ['roce_fy1',
                  'roce_avg_5y', 
                  'roce_std', 
                  'fcf_fy1', 
                  'fcf_avg_5y', 
                  'fcf_std', 
                  'FCF/Net_Income', 
                  'sales_growth',
                  'eps_growth', 
                  'eps_revisions', 
                  'roce_act_chg', 
                  'net_debt_est_chg', 
                  'px_perf_vs_sector', 
                  'net_debt_to_ebitda_est',
                  'interest_coverage_ratio', 
                  'interest_to_cash_from_ops', 
                  'net_debt_to_equity', 
                  'working_cap_to_total_assets', 
                  'buy_rec_percent_all_rec', 
                  'ev_ebitda_est',
                  'ev_ebitda_vs_hist', 
                  'fcf_to_ev', 
                  'px_to_book_value_est', 
                  'px_to_book_value_est_vs_hist', 
                  'per_est', 
                  'per_vs_hist']
        
    for score in score_list:
        is_higher_better = is_fld_higher_better(score)
        to_score(data, score, is_higher_better)
    return data

def handle_nas(col_list, df):
    """ If 40% + of the values are NA then we drop all values, 
    else we substitute with avg of all"""
    for col in col_list:
        num_rows = len(df[col].values)
        num_na = sum(pd.isnull(df[col]))
        if num_na / num_rows > 0.4:
            df[col] = np.nan
        else:
            df[col] = df[col].fillna((df[col].mean()))

def aggregate_scores(df):
    score_list = ['roce_fy1_score', 
                  'roce_avg_5y_score', 
                  'roce_std_score', 
                  'fcf_fy1_score', 
                  'fcf_avg_5y_score', 
                  'fcf_std_score', 
                  'FCF/Net_Income_score', 
                  'sales_growth_score', 
                  'eps_growth_score', 
                  'eps_revisions_score', 
                  'roce_act_chg_score', 
                  'net_debt_est_chg_score', 
                  'px_perf_vs_sector_score', 
                  'net_debt_to_ebitda_est_score', 
                  'interest_coverage_ratio_score', 
                  'interest_to_cash_from_ops_score', 
                  'net_debt_to_equity_score', 
                  'working_cap_to_total_assets_score', 
                  'buy_rec_percent_all_rec_score', 
                  'ev_ebitda_est_score', 
                  'ev_ebitda_vs_hist_score', 
                  'fcf_to_ev_score', 
                  'px_to_book_value_est_score', 
                  'px_to_book_value_est_vs_hist_score', 
                  'per_est_score', 
                  'per_vs_hist_score']

    handle_nas(score_list, df)
    
    calidad = df[['roce_fy1_score', 
                  'roce_avg_5y_score', 
                  'roce_std_score', 
                  'fcf_fy1_score', 
                  'fcf_avg_5y_score', 
                  'fcf_std_score', 
                  'FCF/Net_Income_score' ]]
    df['calidad'] =  calidad.mean(axis=1)

    momentum= df[['sales_growth_score',
                  'eps_growth_score',
                  'eps_revisions_score',
                  'roce_act_chg_score',
                  'net_debt_est_chg_score',
                  'px_perf_vs_sector_score']]
    df['momentum'] =  momentum.mean(axis=1)
    
    
    riesgo= df[['net_debt_to_ebitda_est_score',
                'interest_coverage_ratio_score',
                'interest_to_cash_from_ops_score',
                'net_debt_to_equity_score',
                'working_cap_to_total_assets_score',
                'buy_rec_percent_all_rec_score']]
    df['riesgo'] =  riesgo.mean(axis=1)

    valoracion= df[['ev_ebitda_est_score',
                    'ev_ebitda_vs_hist_score',
                    'fcf_to_ev_score',
                    'px_to_book_value_est_score',
                    'px_to_book_value_est_vs_hist_score',
                    'per_est_score',
                    'per_vs_hist_score']]
    df['valoracion'] =  valoracion.mean(axis=1)
    

    
#     df['calidad'] = (   df['roce_fy1_score'] * 0.1428571 +
#                         df['roce_avg_5y_score'] * 0.1428571 + 
#                         df['roce_std_score'] * 0.1428571 +
#                         df['fcf_fy1_score'] * 0.1428571 +
#                         df['fcf_avg_5y_score'] * 0.1428571 +
#                         df['fcf_std_score'] * 0.1428571 +
#                         df['FCF/Net_Income_score'] * 0.1428571 )

#     df['momentum'] = (  df['sales_growth_score'] * 0.166667 +
#                         df['eps_growth_score'] * 0.166667 +
#                         df['eps_revisions_score'] * 0.166667 +
#                         df['roce_est_chg_score'] * 0.166667 +
#                         df['net_debt_est_chg_score'] * 0.166667 +
#                         df['px_perf_vs_sector_score'] * 0.166667 )

#     df['riesgo'] = (df['net_debt_to_ebitda_est_score'] * 0.166667 +
#                     df['interest_coverage_ratio_score'] * 0.166667 +
#                     df['interest_to_cash_from_ops_score'] * 0.166667 +
#                     df['net_debt_to_equity_score'] * 0.166667 +
#                     df['working_cap_to_total_assets_score'] * 0.166667 +
#                     df['buy_rec_percent_all_rec_score'] * 0.166667 )
        
#     df['valoracion'] = (df['ev_ebitda_est_score'] * 0.1428571 +
#                         df['ev_ebitda_vs_hist_score'] * 0.1428571 +
#                         df['fcf_to_ev_score'] * 0.1428571 +
#                         df['px_to_book_value_est_score'] * 0.1428571 + 
#                         df['px_to_book_value_est_vs_hist_score'] * 0.1428571 +
#                         df['per_est_score'] * 0.1428571 +
#                         df['per_vs_hist_score'] * 0.1428571 )
        
    df['total_score'] = (df['calidad'] * 0.25 + 
                         df['momentum'] * 0.25 + 
                         df['riesgo'] * 0.25 + 
                         df['valoracion'] * 0.25)