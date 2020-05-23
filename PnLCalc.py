import datetime
import pandas as pd
import numpy as np
import os
from API_Lib import load_obj

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 5000)
pd.set_option('max_row', 10)

static_directory = 'C:\\XGBoostProduction\\'
working_directory = 'X:\\Research\\'

run_DART_PnL = True
run_find_offer_prices = False
lmp_filename = '2020_01_05_LMP_DATA_DICT_MASTER'
dart_backtest_filename = 'Backtest_2020_05_04_BACKTEST_DATA_DICT_MASTER_ERCOT_EXP20_SPREAD__'
# dart_backtest_filename = 'Backtest_09_11_2019_GBM_DATA_MISO_V8.0_MASTER_159F_MISO_EXP10_'
# dart_backtest_filename = 'Backtest_09_11_2019_GBM_DATA_PJM_V8.0_MASTER_207F_PJM_EXP10_'
# dart_backtest_filename = 'backtest_PJM_V8.0_all'

dart_sd_location_filter = 'SD1000'  # Leave Blank For No Filter Otherwise Use 'SD4, SD3.5 etc' Format

name_adder = ''

dart_scale_mean_div_sd = False # Keep False
limit_daily_mws = True # True increases compute time greatly. If false, scales to max hour limitations but not daily limits
limit_hourly_mws = True
dart_sd_band = 1.00
dart_cutoff_dolMW = 1.00
cutoff_max_hourly_loss = 100000 #Positive Value!
dart_start_date = datetime.datetime(2014, 8, 24)
dart_end_date = datetime.datetime(2020, 9, 11)

run_spread_PnL = False
spread_cutoff = 0.00
spread_mean_band = 1
max_spreads_per_node = 2
tier2_filter = False
tier2_sd_filter = 'SD1000'
tier2_PnL_cutoff = 0

run_backtest_compare = False # this probably doenst work anymore
equalize_dates = True
equalize_nodes = True
compare_old_filename = 'Backtest_Tier2_Backtest_master_allMISO_FullDataset2015_2019_DataDict_MISO_CV4_EXP5_OLDNODES_SD3.5'
compare_new_filename = 'Backtest_Tier2_Backtest_master_allMISO_FullDataset2015_2019_DataDict_MISO_CV4_EXP5_OLDNODES_SD3.5'
compare_old_filename_short = 'Tier2_0EVBand'
compare_new_filename_short = 'Tier2_10EVBand'

lmp_dict = {'EST':None, 'EPT':None, 'CPT':None}
if run_find_offer_prices:
    lmp_dict = load_obj(static_directory+ '\\ModelUpdateData\\'+lmp_filename)

if 'MISO' in dart_backtest_filename:
    max_trade_mws = 5
    max_hourly_inc_mws = 50
    max_hourly_dec_mws = 50
    target_mws = 1000
    top_hourly_locs = 10
    tier2_backtest = 'Backtest_Tier2_Backtest_Exps_2020_02_24_BACKTEST_DATA_DICT_MASTER_MISO_EXP20_'
    dart_inc_mean_band_peak = 1.25
    dart_inc_mean_band_offpeak = 1.00
    dart_dec_mean_band_peak = dart_inc_mean_band_peak  # Positive Value!
    dart_dec_mean_band_offpeak = dart_inc_mean_band_offpeak  # Positive Value!
    lmp_df = lmp_dict['EST']
elif 'PJM' in dart_backtest_filename:
    max_trade_mws = 5
    max_hourly_inc_mws = 150
    max_hourly_dec_mws = 150
    top_hourly_locs = 30
    target_mws = 2000
    tier2_backtest = 'Backtest_Tier2_Backtest_Exps_2020_02_24_BACKTEST_DATA_DICT_MASTER_PJM_EXP20_'
    dart_inc_mean_band_peak = 0.75
    dart_inc_mean_band_offpeak = 0.75
    dart_dec_mean_band_peak = dart_inc_mean_band_peak  # Positive Value!
    dart_dec_mean_band_offpeak = dart_inc_mean_band_offpeak  # Positive Value!
    lmp_df = lmp_dict['EPT']
elif 'SPP' in dart_backtest_filename:
    max_trade_mws = 2
    max_hourly_inc_mws = 30
    max_hourly_dec_mws = 30
    target_mws = 500
    tier2_backtest = 'Backtest_daily_Tier2_Backtest_12092019_Master_Nodes_Dataset_Dict_SPP_EXP10_'
    top_hourly_locs = 10
    dart_inc_mean_band_peak = 1.00
    dart_inc_mean_band_offpeak = 0.75
    dart_dec_mean_band_peak = dart_inc_mean_band_peak  # Positive Value!
    dart_dec_mean_band_offpeak = dart_inc_mean_band_offpeak  # Positive Value!
    lmp_df = lmp_dict['CPT']
elif 'ERCOT' in dart_backtest_filename:
    max_trade_mws = 2
    max_hourly_inc_mws = 25
    max_hourly_dec_mws = 25
    target_mws = 400
    tier2_backtest = 'Backtest_daily_Tier2_Backtest_12092019_Master_Nodes_Dataset_Dict_ERCOT_EXP10_'
    top_hourly_locs = 15
    dart_inc_mean_band_peak = 1.0
    dart_inc_mean_band_offpeak = 1.0
    dart_inc_mean_band_offpeak = dart_inc_mean_band_peak
    dart_dec_mean_band_peak = dart_inc_mean_band_peak  # Positive Value!
    dart_dec_mean_band_offpeak = dart_inc_mean_band_offpeak  # Positive Value!
    lmp_df = lmp_dict['CPT']
elif 'ISONE' in dart_backtest_filename:
    max_trade_mws = 2
    max_hourly_inc_mws = 15
    max_hourly_dec_mws = 15
    target_mws = 300
    tier2_backtest = 'Backtest_daily_Tier2_Backtest_12092019_Master_Nodes_Dataset_Dict_ISONE_EXP10_'
    top_hourly_locs = 10
    dart_inc_mean_band_peak = 1.25
    dart_inc_mean_band_offpeak = 1.25
    dart_dec_mean_band_peak = dart_inc_mean_band_peak  # Positive Value!
    dart_dec_mean_band_offpeak = dart_inc_mean_band_offpeak  # Positive Value!
    lmp_df = lmp_dict['EPT']

min_trade_mws = min(max_hourly_dec_mws/top_hourly_locs,max_hourly_inc_mws/top_hourly_locs)

def calc_hourly_pnl(backtest_filename, sd_band, inc_mean_band_peak, dec_mean_band_peak, inc_mean_band_offpeak, tier2_PnL_cutoff,tier2_filter,tier2_backtest, dec_mean_band_offpeak, scale_mean_div_sd, start_date, end_date, dart_sd_location_filter,top_hourly_locs, max_trade_mws, min_trade_mws, target_mws, max_hourly_inc_mws, max_hourly_dec_mws,limit_daily_mws,limit_hourly_mws,tier2_sd_filter, working_directory,static_directory):
    load_directory = static_directory + '\BacktestFiles\\'
    # CALCULATES HOURLY PnL AND OUTPUTS DICT WITH DATAFRAMES FOR TOT/INC/DEC PnL and MWs
    input_df=pd.read_csv(load_directory+dart_backtest_filename+'.csv',index_col=['Date','HE'],parse_dates=True)

    input_df.fillna(axis=0,value=0,inplace=True)
    input_df.dropna(axis=0, inplace=True)
    input_df = input_df[(input_df.index.get_level_values('Date')>=start_date) & (input_df.index.get_level_values('Date')<=end_date)]

    pred_df = input_df[[col for col in input_df.columns if 'pred' in col]].copy()
    sd_df = input_df[[col for col in input_df.columns if 'sd' in col]].copy()
    act_df = input_df[[col for col in input_df.columns if 'act' in col]].copy()

    pred_df.columns = [col.replace('_pred','') for col in pred_df.columns]
    sd_df.columns = [col.replace('_sd', '') for col in sd_df.columns]
    act_df.columns = [col.replace('_act', '') for col in act_df.columns]

    pred_df = pred_df[[col for col in pred_df.columns if col[-3:]==dart_sd_location_filter[-3:]]]
    sd_df = sd_df[[col for col in sd_df.columns if col[-3:]==dart_sd_location_filter[-3:]]]
    act_df = act_df[[col for col in act_df.columns if col[-3:]==dart_sd_location_filter[-3:]]]

    pred_df.columns = [col.split('_SD')[0] for col in pred_df.columns]
    sd_df.columns = [col.split('_SD')[0] for col in sd_df.columns]
    act_df.columns = [col.split('_SD')[0] for col in act_df.columns]

    # Apply tier 2 waive-off
    if tier2_filter:
        tier2_backtest_df = pd.read_csv(load_directory + tier2_backtest + '.csv', index_col=['Date','HE'],parse_dates=True)
        tier2_backtest_df = tier2_backtest_df[[col for col in tier2_backtest_df.columns if 'pred' in col]].copy()
        tier2_backtest_df.columns = [col.replace('_pred', '') for col in tier2_backtest_df.columns]
        tier2_backtest_df = tier2_backtest_df[[col for col in tier2_backtest_df.columns if col[-3:] == tier2_sd_filter[-3:]]]
        tier2_backtest_df.reset_index(inplace=True)
        blank_df = pd.DataFrame(index=pred_df.index)
        blank_df.reset_index(inplace=True)
        tier2_backtest_df = pd.merge(blank_df, tier2_backtest_df,on=['Date','HE'],how='outer')
        tier2_backtest_df.set_index(['Date','HE'],inplace=True,drop=True)
        tier2_backtest_df.columns = [col.split('_PnL',1)[0] for col in tier2_backtest_df.columns]
        tier2_backtest_df.fillna(-1000,inplace=True)
        for location in pred_df.columns:
            try:
                #Set preds to zero if the tier2 PnL threashold is not met
                pred_df.loc[(tier2_backtest_df[location] < tier2_PnL_cutoff), location] = 0
            except:
                # Drop the pred if there is no tier2 data for it
                print('Dropping ' + location + '. It is in Tier 1 but not not in Tier2 backtest.')
                pred_df.drop(columns=[location],inplace=True)
                sd_df.drop(columns=[location], inplace=True)
                act_df.drop(columns=[location], inplace=True)

    # Apply Filters

    # Use this snippet to take the abs value of the preds when ranking
    pred_df = pred_df.mask(abs(pred_df).rank(axis=1, method='min', ascending=False) > top_hourly_locs, 0)


    # Use this snippet to not take the abs value of the inc and dec ranks when ranking (results in more spreads per hour)
    # dec_preds_tier1_df = pred_df.mask(pred_df.rank(axis=1, method='min', ascending=True) > top_hourly_locs, 0)
    # inc_preds_tier1_df = pred_df.mask(pred_df.rank(axis=1, method='min', ascending=False) > top_hourly_locs, 0)
    #
    # pred_df = pd.concat([dec_preds_tier1_df,inc_preds_tier1_df])
    #
    # pred_df.reset_index(inplace=True)
    # pred_df = pred_df.groupby(['Date','HE']).sum()
    # pred_df = pred_df.sort_values(by=['Date','HE'])


    for col in pred_df.columns:
        # SD Filter
        pred_df.loc[abs(pred_df[col]) < (sd_df[col] * sd_band), col] = 0

        # OnPeak and OffPeak median bands
        for hour in pred_df.index.get_level_values('HE').unique():
            if hour in [1,2,3,4,5,6,23,24]:
                pred_df.loc[(pred_df[col]>0) & (pred_df.index.get_level_values('HE') ==hour) & (pred_df[col]< inc_mean_band_offpeak), col] = 0
                pred_df.loc[(pred_df[col]<0) & (pred_df.index.get_level_values('HE') ==hour) & (pred_df[col]>-dec_mean_band_offpeak), col] = 0
            else:
                pred_df.loc[(pred_df[col] > 0) & (pred_df.index.get_level_values('HE')==hour) & (pred_df[col] < inc_mean_band_peak), col] = 0
                pred_df.loc[(pred_df[col] < 0) & (pred_df.index.get_level_values('HE') ==hour) & (pred_df[col] > -dec_mean_band_peak), col] = 0




    # MW Volumne
    if scale_mean_div_sd:
        mw_df = round(abs(pred_df/sd_df),1)
    else:
        mw_df = round(abs(pred_df/pred_df),1)


    if limit_daily_mws:
        # Scale daily MWs to meet hourly caps and daily caps
        scaled_mw_df = pd.DataFrame()
        orig_target_mws = target_mws
        for day in mw_df.index.get_level_values('Date').unique():
            day_mw_df = mw_df.loc[mw_df.index.get_level_values('Date')==day].copy()
            day_pred_df = pred_df.loc[pred_df.index.get_level_values('Date')==day].copy()
            print(day)
            target_mws = orig_target_mws

            counter = 0
            hour_counter = 25

            while (target_mws > 0) and (hour_counter > 1):
                mws = day_mw_df.sum().sum()
                scaling_factor = min(target_mws / mws, max_trade_mws)
                day_mw_df = day_mw_df * scaling_factor

                for location in day_mw_df.columns:
                    day_mw_df.loc[(day_mw_df[location] > max_trade_mws), location] = max_trade_mws

                day_mw_df['INC_Hourly_Total_MW'] = day_mw_df[day_pred_df > 0].sum(axis=1)
                day_mw_df['DEC_Hourly_Total_MW'] = day_mw_df[day_pred_df < 0].sum(axis=1)

                hour_counter = 1
                for hour in day_mw_df.index.get_level_values('HE'):
                    hourly_mw_df = day_mw_df[day_mw_df.index.get_level_values('HE') == hour]
                    hourly_pred_df = day_pred_df[day_pred_df.index.get_level_values('HE') == hour]
                    inc_hourly_total = hourly_mw_df['INC_Hourly_Total_MW'][0]
                    dec_hourly_total = hourly_mw_df['DEC_Hourly_Total_MW'][0]

                    if (inc_hourly_total >= max_hourly_inc_mws) or (dec_hourly_total >= max_hourly_dec_mws):
                        if inc_hourly_total == 0: inc_hourly_total = max_hourly_inc_mws
                        if dec_hourly_total == 0: dec_hourly_total = max_hourly_dec_mws

                        inc_ratio = max_hourly_inc_mws / inc_hourly_total
                        dec_ratio = max_hourly_dec_mws / dec_hourly_total
                        smallest_ratio = min(inc_ratio, dec_ratio)

                        # Preserve the ratio of INC to DEC trades within the hour but ensure neither breech their respective hourly caps
                        hourly_mw_df = hourly_mw_df * smallest_ratio

                        # Ensure no trades are below the minimum trade size
                        for trade in hourly_mw_df.columns:
                            # if (hourly_mw_df[trade][0] < min_trade_mws / 2) and (hourly_mw_df[trade][0] > 0):hourly_mw_df[trade][0] = 0
                            if (hourly_mw_df[trade][0] < min_trade_mws) and (hourly_mw_df[trade][0] > 0):hourly_mw_df[trade][0] = min_trade_mws
                            if (hourly_mw_df[trade][0] > max_trade_mws) and (hourly_mw_df[trade][0] > 0):hourly_mw_df[trade][0] = max_trade_mws


                        hourly_mws = hourly_mw_df.sum().sum() - hourly_mw_df['INC_Hourly_Total_MW'].sum() - hourly_mw_df['DEC_Hourly_Total_MW'].sum()
                        # Reduce target MWs by the number of MWs in the 'full' hour and add the full hour to the final trades df
                        target_mws = target_mws - hourly_mws
                        scaled_mw_df = pd.concat([scaled_mw_df, hourly_mw_df], sort=True)

                        # Drop the 'full' hour from the mw matrix
                        day_pred_df.drop(index=hourly_pred_df.index, inplace=True)
                        day_mw_df.drop(index=hourly_mw_df.index, inplace=True)
                        mw_df.drop(index=hourly_mw_df.index, inplace=True)
                        hour_counter += 1

                    # Ensure no trades are below the minimum trade size
                    for trade in hourly_mw_df.columns:
                        # if (hourly_mw_df[trade][0] < min_trade_mws / 2) and (hourly_mw_df[trade][0] > 0):hourly_mw_df[trade][0] = 0
                        if (hourly_mw_df[trade][0] < min_trade_mws) and (hourly_mw_df[trade][0] > 0): hourly_mw_df[trade][0] = min_trade_mws
                        if (hourly_mw_df[trade][0] > max_trade_mws) and (hourly_mw_df[trade][0] > 0): hourly_mw_df[trade][0] = max_trade_mws

                counter += 1

            scaled_mw_df = pd.concat([day_mw_df, scaled_mw_df], sort=True)
            scaled_mw_df.sort_values(['Date','HE'],inplace=True, ascending=True)
            scaled_mw_df['INC_Hourly_Total_MW'] = scaled_mw_df[pred_df > 0].sum(axis=1)
            scaled_mw_df['DEC_Hourly_Total_MW'] = scaled_mw_df[pred_df < 0].sum(axis=1)
        mw_df = scaled_mw_df

    elif limit_hourly_mws:
        #####  LIMIT MAX HOURLY MWs
        for index, row in mw_df.iterrows():
            row_sum = row.sum()
            if row_sum>max_hourly_inc_mws:
                mw_df.loc[index] = max_hourly_inc_mws/(mw_df.loc[index]*row_sum)


    # PnL Calc
    PnL_df = (np.sign(pred_df)*np.sign(act_df))*abs(act_df)*mw_df

    inc_mw_df=mw_df[pred_df>0].fillna(0)
    dec_mw_df=mw_df[pred_df<0].fillna(0)
    inc_PnL_df=PnL_df[pred_df>0].fillna(0)
    dec_PnL_df=PnL_df[pred_df<0].fillna(0)

    output_dict = {'mw_df':mw_df,
                   'PnL_df':PnL_df,
                   'inc_mw_df': inc_mw_df,
                   'inc_PnL_df': inc_PnL_df,
                   'dec_mw_df': dec_mw_df,
                   'dec_PnL_df': dec_PnL_df}

    print('Hourly PnL Complete: '+backtest_filename)
    print('')

    return output_dict

def calc_summary_pnl(input_dict):
    # CALCULATES SUMMARY STATISTICS

    mw_df =input_dict['mw_df']
    PnL_df =input_dict['PnL_df']
    inc_mw_df =input_dict['inc_mw_df']
    inc_PnL_df =input_dict['inc_PnL_df']
    dec_mw_df =input_dict['dec_mw_df']
    dec_PnL_df =input_dict['dec_PnL_df']

    summary_df = pd.DataFrame(
        columns=['Node', '$/MWhr', '$/MWhr_NoOut', '$/MWhr/CVAR98', 'CVAR98_Unit', 'CVAR02_Unit', 'VAR98',
                 'HR%', 'Inc_HR%','Dec_HR%', 'MWs', 'Profit', 'Inc_Profit', 'Dec_Profit',
                 'Max_Gain', 'Max_Loss', '%Trd'])

    for col in PnL_df.columns:
        mw = mw_df[col].sum()

        inc_profit = inc_PnL_df[col].sum()
        dec_profit = dec_PnL_df[col].sum()
        profit = PnL_df[col].sum()

        inc_trades = (inc_mw_df[col]>0).sum()
        dec_trades = (dec_mw_df[col]>0).sum()
        trades = (mw_df[col]>0).sum()

        inc_corr_trades = (inc_PnL_df[col]>0).sum()
        dec_corr_trades = (dec_PnL_df[col]>0).sum()
        corr_trades = (PnL_df[col]>0).sum()

        hours = len(PnL_df)

        max_loss = PnL_df[col].min()
        max_gain = PnL_df[col].max()

        var_98 = PnL_df[col].quantile(0.02)
        cvar_98 =      PnL_df[PnL_df[col] < var_98][col].sum()
        cvar_98_mw =   mw_df[PnL_df[col] < var_98][col].sum()

        var_02 = PnL_df[col].quantile(0.98)
        cvar_02 =      PnL_df[PnL_df[col] > var_02][col].sum()
        cvar_02_mw =   mw_df[PnL_df[col] > var_02][col].sum()

        profit_no_02_98 = profit-cvar_98-cvar_02
        mw_no_02_98 = mw - cvar_98_mw - cvar_02_mw

        dict = {'Node': col,
                '$/MWhr': round(profit / mw, 2),
                '$/MWhr_NoOut': round(profit_no_02_98 / mw_no_02_98, 2),
                '$/MWhr/CVAR98' : -round((profit/mw) / (cvar_98 / cvar_98_mw),3),
                'VAR98': round(var_98,0),
                'CVAR98_Unit': round(cvar_98 / cvar_98_mw , 2),
                'CVAR02_Unit': round(cvar_02 / cvar_02_mw, 2),
                'HR%': round(corr_trades / trades, 3),
                'Inc_HR%': round(inc_corr_trades / inc_trades, 3),
                'Dec_HR%': round(dec_corr_trades / dec_trades, 3),
                'MWs': round(mw,0),
                'Profit': round(profit,0),
                'Inc_Profit': round(inc_profit,0),
                'Dec_Profit': round(dec_profit,0),
                'Max_Gain': round(max_gain,0),
                'Max_Loss': round(max_loss,0),
                '%Trd': round(trades / hours, 2)}

        summary_df = summary_df.append(dict, ignore_index=True)

    return summary_df

def do_dart_PnL(backtest_filename, save, sd_band, inc_mean_band_peak, dec_mean_band_peak, inc_mean_band_offpeak,tier2_PnL_cutoff, dec_mean_band_offpeak, tier2_filter,tier2_backtest, scale_mean_div_sd, start_date, end_date,cutoff_dolMW,cutoff_max_hourly_loss,dart_sd_location_filter,top_hourly_locs, max_trade_mws, min_trade_mws, target_mws,save_name, max_hourly_inc_mws, max_hourly_dec_mws,tier2_sd_filter,working_directory,static_directory,limit_daily_mws,limit_hourly_mws,locations=None):
    save_directory = working_directory + '\PnLFiles\\'
    hourly_PnL_dict = calc_hourly_pnl(backtest_filename=backtest_filename,
                                      sd_band=sd_band,
                                      inc_mean_band_peak=inc_mean_band_peak,
                                      dec_mean_band_peak=dec_mean_band_peak,
                                      inc_mean_band_offpeak=inc_mean_band_offpeak,
                                      dec_mean_band_offpeak=dec_mean_band_offpeak,
                                      tier2_PnL_cutoff=tier2_PnL_cutoff,
                                      scale_mean_div_sd=scale_mean_div_sd,
                                      tier2_filter=tier2_filter,
                                      tier2_backtest = tier2_backtest,
                                      start_date=start_date,
                                      end_date=end_date,
                                      top_hourly_locs=top_hourly_locs,
                                      max_trade_mws=max_trade_mws,
                                      min_trade_mws=min_trade_mws,
                                      max_hourly_dec_mws=max_hourly_dec_mws,
                                      max_hourly_inc_mws=max_hourly_inc_mws,
                                      limit_daily_mws=limit_daily_mws,
                                      limit_hourly_mws=limit_hourly_mws,
                                      target_mws = target_mws,
                                      dart_sd_location_filter=dart_sd_location_filter,
                                      tier2_sd_filter=tier2_sd_filter,
                                      working_directory=working_directory,
                                      static_directory=static_directory)

    daily_PnL_dict = dict()
    monthly_PnL_dict = dict()

    # Remove Locations That Do Not Meet The Minimum $/MWhr Cutoff
    if locations is None:
        summary_dolMW_df = pd.DataFrame(columns=['Node','$/MWhr','MaxLoss'])
        for node in hourly_PnL_dict['PnL_df'].columns:
            profit = hourly_PnL_dict['PnL_df'][node].sum()
            max_loss = hourly_PnL_dict['PnL_df'][node].min()
            mw = hourly_PnL_dict['mw_df'][node].sum()
            summary_dolMW_df = summary_dolMW_df.append({'Node':node,'$/MWhr':profit/mw,'MaxLoss':max_loss},ignore_index=True)
            locations = summary_dolMW_df[(summary_dolMW_df['$/MWhr'] > cutoff_dolMW) &(summary_dolMW_df['MaxLoss'] > -cutoff_max_hourly_loss)]['Node'].values

        print('Locations Removed In First Pass Due To Not Meeting $' + str(cutoff_dolMW) + '/MWhr Threshold:')
        print(set(hourly_PnL_dict['PnL_df'].columns) - set(locations))
        print('')

        for name, df in hourly_PnL_dict.items():
            hourly_PnL_dict[name] = df[locations]


    print('Calculating Totals...')

    for name, df in hourly_PnL_dict.items():
        hourly_total = df.sum(axis=1)
        df.insert(0, 'Total$Total_'+dart_sd_location_filter, hourly_total)
        hourly_PnL_dict.update({name: df})

        daily_df = df.groupby(['Date']).sum()
        daily_PnL_dict.update({name:daily_df})

        monthly_df = df.groupby([df.index.get_level_values('Date').year ,df.index.get_level_values('Date').month]).sum()
        monthly_df.index.names = ['Year','Month']
        monthly_PnL_dict.update({name:monthly_df})

    print('Calculating Summaries...' + str(len(locations)) + ' locations')
    hourly_summary_PnL = calc_summary_pnl(input_dict=hourly_PnL_dict)

    daily_summary_PnL = calc_summary_pnl(input_dict=daily_PnL_dict)


    monthly_PnL_df = monthly_PnL_dict['PnL_df']
    monthly_mw_df = monthly_PnL_dict['mw_df']
    monthly_dol_mw_df = monthly_PnL_df/monthly_mw_df
    daily_PnL_df = daily_PnL_dict['PnL_df']
    daily_mw_df = daily_PnL_dict['mw_df']
    daily_PnL_df.insert(1, 'RollingTotal', daily_PnL_df['Total$Total_'+dart_sd_location_filter].cumsum())
    daily_PnL_df.index = daily_PnL_df.index.strftime('%m/%d/%Y')
    daily_PnL_df.index.name = 'Date'
    hourly_PnL_df = hourly_PnL_dict['PnL_df']

    if save:
        print('Writing File...')
        writer = pd.ExcelWriter(save_directory+'PnL_Results_'+save_name+'.xlsx', engine='openpyxl')
        hourly_summary_PnL.to_excel(writer, sheet_name='Hourly_Summary', index=False)
        daily_summary_PnL.to_excel(writer, sheet_name='Daily_Summary', index=False)
        monthly_dol_mw_df.round(2).to_excel(writer, sheet_name='Monthly_$_MWHr')
        monthly_PnL_df.round(0).to_excel(writer, sheet_name='Monthly_PnL')
        daily_PnL_df.round(2).to_excel(writer, sheet_name='Daily_PnL')
        daily_mw_df.round(2).to_excel(writer, sheet_name='Daily_MW')
        hourly_PnL_df.round(2).to_excel(writer, sheet_name='Hourly_PnL')
        writer.save()
        writer.close()

    return hourly_summary_PnL,daily_summary_PnL,monthly_dol_mw_df,monthly_PnL_df,daily_PnL_df

def do_DART_backtest_compare(compare_old_filename,compare_new_filename,compare_filename, sd_band, inc_mean_band, dec_mean_band, scale_mean_div_sd,cutoff_dolMW, dart_sd_location_filter=dart_sd_location_filter):
    save_directory = os.getcwd() + '\PnLFiles\\'
    old_input_df = pd.read_csv(compare_old_filename + '.csv', parse_dates=True)
    new_input_df = pd.read_csv(compare_new_filename + '.csv', parse_dates=True)

    ### Make sure date ranges are the same
    start_old = old_input_df['Date'][0]
    start_new = new_input_df['Date'][0]
    end_old = old_input_df['Date'][len(old_input_df)-1]
    end_new = new_input_df['Date'][len(new_input_df)-1]
    start_date = max(start_old, start_new)
    end_date = min(end_old, end_new)

    old_hourly_summary_PnL,old_daily_summary_PnL,old_monthly_dol_mw_df,old_monthly_PnL_df,old_daily_PnL_df = do_dart_PnL(backtest_filename=compare_old_filename,
                                                                                                                         save=False,
                                                                                                                         dart_sd_location_filter=dart_sd_location_filter,
                                                                                                                         sd_band=dart_sd_band,
                                                                                                                         inc_mean_band=inc_mean_band,
                                                                                                                         dec_mean_band=dec_mean_band,
                                                                                                                         scale_mean_div_sd=scale_mean_div_sd,
                                                                                                                         start_date=start_date,
                                                                                                                         end_date=end_date,
                                                                                                                         cutoff_dolMW=cutoff_dolMW)

    new_hourly_summary_PnL,new_daily_summary_PnL,new_monthly_dol_mw_df,new_monthly_PnL_df,new_daily_PnL_df = do_dart_PnL(backtest_filename=compare_new_filename,
                                                                                                                         save=False,
                                                                                                                         dart_sd_location_filter=dart_sd_location_filter,
                                                                                                                         sd_band=sd_band,
                                                                                                                         inc_mean_band=inc_mean_band,
                                                                                                                         dec_mean_band=dec_mean_band,
                                                                                                                         scale_mean_div_sd=scale_mean_div_sd,
                                                                                                                         start_date=start_date,
                                                                                                                         end_date=end_date,
                                                                                                                         cutoff_dolMW=cutoff_dolMW)

    hourly_summary_delta = pd.concat([new_hourly_summary_PnL.iloc[:, :1], old_hourly_summary_PnL.iloc[:, 1:] - new_hourly_summary_PnL.iloc[:, 1:]],axis=1)

    hourly_summary_only_total = pd.concat([old_hourly_summary_PnL.iloc[:1, :], new_hourly_summary_PnL.iloc[:1, :],
                                           hourly_summary_delta.iloc[:1, :]], axis=0)
    hourly_summary_only_total['Node'] = ['New', 'Old', 'Delta']

    print('Writing File...')
    writer = pd.ExcelWriter(save_directory+'PnL_Compare_' + compare_filename + '.xlsx', engine='openpyxl')
    hourly_summary_only_total.to_excel(writer, sheet_name='Hourly_Summary_Delta', index=False)
    old_hourly_summary_PnL.to_excel(writer, sheet_name='Hourly_Summary_Old', index=False)
    new_hourly_summary_PnL.to_excel(writer, sheet_name='Hourly_Summary_New', index=False)

    monthly_dol_mw_summary = pd.DataFrame(index=new_monthly_PnL_df.index)
    monthly_dol_mw_summary['New'] = new_monthly_dol_mw_df['Total$Total_'+dart_sd_location_filter]
    monthly_dol_mw_summary['Old'] = old_monthly_dol_mw_df['Total$Total_'+dart_sd_location_filter]
    monthly_dol_mw_summary['Delta'] = monthly_dol_mw_summary['New']-monthly_dol_mw_summary['Old']
    monthly_dol_mw_summary.round(2).to_excel(writer, sheet_name='Monthly_$_MWHr_Delta')

    daily_PnL_summary = pd.DataFrame(index=new_daily_PnL_df.index)
    daily_PnL_summary['New'] = new_daily_PnL_df['Total$Total_'+dart_sd_location_filter]
    daily_PnL_summary['Old'] = old_daily_PnL_df['Total$Total_'+dart_sd_location_filter]
    daily_PnL_summary['Delta'] = daily_PnL_summary['New']-daily_PnL_summary['Old']
    daily_PnL_summary['RollingNew'] = new_daily_PnL_df['RollingTotal']
    daily_PnL_summary['RollingOld'] = old_daily_PnL_df['RollingTotal']
    daily_PnL_summary['RollingDelta'] = daily_PnL_summary['RollingNew'] - daily_PnL_summary['RollingOld']
    daily_PnL_summary.round(2).to_excel(writer, sheet_name='Daily_PnL_Delta')

    writer.save()

def find_offer_prices(backtest_filename, lmp_df, sd_band, inc_mean_band_peak, dec_mean_band_peak, inc_mean_band_offpeak, dec_mean_band_offpeak, start_date, end_date, dart_sd_location_filter, working_directory, save_name,cutoff_dolMW,top_hourly_locs, max_trade_mws, min_trade_mws, target_mws, max_hourly_inc_mws, max_hourly_dec_mws,limit_daily_mws,limit_hourly_mws):
    lmp_df = lmp_df.apply(pd.to_numeric, errors='ignore')
    lmp_df.fillna(0,inplace=True)

    backtest_directory = working_directory + '\BacktestFiles\\'
    bids_offers_directory = working_directory + '\BidsOffers\\'
    # CALCULATES HOURLY PnL AND OUTPUTS DICT WITH DATAFRAMES FOR TOT/INC/DEC PnL and MWs
    input_df=pd.read_csv(backtest_directory+dart_backtest_filename+'.csv',index_col=['Date','HE'],parse_dates=True)
    lmp_df.index.names = ['Date','HE']

    input_df.fillna(axis=0,value=0,inplace=True)
    input_df.dropna(axis=0, inplace=True)
    input_df = input_df[(input_df.index.get_level_values('Date')>=start_date) & (input_df.index.get_level_values('Date')<=end_date)]

    pred_df = input_df[[col for col in input_df.columns if 'pred' in col]].copy()
    sd_df = input_df[[col for col in input_df.columns if 'sd' in col]].copy()
    act_df = input_df[[col for col in input_df.columns if 'act' in col]].copy()

    pred_df.columns = [col.replace('_pred','') for col in pred_df.columns]
    sd_df.columns = [col.replace('_sd', '') for col in sd_df.columns]
    act_df.columns = [col.replace('_act', '') for col in act_df.columns]

    pred_df = pred_df[[col for col in pred_df.columns if col[-3:]==dart_sd_location_filter[-3:]]]
    sd_df = sd_df[[col for col in sd_df.columns if col[-3:]==dart_sd_location_filter[-3:]]]
    act_df = act_df[[col for col in act_df.columns if col[-3:]==dart_sd_location_filter[-3:]]]

    # Apply Filters
    for col in pred_df.columns:
        # SD Filter
        pred_df.loc[abs(pred_df[col]) < (sd_df[col] * sd_band), col] = 0

        # OnPeak and OffPeak median bands
        for hour in pred_df.index.get_level_values('HE').unique():
            if hour in [1,2,3,4,5,6,23,24]:
                pred_df.loc[(pred_df[col]>0) & (pred_df.index.get_level_values('HE') ==hour) & (pred_df[col]< inc_mean_band_offpeak), col] = 0
                pred_df.loc[(pred_df[col]<0) & (pred_df.index.get_level_values('HE') ==hour) & (pred_df[col]>-dec_mean_band_offpeak), col] = 0
            else:
                pred_df.loc[(pred_df[col] > 0) & (pred_df.index.get_level_values('HE')==hour) & (pred_df[col] < inc_mean_band_peak), col] = 0
                pred_df.loc[(pred_df[col] < 0) & (pred_df.index.get_level_values('HE') ==hour) & (pred_df[col] > -dec_mean_band_peak), col] = 0

    # Only take top preds from the row (scaled by median/sd)
    pred_df = pred_df.mask(abs(pred_df).rank(axis=1, method='min', ascending=False) > top_hourly_locs, 0)

    # MW Volumne
    mw_df = round(abs(pred_df/pred_df),1)

    if limit_daily_mws:
        # Scale daily MWs to meet hourly caps and daily caps
        scaled_mw_df = pd.DataFrame()
        orig_target_mws = target_mws
        for day in mw_df.index.get_level_values('Date').unique():
            day_mw_df = mw_df.loc[mw_df.index.get_level_values('Date')==day]
            day_pred_df = pred_df.loc[pred_df.index.get_level_values('Date')==day]
            print(day)
            target_mws = orig_target_mws

            counter = 0
            hour_counter = 25

            while (target_mws > 0) and (hour_counter > 1):
                mws = day_mw_df.sum().sum()
                scaling_factor = min(target_mws / mws, max_trade_mws)
                day_mw_df = day_mw_df * scaling_factor

                for location in day_mw_df.columns:
                    day_mw_df.loc[(day_mw_df[location] > max_trade_mws), location] = max_trade_mws

                day_mw_df['INC_Hourly_Total_MW'] = day_mw_df[day_pred_df > 0].sum(axis=1)
                day_mw_df['DEC_Hourly_Total_MW'] = day_mw_df[day_pred_df < 0].sum(axis=1)

                hour_counter = 1
                for hour in day_mw_df.index.get_level_values('HE'):
                    hourly_mw_df = day_mw_df[day_mw_df.index.get_level_values('HE') == hour]
                    hourly_pred_df = day_pred_df[day_pred_df.index.get_level_values('HE') == hour]
                    inc_hourly_total = hourly_mw_df['INC_Hourly_Total_MW'][0]
                    dec_hourly_total = hourly_mw_df['DEC_Hourly_Total_MW'][0]

                    if (inc_hourly_total >= max_hourly_inc_mws) or (dec_hourly_total >= max_hourly_dec_mws):
                        if inc_hourly_total == 0: inc_hourly_total = max_hourly_inc_mws
                        if dec_hourly_total == 0: dec_hourly_total = max_hourly_dec_mws

                        inc_ratio = max_hourly_inc_mws / inc_hourly_total
                        dec_ratio = max_hourly_dec_mws / dec_hourly_total
                        smallest_ratio = min(inc_ratio, dec_ratio)

                        # Preserve the ratio of INC to DEC trades within the hour but ensure neither breech their respective hourly caps
                        hourly_mw_df = hourly_mw_df * smallest_ratio

                        # Ensure no trades are below the minimum trade size
                        for trade in hourly_mw_df.columns:
                            # if (hourly_mw_df[trade][0] < min_trade_mws / 2) and (hourly_mw_df[trade][0] > 0):hourly_mw_df[trade][0] = 0
                            if (hourly_mw_df[trade][0] < min_trade_mws) and (hourly_mw_df[trade][0] > 0):hourly_mw_df[trade][0] = min_trade_mws
                            if (hourly_mw_df[trade][0] > max_trade_mws) and (hourly_mw_df[trade][0] > 0):hourly_mw_df[trade][0] = max_trade_mws


                        hourly_mws = hourly_mw_df.sum().sum() - hourly_mw_df['INC_Hourly_Total_MW'].sum() - hourly_mw_df['DEC_Hourly_Total_MW'].sum()
                        # Reduce target MWs by the number of MWs in the 'full' hour and add the full hour to the final trades df
                        target_mws = target_mws - hourly_mws
                        scaled_mw_df = pd.concat([scaled_mw_df, hourly_mw_df], sort=True)

                        # Drop the 'full' hour from the mw matrix
                        day_pred_df.drop(index=hourly_pred_df.index, inplace=True)
                        day_mw_df.drop(index=hourly_mw_df.index, inplace=True)
                        mw_df.drop(index=hourly_mw_df.index, inplace=True)
                        hour_counter += 1

                    # Ensure no trades are below the minimum trade size
                    for trade in hourly_mw_df.columns:
                        # if (hourly_mw_df[trade][0] < min_trade_mws / 2) and (hourly_mw_df[trade][0] > 0):hourly_mw_df[trade][0] = 0
                        if (hourly_mw_df[trade][0] < min_trade_mws) and (hourly_mw_df[trade][0] > 0): hourly_mw_df[trade][0] = min_trade_mws
                        if (hourly_mw_df[trade][0] > max_trade_mws) and (hourly_mw_df[trade][0] > 0): hourly_mw_df[trade][0] = max_trade_mws

                counter += 1

            scaled_mw_df = pd.concat([day_mw_df, scaled_mw_df], sort=True)
            scaled_mw_df.sort_values(['Date','HE'],inplace=True, ascending=True)
            scaled_mw_df['INC_Hourly_Total_MW'] = scaled_mw_df[pred_df > 0].sum(axis=1)
            scaled_mw_df['DEC_Hourly_Total_MW'] = scaled_mw_df[pred_df < 0].sum(axis=1)
        mw_df = scaled_mw_df

    elif limit_hourly_mws:
        #####  LIMIT MAX HOURLY MWs
        for index, row in mw_df.iterrows():
            row_sum = row.sum()
            if row_sum>max_hourly_inc_mws:
                mw_df.loc[index] = max_hourly_inc_mws/(mw_df.loc[index]*row_sum)



    # PnL Calc
    PnL_df = (np.sign(pred_df)*np.sign(act_df))*abs(act_df)*mw_df

    inc_mw_df=mw_df[pred_df>0].fillna(0)
    dec_mw_df=mw_df[pred_df<0].fillna(0)
    inc_PnL_df=PnL_df[pred_df>0].fillna(0)
    dec_PnL_df=PnL_df[pred_df<0].fillna(0)

    output_dict = {'mw_df':mw_df,
                   'PnL_df':PnL_df,
                   'inc_mw_df': inc_mw_df,
                   'inc_PnL_df': inc_PnL_df,
                   'dec_mw_df': dec_mw_df,
                   'dec_PnL_df': dec_PnL_df}

    print('Hourly PnL Complete: '+backtest_filename)
    print('')

    bid_offer_df = pd.DataFrame(columns=['Node','TradeType', 'Price'])


    inc_PnL_df.columns = [col.split('_SD')[0] for col in inc_PnL_df]
    dec_PnL_df.columns = [col.split('_SD')[0] for col in dec_PnL_df]

    all_inc_nodes_df = pd.DataFrame(columns=['PnL','DALMP'], index=['Date','HE'])
    all_dec_nodes_df = pd.DataFrame(columns=['PnL', 'DALMP'], index=['Date', 'HE'])

    for location in inc_PnL_df.columns:
        try:
            temp_inc_PnL_df = inc_PnL_df[[col for col in inc_PnL_df if location in col]]
            temp_inc_dalmp_df = lmp_df[[col for col in lmp_df if location.replace('DART','DALMP') in col]]
            temp_inc_PnL_df.columns = [col.replace('DART','PnL') for col in temp_inc_PnL_df]
            temp_df = pd.merge(temp_inc_PnL_df, temp_inc_dalmp_df, on=['Date','HE'], how='inner')
            temp_df=temp_df.replace(0,np.nan)
            temp_df.dropna(inplace=True, axis=0)
            temp_df.columns = ['PnL','DALMP']
            all_inc_nodes_df = pd.concat([all_inc_nodes_df, temp_df], axis=0)
            temp_df.sort_values(by=['DALMP'],ascending=True, inplace=True)
            temp_df.reset_index(inplace=True, drop=True)
            temp_df['RunningPnL'] = temp_df['PnL'].cumsum()
            min_PnL = temp_df['RunningPnL'].min()
            min_sample_df = temp_df[temp_df['RunningPnL']==min_PnL]
            inc_offer = min_sample_df['DALMP'].values[0]
            bid_offer_df= bid_offer_df.append({'Node': location, 'TradeType': 'INC', 'Price': inc_offer}, ignore_index=True)
        except:
            print('failed location:' + location)
            print(temp_df)

    # Run for all nodes
    all_inc_nodes_df.sort_values(by=['DALMP'], ascending=True, inplace=True)
    all_inc_nodes_df.reset_index(inplace=True, drop=True)
    all_inc_nodes_df['RunningPnL'] = all_inc_nodes_df['PnL'].cumsum()
    all_inc_nodes_df.to_csv(bids_offers_directory+save_name+'_INC_OFFERS.csv')
    min_PnL = all_inc_nodes_df['RunningPnL'].min()
    min_sample_df = all_inc_nodes_df[all_inc_nodes_df['RunningPnL'] == min_PnL]
    inc_offer = min_sample_df['DALMP'].values[0]
    bid_offer_df = bid_offer_df.append({'Node': 'ALL', 'TradeType': 'INC', 'Price': inc_offer}, ignore_index=True)

    for location in dec_PnL_df.columns:
        try:
            temp_dec_PnL_df = dec_PnL_df[[col for col in dec_PnL_df if location in col]]
            temp_dec_dalmp_df = lmp_df[[col for col in lmp_df if location.replace('DART','DALMP') in col]]
            temp_dec_PnL_df.columns = [col.replace('DART','PnL') for col in temp_dec_PnL_df]
            temp_df = pd.merge(temp_dec_PnL_df, temp_dec_dalmp_df, on=['Date','HE'], how='inner')
            temp_df=temp_df.replace(0,np.nan)
            temp_df.dropna(inplace=True, axis=0)
            temp_df.columns = ['PnL','DALMP']
            all_dec_nodes_df = pd.concat([all_dec_nodes_df, temp_df], axis=0)
            temp_df.sort_values(by=['DALMP'],ascending=False, inplace=True)
            temp_df.reset_index(inplace=True, drop=True)
            temp_df['RunningPnL'] = temp_df['PnL'].cumsum()
            min_PnL = temp_df['RunningPnL'].min()
            min_sample_df = temp_df[temp_df['RunningPnL']==min_PnL]
            dec_offer = min_sample_df['DALMP'].values[0]
            bid_offer_df= bid_offer_df.append({'Node': location, 'TradeType': 'DEC', 'Price': dec_offer}, ignore_index=True)
        except:
            print('failed location:' + location)
            print(temp_df)

    all_dec_nodes_df.sort_values(by=['DALMP'], ascending=False, inplace=True)
    all_dec_nodes_df.reset_index(inplace=True, drop=True)
    all_dec_nodes_df['RunningPnL'] = all_dec_nodes_df['PnL'].cumsum()
    all_dec_nodes_df.to_csv(bids_offers_directory+save_name+'_DEC_BIDS.csv')
    min_PnL = all_dec_nodes_df['RunningPnL'].min()
    min_sample_df = all_dec_nodes_df[all_dec_nodes_df['RunningPnL'] == min_PnL]
    dec_offer = min_sample_df['DALMP'].values[0]
    bid_offer_df = bid_offer_df.append({'Node': 'ALL', 'TradeType': 'DEC', 'Price': dec_offer}, ignore_index=True)

    bid_offer_df['Price'] = bid_offer_df['Price'].round(2)


    bid_offer_df = bid_offer_df.pivot(index='Node',columns='TradeType', values='Price')
    bid_offer_df.to_csv(bids_offers_directory+save_name+'_NODE_PRICES.csv')

    return output_dict


if run_DART_PnL:
    if tier2_filter==True:
        tier2_name_adder='_tier2'
    else:
        tier2_name_adder = ''
    save_name = dart_backtest_filename+'_'+dart_sd_location_filter+str(dart_inc_mean_band_peak)+'_'+name_adder +'_'+tier2_name_adder
    print('Running: '+ save_name)
    do_dart_PnL(backtest_filename=dart_backtest_filename,
                save=True,
                sd_band=dart_sd_band,
                inc_mean_band_peak=dart_inc_mean_band_peak,
                dec_mean_band_peak=dart_dec_mean_band_peak,
                inc_mean_band_offpeak=dart_inc_mean_band_offpeak,
                dec_mean_band_offpeak=dart_dec_mean_band_offpeak,
                scale_mean_div_sd=dart_scale_mean_div_sd,
                start_date=dart_start_date,
                end_date=dart_end_date,
                cutoff_dolMW=dart_cutoff_dolMW,
                tier2_PnL_cutoff=tier2_PnL_cutoff,
                dart_sd_location_filter=dart_sd_location_filter,
                cutoff_max_hourly_loss=cutoff_max_hourly_loss,
                top_hourly_locs=top_hourly_locs,
                max_trade_mws=max_trade_mws,
                min_trade_mws=min_trade_mws,
                max_hourly_inc_mws=max_hourly_inc_mws,
                max_hourly_dec_mws=max_hourly_dec_mws,
                tier2_filter=tier2_filter,
                tier2_backtest=tier2_backtest,
                target_mws = target_mws,
                tier2_sd_filter=tier2_sd_filter,
                save_name=save_name,
                working_directory=working_directory,
                limit_daily_mws=limit_daily_mws,
                limit_hourly_mws=limit_hourly_mws,
                static_directory=static_directory)

if run_find_offer_prices:
    save_name = dart_backtest_filename+'_'+dart_sd_location_filter+str(dart_inc_mean_band_peak)+'_OfferPrices_'+name_adder
    print('Running: '+ save_name)
    find_offer_prices(backtest_filename=dart_backtest_filename,
                      lmp_df=lmp_df,
                      sd_band=dart_sd_band,
                      inc_mean_band_peak=dart_inc_mean_band_peak,
                      dec_mean_band_peak=dart_dec_mean_band_peak,
                      inc_mean_band_offpeak=dart_inc_mean_band_offpeak,
                      dec_mean_band_offpeak=dart_dec_mean_band_offpeak,
                      start_date=dart_start_date,
                      end_date=dart_end_date,
                      dart_sd_location_filter=dart_sd_location_filter,
                      working_directory=working_directory,
                      save_name=save_name,
                      limit_daily_mws=limit_daily_mws,
                      limit_hourly_mws=limit_hourly_mws,
                      target_mws = target_mws,
                      cutoff_dolMW=cutoff_max_hourly_loss,
                      top_hourly_locs=top_hourly_locs,
                      max_trade_mws=max_trade_mws,
                      min_trade_mws=min_trade_mws,
                      max_hourly_inc_mws=max_hourly_inc_mws,
                      max_hourly_dec_mws=max_hourly_dec_mws)

if run_backtest_compare:
    compare_filename = compare_new_filename_short + '_to_' + compare_old_filename_short

    do_DART_backtest_compare(compare_old_filename=compare_old_filename,
                             compare_new_filename=compare_new_filename,
                             compare_filename=compare_filename,
                             sd_band=dart_sd_band,
                             inc_mean_band=dart_inc_mean_band,
                             dec_mean_band=dart_dec_mean_band,
                             scale_mean_div_sd=dart_scale_mean_div_sd,
                             cutoff_dolMW=dart_cutoff_dolMW)