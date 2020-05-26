import pandas as pd
import numpy as np
import datetime
import subprocess
import plotly as plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import xgboost as xgb
from API_Lib import load_obj
from API_Lib import process_YES_daily_price_tables
from API_Lib import get_spreads
from API_Lib import save_obj
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import warnings

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 5000)
pd.set_option('max_row', 100)


def create_features(input_df, iso, cat_vars, static_directory,target_name='', daily_pred=False, feat_dict=None, all_best_features_df=None):
    temp_data_directory = static_directory + '\ModelUpdateData\\Temperature_files\\'

    top_feats = pd.DataFrame()

    # Grab the target variable first if its not a daily pred (target would not exist)
    if not daily_pred:
        Y = pd.DataFrame(input_df[target_name])

    # Different data sources for PJM loads need name changes. Rename cols to correct things
    if daily_pred:
        try:
            input_df['PJM_EKPCPJMISO_FLOAD'] = input_df['PJM_EKPC_FLOAD']
            input_df['PJM_MID-ATLANTICREGION_FLOAD'] = input_df['PJM_MID_ATL_TOTAL_FLOAD']
            input_df['PJM_AECO_FLOAD'] = input_df['PJM_AE_FLOAD']
            input_df['PJM_RTOCOMBINED_FLOAD'] = input_df['PJM_RTO_TOTAL_FLOAD']
            input_df['PJM_SOUTHERNREGION_FLOAD'] = input_df['PJM_SOUTH_TOTAL_FLOAD']
            input_df['PJM_WESTERNREGION_FLOAD'] = input_df['PJM_WEST_TOTAL_FLOAD']

        except:
            pass
        input_df.columns = [col.replace('TOTAL_RESOURCE_CAP_OUT','_OUTAGE').replace('ISO_CAPACITY_OFFLINE','_OUTAGE').replace('_OUTAGES','_OUTAGE').replace(' ','') for col in input_df.columns]
        input_df.columns = [col.replace('OUTAGE','OUTAGE_RAW').replace('FLOAD', 'FLOAD_RAW').replace('FTEMP', 'FTEMP_RAW').replace('LAG', 'LAG1') for col in input_df.columns]


    ## Get top features for the target node
    if not daily_pred:
        for feat_type, score_cutoff in feat_dict.items():
            col_name= iso+'_'+feat_type+"_"+target_name
            importance_df = pd.DataFrame(all_best_features_df[col_name])
            importance_df.dropna(axis=0, inplace=True)

            importance_df['Score'] = importance_df[col_name].apply([lambda row: float(row.split(',')[0].replace('(','').replace(' ','').replace("'",''))],axis=0)
            importance_df['FeatName'] = importance_df[col_name].apply([lambda row: row.split(',')[1].replace(')', '').replace(' ','').replace("'",'')], axis=0)
            importance_df = importance_df.iloc[0:score_cutoff,:]
            top_feats= pd.concat([top_feats,importance_df], axis=0,sort=True)

        input_df.columns = [col.replace('TOTAL_RESOURCE_CAP_OUT', '_OUTAGE').replace('ISO_CAPACITY_OFFLINE', '_OUTAGE').replace('_OUTAGES', '_OUTAGE').replace(' ', '') for col in input_df.columns]
        input_df = input_df[top_feats['FeatName']].copy()
        input_df.columns = [col.replace('OUTAGE','OUTAGE_RAW').replace('LAG', 'LAG1').replace('FLOAD', 'FLOAD_RAW').replace('FTEMP', 'FTEMP_RAW') for col in input_df.columns]


    ### Sub 60 from temps
    for col in [col for col in input_df.columns if 'FTEMP' in col]:
        input_df[col] = input_df[col] - 60
        input_df.rename(columns={col:col.replace('FTEMP','FTEMP-60')},inplace=True)


    ### Add accelerations FLOAD
    temp_df = input_df[[col for col in input_df.columns if 'FLOAD_RAW' in col]]
    temp_df_p1 = temp_df.shift(1, axis=0)
    temp_df_p2 = temp_df.shift(2, axis=0)
    temp_df_a1 = temp_df.shift(-1, axis=0)
    temp_df_a2 = temp_df.shift(-2, axis=0)
    for col in temp_df.columns:
        input_df[col.replace('RAW','PACC')] = (temp_df[col]- temp_df_p1[col]) - (temp_df_p1[col] - temp_df_p2[col])
        input_df[col.replace('RAW','AACC')] = (temp_df_a2[col] - temp_df_a1[col]) - (temp_df_a1[col] - temp_df[col])

    ### Add velocities FTEMP
    # temp_df = input_df[[col for col in input_df.columns if 'FTEMP-60_RAW' in col]]
    # temp_df_p1 = temp_df.shift(1, axis=0)
    # temp_df_p2 = temp_df.shift(2, axis=0)
    # temp_df_a1 = temp_df.shift(-1, axis=0)
    # temp_df_a2 = temp_df.shift(-2, axis=0)
    # for col in temp_df.columns:
    #     input_df[col.replace('RAW','PVEL')] = temp_df[col] - temp_df_p1[col]
    #     input_df[col.replace('RAW','AVEL')] = temp_df_a1[col] - temp_df[col]


    ### Remove days without 24 hours (tails from first day and last day vel and acc
    if not daily_pred:
        daily_df = input_df.reset_index()
        daily_count_df = daily_df.groupby(['Date']).count()
        remove_dates = daily_count_df[daily_count_df['HE'] < 24].index.values
        try:
            input_df = input_df.drop(remove_dates, level='Date')
        except:
            pass

    ### Add second day dart lag
    lagged_df = input_df[[col for col in input_df if 'DART_LAG' in col]].reset_index()
    lagged_df['Date'] = lagged_df['Date'] + datetime.timedelta(days=1)
    lagged_df.set_index(['Date','HE'],inplace=True,drop=True)
    lagged_df.columns = [col.replace('LAG1','LAG2') for col in lagged_df.columns]
    input_df = input_df.join(lagged_df,how='outer').sort_values(by=['Date','HE'],ascending=True)

    ### Add day over day change in outages
    curr_outage_df = input_df[[col for col in input_df if ('OUTAGE_RAW' in col)]]
    lagged_df = curr_outage_df.reset_index()
    lagged_df['Date'] = lagged_df['Date'] + datetime.timedelta(days=1)
    lagged_df.set_index(['Date','HE'],inplace=True,drop=True)
    outage_increase_df = lagged_df - curr_outage_df
    outage_increase_df.columns = [col.replace('RAW','DoDInc') for col in outage_increase_df.columns]
    input_df = input_df.join(outage_increase_df,how='outer').sort_values(by=['Date','HE'],ascending=True)

    ### Add daily statistics to load
    daily_df = input_df.reset_index()
    daily_df = daily_df.groupby(['Date'])
    daily_avg_df = daily_df.mean()
    daily_max_df = daily_df.max()
    daily_min_df = daily_df.min()
    input_df.reset_index(inplace=True)
    input_df.set_index('Date',inplace=True)
    for col in [col for col in input_df.columns if 'FLOAD_RAW' in col]:
        input_df[col.replace('RAW','-AVGL')] = input_df[col] - daily_avg_df[col]
        input_df[col.replace('RAW', '-MAXL')] = input_df[col] - daily_max_df[col]
        input_df[col.replace('RAW', '-MINL')] = input_df[col] - daily_min_df[col]
    input_df.reset_index(inplace=True)
    input_df.set_index(['Date','HE'],inplace=True,drop=True)

    ###Add Temp Depatures
    temp_df = input_df[[col for col in input_df.columns if 'FTEMP-60_RAW' in col]]
    temp_df = temp_df + 60
    temp_df.reset_index(inplace=True)

    temp_df['Date'] = temp_df['Date'].astype('datetime64[ns]')
    temp_df['Month'] = temp_df['Date'].dt.month
    temp_df['Day'] = temp_df['Date'].dt.day

    temp_df.set_index(['Month','Day', 'HE'], inplace=True, drop=True)
    all_normals = pd.read_csv(temp_data_directory+'temp_normals_all.csv', index_col=['Month', 'Day', 'HE'])

    temp_df = temp_df.merge(all_normals, on=['Month', 'Day', 'HE'])

    for col in all_normals.columns:
        try:
            root_col = col.replace('_TNORM', '')
            temp_df[root_col + '_FTEMP_DEPART'] = temp_df[root_col + '_FTEMP-60_RAW'] - temp_df[root_col + '_TNORM']
        except:
            pass

    temp_df.reset_index(inplace=True)
    temp_df.sort_values(by=['Date', 'HE'], ascending=True, inplace=True)
    temp_df.set_index(['Date', 'HE'], inplace=True)
    temp_df = temp_df[[col for col in temp_df.columns if '_FTEMP_DEPART' in col]]
    input_df = pd.concat([input_df, temp_df], axis=1)
    input_df.replace('',np.nan,inplace=True)
    input_df.replace(' ', np.nan, inplace=True)

    # Drop NA rows if not a daily pred
    if not daily_pred:
        original_rows = len(input_df)
        input_df.dropna(inplace=True)
        new_rows = len(input_df)
        percent_less = round((original_rows-new_rows)/original_rows *100,0)
        print('NaN Rows Dropped: '+str(original_rows-new_rows)+'.  Percent Dropped: ' + str(percent_less)+'%')

    input_df.reset_index(inplace=True)
    input_df['Month'] = input_df['Date'].dt.month
    input_df['Weekday'] = input_df['Date'].dt.weekday
    # input_df['HourEnding'] = input_df['HE']
    input_df.set_index(['Date','HE'],drop=True,inplace=True)

    # Join the new features with the original target if the target exists. For daily predictions no target exists

    if daily_pred:
        output_df = input_df.sort_values(by=['Date', 'HE'], ascending=True)
    else:
        output_df = input_df.join(Y, how='inner').sort_values(by=['Date', 'HE'], ascending=True)


    output_df = round(output_df,3)

    print('Num Features Without Categoricals: ' + str(len(output_df.columns) - 1))
    # Add Categoricals
    output_df = pd.get_dummies(output_df, columns=cat_vars)


    return output_df

def create_tier2_features(backtest_df,target, daily_pred=False, hourly_PnL_df=None):
    # Takes Daily Predictions Per Target and Turns The Preds and SDs Into Features For A Daily Model

    # Take out the target variable if it is not a daily prediction
    if not daily_pred:
        # Use if target is PnL
        # y_df = pd.DataFrame(hourly_PnL_df[[col for col in hourly_PnL_df.columns if target in col]])

        # Use if target is DART
        y_df = pd.DataFrame(backtest_df[[col for col in backtest_df.columns if ((target in col) & ('_act' in col))]])

        try:
            y_df.columns = [target + '_Tier2Target']
        except:
            y_df[target + '_Tier2Target'] = np.nan # if there is no PnL for the backtest node (ie it was filtered out by not meeting min $/mwhr number) then set all hours to NaN


    x_df = backtest_df[[col for col in backtest_df.columns if (target in col)&('_act' not in col)]]
    # x_df['HourEnding'] = x_df.index.get_level_values('HE')
    # x_df['Month'] = x_df.index.get_level_values('Date').month
    # x_df['Weekday'] = x_df.index.get_level_values('Date').weekday


    # Add the target back in if it was removed to start with
    if not daily_pred:
        output_df = x_df.merge(y_df,on=['Date','HE'],how='inner')
    else:
        output_df = x_df


    return output_df

def std_dev_outlier_remove(input_df, target, sd_limit, verbose=True):
    # REMOVES OUTLIERS BASED ON THE Y TARGET COLUMN AND ITS DEVIATION FROM THE MEAN
    summary_stats = pd.DataFrame()
    input_df = input_df.replace('', np.nan)
    input_df = input_df.replace(' ', np.nan)
    original_rows = len(input_df)
    input_df.dropna(inplace=True)
    new_rows = len(input_df)
    percent_less = round((original_rows - new_rows) / original_rows * 100, 0)
    if verbose:
        print('NaN Rows Dropped: ' + str(original_rows - new_rows) + '.  Percent Dropped: ' + str(percent_less) + '%')

    # Calculate Summary Statistics
    mean = input_df[target].values.mean()
    std = input_df[target].values.std()
    count = sum(np.abs(input_df[target] - mean) >= sd_limit * std)
    summary_stats[target] = [mean, std, count]

    # Remove Samples Above Limit
    output_df = input_df[input_df.loc[:, target] <= (summary_stats.loc[0, target] + summary_stats.loc[1, target] * sd_limit)]
    output_df = output_df[output_df.loc[:, target] >= (summary_stats.loc[0, target] - summary_stats.loc[1, target] * sd_limit)]

    if verbose:
        print('Outliers removed: ' + str(len(input_df) - len(output_df)) + ', Remaining Samples: ' + str(len(output_df)) + ' (' + str(round(len(output_df) / len(input_df) * 100,0)) + '%).')

    return output_df

def read_clean_data(input_filename, input_file_type, iso, verbose=True):
    # Read In Input File
    if input_file_type.upper() == 'DICT':
        timezone_dict = {'MISO': 'EST', 'PJM': 'EPT', 'ISONE': 'EPT', 'NYISO': 'EPT', 'ERCOT': 'CPT', 'SPP':'CPT'}
        master_df = load_obj(input_filename)[timezone_dict[iso.upper()]]
    elif input_file_type.upper() == 'CSV':
        master_df = pd.read_csv(input_filename + '.csv', index_col=['Date', 'HourEnding'])
        master_df.columns = [col.replace('DART_LAG', 'DA_RT_LAG') for col in
                             master_df.columns]  # Rename lag columns in CSV file so they can be selected later
        dart_list = master_df.columns[master_df.columns.str.contains('DART')]
        master_df.rename(columns=dict(zip(dart_list, iso+'_' + dart_list)), inplace=True)
    else:
        print('Unexpected File Type')
        exit()

    # Create a Multi-Index to Keep Things Orderly. Date and HE remain as multi-index throughout code.
    master_df.reset_index(inplace=True)
    master_df['HE'] = master_df['HourEnding']
    master_df['Date'] = master_df['Date'].astype('datetime64[ns]')
    master_df.set_index(['Date', 'HE'], inplace=True, drop=True)

    # Remove Duplicated Rows and Coerce to Numeric
    initial_rows = len(master_df)
    for col in master_df.columns:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
    master_df = master_df.loc[~master_df.index.duplicated(keep='first')]

    orig_cols = set(master_df.columns)
    inactive_nodes = ['MISO_AMIL.COFFEEN1_DART','MISO_AMIL.HAVANA86_DART','MISO_AMIL.HENNEPN82_DART','MISO_DMGEN3.AGG_DART','MISO_EES.CC.WPEC_DART','MISO_EES.WRPP1_DART',
                      'MISO_AMIL.COFFEEN1_DA_RT_LAG','MISO_AMIL.HAVANA86_DA_RT_LAG','MISO_AMIL.HENNEPN82_DA_RT_LAG','MISO_DMGEN3.AGG_DA_RT_LAG','MISO_EES.CC.WPEC_DA_RT_LAG','MISO_EES.WRPP1_DA_RT_LAG'
                      ]
    master_df = master_df[[col for col in master_df if col not in inactive_nodes]]
    new_cols = set(master_df.columns)

    if verbose:
        print("Duplicated rows dropped = " + str(initial_rows - len(master_df)))
        print('While reading and cleaning data dropped inactive nodes:')
        print(orig_cols-new_cols)

    return master_df

def xgb_train(test_df, train_df, eval_df, target, sd_limit, fit_params, gpu_train, early_stopping, nrounds, verbose=True):
    # TRAINS MODEL AND PREDICTS RESULTS


    # Remove Outliers From Train Set and Eval Set
    train_df = std_dev_outlier_remove(input_df=train_df,
                                      target=target,
                                      sd_limit=sd_limit,
                                      verbose=verbose)

    eval_df = std_dev_outlier_remove(input_df=eval_df,
                                     target=target,
                                     sd_limit=sd_limit,
                                     verbose=verbose)

    # Split Test, Train, and Eval Sets Into X and Y and Create DMatrix
    x_test_df = test_df[[col for col in test_df.columns if target not in col]]
    y_test_df = test_df[[col for col in test_df.columns if target in col]]
    x_train_df = train_df[[col for col in train_df.columns if target not in col]]
    y_train_df = train_df[[col for col in train_df.columns if target in col]]
    x_eval_df = eval_df[[col for col in eval_df.columns if target not in col]]
    y_eval_df = eval_df[[col for col in eval_df.columns if target in col]]


    dtrain = xgb.DMatrix(data=x_train_df, label=y_train_df)
    deval = xgb.DMatrix(data=x_eval_df, label=y_eval_df)
    dtest = xgb.DMatrix(data=x_test_df, label=y_test_df)

    # Set Additional Model Parameters
    watchlist = [(dtrain, 'train'), (deval, 'test')]
    fit_params['objective'] = 'reg:squarederror'
    fit_params['max_bin'] = 85
    if gpu_train: fit_params['tree_method'] = 'gpu_hist'
    else: fit_params['tree_method'] = 'hist'
    try:
        fit_params['lambda'] = fit_params['reg_lambda'] #add correct param names since gridsearch was done with scikit wrapper
    except:
        pass
    try:
        fit_params['alpha'] = fit_params['reg_alpha'] #add correct param names since gridsearch was done with scikit wrapper
    except:
        pass
    try:
        fit_params['eta'] = fit_params['learning_rate'] #add correct param names since gridsearch was done with scikit wrapper
    except:
        pass

    eval_dict = dict()

    # Train Model
    gbm = xgb.train(params=fit_params,
                    dtrain=dtrain,
                    num_boost_round=nrounds,
                    evals=watchlist,
                    early_stopping_rounds=early_stopping,
                    verbose_eval=False,
                    evals_result=eval_dict)

    # Predict Model
    pred_df = pd.DataFrame(index=y_test_df.index)

    # Only Predict If A Test_df exists
    if not test_df.empty:
        pred_df[target+'_pred'] = gbm.predict(dtest)

    return pred_df, gbm

def xgb_gridsearch(train_df, target, cv_folds, iterations, sd_limit, gpu_train, nrounds):
    # Remove Outliers From Train Set and Eval Set
    train_df = std_dev_outlier_remove(input_df=train_df,
                                      target=target,
                                      sd_limit=sd_limit,
                                      verbose=True)

    x_train_df = train_df[[col for col in train_df.columns if target not in col]]
    y_train_df = train_df[[col for col in train_df.columns if target in col]]


    if gpu_train:
        tree_method = 'gpu_hist'
        n_jobs = None
    else:
        tree_method = 'hist'
        n_jobs = -6

    model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=nrounds,
                             tree_method=tree_method)

    skf = GroupKFold(n_splits=cv_folds)


    #XGBOOST TIER 1 GRID PJM ***DART***
    # param_grid = {'min_child_weight': [2],
    #               'learning_rate': [0.01],
    #               'reg_lambda': [3],
    #               'reg_alpha' : [0.1],
    #               # 'gamma': [0,1,2],  ## Gamma does not affect results with such low min child weight
    #               'subsample': [0.85,0.90,0.95],
    #               'colsample_bytree': [0.15,0.2,0.25],
    #               'max_depth': [12,14,16]}

    # # XGBOOST TIER 1 GRID MISO ***DART***
    # param_grid = {'min_child_weight': [2],
    #               'learning_rate': [0.007],
    #               'reg_lambda': [5],
    #               'reg_alpha' : [0.30], ## doesnt really change much
    #               # 'gamma': [0,1,2],  ## Gamma does not affect results with such low min child weight
    #               'subsample': [0.8,0.85,0.9],
    #               'colsample_bytree': [0.15,0.2,0.25],
    #               'max_depth': [12,14,16]}


    # # XGBOOST TIER 1 GRID ISONE ***DART***
    # param_grid = {'min_child_weight': [2],
    #               'learning_rate': [0.03],
    #               'reg_lambda': [3],
    #               'reg_alpha' : [0.10],
    #               # 'gamma': [0,1,2],  ## Gamma does not affect results with such low min child weight
    #               'subsample': [0.85,0.9,0.95],
    #               'colsample_bytree': [0.05,0.1,0.15],
    #               'max_depth': [6,8,10]}
    #

    # # XGBOOST TIER 1 GRID SPP ***DART***
    param_grid = {'min_child_weight': [2],
                  'learning_rate': [0.01],
                  'reg_lambda': [3],
                  'reg_alpha' : [0.10],
                  # 'gamma': [0,1,2],  ## Gamma does not affect results with such low min child weight
                  'subsample': [0.8,0.85,0.9],
                  'colsample_bytree': [0.05,0.1,0.15],
                  'max_depth': [13,15,17]}


    # # XGBOOST TIER 1 GRID ERCOT **SPREAD**
    # param_grid = {'min_child_weight': [2],
    #               'learning_rate': [0.0009],
    #               'reg_lambda': [3],
    #               'reg_alpha' : [0.1],
    #               # 'gamma': [0,1,2],  ## Gamma does not affect results with such low min child weight
    #               'subsample': [0.8,0.85,0.9],
    #               'colsample_bytree': [0.05,0.1,0.15],
    #               'max_depth': [6,8,10]}

    # # XGBOOST TIER 1 GRID MISO ***SPREAD***
    # param_grid = {'min_child_weight': [2],
    #               'learning_rate': [0.01],
    #               'reg_lambda': [5],
    #               'reg_alpha' : [0.30], ## doesnt really change much
    #               # 'gamma': [0,1,2],  ## Gamma does not affect results with such low min child weight
    #               'subsample': [0.85,0.9,0.95],
    #               'colsample_bytree': [0.1,0.15,0.2],
    #               'max_depth': [8,10,12]}

    # # XGBOOST TIER 1 GRID SPP ***SPREAD***
    # param_grid = {'min_child_weight': [2],
    #               'learning_rate': [0.01],
    #               'reg_lambda': [5],
    #               'reg_alpha' : [0.30], ## doesnt really change much
    #               # 'gamma': [0,1,2],  ## Gamma does not affect results with such low min child weight
    #               'subsample': [0.85,0.9,0.95],
    #               'colsample_bytree': [0.05,0.1,0.15],
    #               'max_depth': [9,11,13]}

    #XGBOOST TIER 1 GRID PJM ****SPREAD****
    # param_grid = {'min_child_weight': [2],
    #               'learning_rate': [0.01],
    #               'reg_lambda': [3],
    #               'reg_alpha' : [0.1],
    #               # 'gamma': [0,1,2],  ## Gamma does not affect results with such low min child weight
    #               'subsample': [0.85,0.9,0.95],
    #               'colsample_bytree': [0.05,0.1,0.15],
    #               'max_depth': [12,14,16]}

    # # XGBOOST TIER 1 GRID ISONE ***SPREAD***
    # param_grid = {'min_child_weight': [2],
    #               'learning_rate': [0.01],
    #               'reg_lambda': [3],
    #               'reg_alpha' : [0.10],
    #               # 'gamma': [0,1,2],  ## Gamma does not affect results with such low min child weight
    #               'subsample': [0.85,0.9,0.95],
    #               'colsample_bytree': [0.05,0.1,0.15],
    #               'max_depth': [8,10,12]}

    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=param_grid,
                                       n_iter=iterations,
                                       cv=skf.split(x_train_df, y_train_df, groups=x_train_df.index.get_level_values('Date')),
                                       verbose=3,
                                       scoring='neg_mean_squared_error',
                                       n_jobs=n_jobs)

    random_search.fit(x_train_df, y_train_df)
    results = pd.DataFrame(random_search.cv_results_)
    results = results.sort_values(by='rank_test_score', ascending=True)
    print('Best XGB Hyper Params:\n', results)

    return results

def xgb_predict(input_df, target,model_directory):
    # Runs all models for a given location and takes the median and std

    feature_match_success = True
    model_load_success = True

    # Make sure tier 1 input data has all the right features and they are in the right order
    try:
        tier1_model = load_obj(model_directory + 'ModelFile_'+target+'_cpu_1')
        if tier1_model is None:
            error = tier1_model.feature_names
    except:
        print('')
        print('ERROR: No Models Exist for target: '+model_directory + 'ModelFile_'+target+'_cpu_1')
        return pd.DataFrame(),pd.DataFrame(),True,False

    model_feature_names = tier1_model.feature_names
    missing_cols = set(model_feature_names)-set(input_df.columns)
    for feature in missing_cols:
        if ('Weekday' in feature) or ('Month' in feature) or ('HourEnding' in feature):
            pass
        else:
            print('MISSING: ' +feature+ ' IN INPUT TRAINING SET')
            feature_match_success=False
            return None, None, feature_match_success, model_load_success

    for missing_col in missing_cols:
        input_df[missing_col] = 0
    input_df = input_df[model_feature_names]


    exp_tier1_df = pd.DataFrame(index=input_df.index)
    pred_tier1_df = pd.DataFrame(index=input_df.index)
    dtest_tier1 = xgb.DMatrix(data=input_df)

    #Load and predict all Tier1 model files for this location
    exp_no=1
    for file_name in glob.glob(model_directory + 'ModelFile_'+target+'*'):
        file_name = file_name.replace('.pkl','')
        try:
            tier1_model = load_obj(file_name)
            exp_tier1_df['Exp_' + str(exp_no)] = tier1_model.predict(dtest_tier1)
        except:
            model_load_success=False
            print('No Tier 1 Model Files Found for Selected Location: '+ file_name)
            return None, None, feature_match_success, model_load_success
        exp_no+=1

    #Create summary stats for this locations tier1 preds
    pred_tier1_df[target+'_pred'] = exp_tier1_df.median(axis=1)
    pred_tier1_df[target + '_sd'] = exp_tier1_df.std(axis=1)

    pred_tier2_df = pd.DataFrame()

    # #Create, load, and predict all Tier2 model files for this location
    # pred_tier2_data = create_tier2daily_features(backtest_df=pred_tier1_df,
    #                                              target=target,
    #                                              daily_pred=True)
    #
    # pred_tier2_df = pd.DataFrame(index=pred_tier2_data.index)
    #
    # # Make sure tier2 input data has all the right features and they are in the right order
    # try:
    #     tier2_model = load_obj(model_directory + 'ModelFile_Tier2_'+target+'_PnL'+'_1')
    #
    #     model_feature_names = tier2_model.feature_names
    #     missing_cols = set(model_feature_names) - set(pred_tier2_data.columns)
    #
    #     for missing_col in missing_cols:
    #         pred_tier2_data[missing_col] = 0
    #     pred_tier2_data = pred_tier2_data[model_feature_names]
    #
    #     exp_tier2_df = pd.DataFrame(index=pred_tier2_data.index)
    #     dtest_tier2 = xgb.DMatrix(data=pred_tier2_data)
    #     exp_no = 1
    #
    #
    #     for file_name in glob.glob(model_directory + 'ModelFile_Tier2_' + target +'_PnL' + '*'):
    #         file_name = file_name.replace('.pkl', '')
    #         try:
    #             tier2_model = load_obj(file_name)
    #             exp_tier2_df['Exp_' + str(exp_no)] = tier2_model.predict(dtest_tier2)
    #         except:
    #             model_load_success=False
    #             print('No Tier 2 Model Files Found for Selected Location: '+ target)
    #             return None, None, feature_match_success, model_load_success
    #         exp_no += 1
    #
    #     # Create summary stats for this locations tier1 preds
    #     pred_tier2_df[target + '_pred'] = exp_tier2_df.median(axis=1)
    #     pred_tier2_df[target + '_sd'] = exp_tier2_df.std(axis=1)
    #
    # except:
    #     success = False
    #     print('No Tier 2 Model Files Found for Selected Location: ' + target)

    return pred_tier1_df, pred_tier2_df, feature_match_success, model_load_success

def do_xgb_prediction(predict_date_str_mm_dd_yyyy, iso, daily_trade_file_name, working_directory, static_directory, model_type):
    #Predicts all locations for a given ISO and date
    pred_save_directory = working_directory + '\PredFiles\\'
    input_file_directory = working_directory + '\InputFiles\\'
    daily_trade_files_directory = working_directory + '\DailyTradeFiles\\'
    feat_import_files_directory = working_directory + '\FeatureImportanceFiles\\'
    model_directory = static_directory + '\ModelFiles\\NewModelFiles\\'

    print('')
    print('**********************************************************************************')
    print('')
    print('Predicting ' + iso + ' For Date: ' + predict_date_str_mm_dd_yyyy + '...')
    print('')
    print('**********************************************************************************')

    try:
        date = datetime.datetime.strptime(predict_date_str_mm_dd_yyyy,'%m_%d_%Y')
    except:
        print('Input date in wrong format. Ensure format is mm_dd_yyyy')

    iso = iso.upper()

    cat_vars = ['Month','Weekday']


    input_filename = input_file_directory+predict_date_str_mm_dd_yyyy+'_RAW_DAILY_INPUT_DATA_DICT'
    input_file_type = 'dict'

    # Read input file
    input_df = read_clean_data(input_filename=input_filename,
                               input_file_type=input_file_type,
                               iso = iso,
                               verbose=False)

    preds_tier1_df  = pd.DataFrame(index=input_df.index)
    preds_tier2_df = pd.DataFrame(index=input_df.index)

    # Read daily trade variables file
    trade_variables = pd.ExcelFile(daily_trade_files_directory+daily_trade_file_name + '.xlsx')
    all_locations_variables_df = pd.read_excel(trade_variables, 'Locations')
    all_locations_variables_df = all_locations_variables_df[all_locations_variables_df['ISO']==iso]
    all_locations_variables_df = all_locations_variables_df[all_locations_variables_df['active_trading_location'] == 1]
    if model_type == 'DART':
        all_locations_variables_df = all_locations_variables_df[all_locations_variables_df['Location'].str.contains('DART')]
    elif (model_type == 'SPREAD') or (model_type == 'SYN_SPREAD'):
        all_locations_variables_df = all_locations_variables_df[all_locations_variables_df['Location'].str.contains('SPREAD')]

    all_ISOs_variables_df = pd.read_excel(trade_variables, 'ISOs')
    all_ISOs_variables_df = all_ISOs_variables_df[all_ISOs_variables_df['ISO'] == iso]
    all_ISOs_variables_df.reset_index(inplace=True, drop=True)

    if len(all_locations_variables_df)<1:
        print('No Locations Turned On For ISO: '+iso+'. Check Daily Trade Variables Sheet.')
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    all_locations_variables_df.reset_index(inplace=True,drop=True)
    feature_error_limits_df = pd.read_excel(trade_variables, 'FeatureErrorLimits')

    # Replace blanks with NA
    input_df = input_df.replace('', np.nan)

    # Drop features which are missing more than 4 hours of data in the target day
    temp_df = input_df[input_df.index.get_level_values('Date')==date]
    temp_df = temp_df.dropna(axis=1, thresh=4)
    dropped_cols = set(input_df)-set(temp_df)
    print('Dropped '+str(len(dropped_cols))+ ' cols due to NaN/Blank values > 4 in input data target day')
    print(dropped_cols)
    input_df = input_df[temp_df.columns]

    # Forward and backfill data for missing data
    input_df.fillna(method='ffill',inplace=True)
    input_df.fillna(method='bfill',inplace=True)


    input_df = input_df.apply(pd.to_numeric, errors='coerce')


    # Drop or ignore features which exceed max or min limits
    for feature_type in feature_error_limits_df['FeatureName']:
        feature_type_df = feature_error_limits_df[feature_error_limits_df['FeatureName']==feature_type]
        feature_type_df.reset_index(inplace=True, drop=True)
        min_limit = feature_type_df['MinLimit'][0]
        max_limit = feature_type_df['MaxLimit'][0]
        action = feature_type_df['Action'][0]
        for feature in input_df.columns:
            if feature_type in feature:
                # Drop, ignore, or set to max if limits are broken
                if input_df[feature].max() > max_limit:
                    if action == 'Delete':
                        print('**********************************')
                        print('          SELECT ACTION!          ')
                        print('**********************************')
                        print('Drop ' + feature + ' due to max limit breech of ' + str(round(input_df[feature].max(),2))+' (limit is set at '+str(max_limit)+')???')
                        print('If feature is dropped (Y) all locations which use this feature will not be predicted.')
                        print('If feature is not dropped (N) then the data which exceeded the historic max/min limits will be used.')
                        y_n=input('Enter Y to drop feature or N to keep feature and press ENTER.')
                        if y_n.upper() == 'Y':
                            input_df.drop(columns=[feature],inplace=True)
                            print(feature + ' feature dropped.')
                        elif y_n.upper()== 'N':
                            print(feature+ ' max error limit ignored.')
                        continue
                    elif action == 'Ignore':
                        print('Ignored ' + feature + ' max limit breech of ' + str(round(input_df[feature].max(),2))+ '. Feature used anyways. (limit is set at '+str(round(max_limit,2))+')')

                if input_df[feature].min() < min_limit:
                    if action == 'Delete':
                        print('**********************************')
                        print('          SELECT ACTION!          ')
                        print('**********************************')
                        print('Drop ' + feature + ' due to min limit breech of ' + str(round(input_df[feature].min(),2))+' (limit is set at '+str(min_limit)+')???')
                        print('If feature is dropped (Y) all locations which use this feature will not be predicted.')
                        print('If feature is not dropped (N) then the data which exceeded the historic max/min limits will be used.')
                        y_n=input('Enter Y to drop feature or N to keep feature and press ENTER.')
                        if y_n.upper() == 'Y':
                            input_df.drop(columns=[feature],inplace=True)
                            print(feature + ' feature dropped.')
                        elif y_n.upper()== 'N':
                            print(feature+ ' min error limit ignored.')
                        continue
                    elif action == 'Ignore':
                        print('Ignored ' + feature + ' min limit breech of ' + str(round(input_df[feature].min(),2))+ '. Feature used anyways. (limit is set at '+str(round(min_limit,2))+')')


    # Create all the features
    print('Creating Features...')
    location_input_df = create_features(input_df=input_df,
                                        iso=iso,
                                        cat_vars=cat_vars,
                                        static_directory=static_directory,
                                        daily_pred=True)

    # Predit only the target day and save input file
    location_input_df = location_input_df.loc[location_input_df.index.get_level_values('Date') == date]
    location_input_df.to_csv(input_file_directory+predict_date_str_mm_dd_yyyy + '_DAILY_INPUT_FILE_'+iso+'.csv')

    #Predict each location
    print('Predicting ISO: '+iso+'. Number Of Locations: ' +str(len(all_locations_variables_df)))

    loc_counter = 0
    failed_model_load_locations = []
    failed_feature_match_locations = []
    for location in all_locations_variables_df['Location']:
        print('Predicting Target: '+location + ' | ' + str(round(100*loc_counter/len(all_locations_variables_df['Location']),1)) + '% Done')
        #Remove Target Variable
        location_input_df = location_input_df[[col for col in location_input_df.columns if location not in col]]

        pred_tier1_df, pred_tier2_df, feature_match_success, model_load_success = xgb_predict(input_df=location_input_df,
                                                                                                target=location,
                                                                                                model_directory=model_directory)

        if (model_load_success==True) and (feature_match_success==True):
            preds_tier1_df = preds_tier1_df.merge(pred_tier1_df, on=['Date', 'HE'])
            if pred_tier2_df.empty == False:
                preds_tier2_df = preds_tier2_df.merge(pred_tier2_df, on=['Date'])
        elif feature_match_success==False:
            failed_feature_match_locations.append(location)
        elif model_load_success==False:
            failed_model_load_locations.append(location)

        loc_counter += 1
    failed_feature_match_df = pd.DataFrame(failed_feature_match_locations, columns=['FailedFeatureMatch'])
    failed_model_load_df = pd.DataFrame(failed_model_load_locations,columns=['FailedModelLoad'])
    failed_locations_df = pd.concat([failed_feature_match_df, failed_model_load_df], ignore_index=True, axis=1)
    failed_locations_df.columns=['FailedFeatureMatch','FailedModelLoad']


    print('')
    print('**********************************************************************************')
    print('')
    print('Prediction complete: ' + iso)
    print('')
    print('**********************************************************************************')
    print('')
    print(str(len(failed_feature_match_locations)) + ' Locations failed due to missing features in the input dataset:')
    print(failed_feature_match_locations)
    print(str(len(failed_model_load_locations)) + ' Locations failed due to model files not existing:')
    print(failed_model_load_locations)
    print('')
    print('**********************************************************************************')

    preds_tier1_df = preds_tier1_df.round(3)
    preds_tier2_df = preds_tier2_df.round(3)


    preds_tier1_df.to_csv(pred_save_directory + predict_date_str_mm_dd_yyyy + '_RAW_PREDS_TIER1_'+model_type+'_' + iso + '.csv', index=True)

    if preds_tier2_df.empty == False:
        preds_tier2_df.set_index([preds_tier1_df.index], inplace=True, drop=True)
        preds_tier2_df.to_csv(pred_save_directory + predict_date_str_mm_dd_yyyy + '_RAW_PREDS_TIER2_'+model_type+'_' + iso + '.csv', index=True)

    failed_locations_df.to_csv(pred_save_directory + predict_date_str_mm_dd_yyyy + '_FAILED_LOCATIONS_' +model_type+'_' + iso + '.csv')

    return preds_tier1_df, preds_tier2_df, failed_locations_df

def post_process_trades(iso, predict_date_str_mm_dd_yyyy, daily_trade_file_name,name_adder, working_directory, static_directory, model_type):
    print('')
    print('Post-Processing '+iso+' For Date: ' + predict_date_str_mm_dd_yyyy+'...')
    print('')
    print('**********************************************************************************')
    print('')

    pred_save_directory = working_directory + '\\PredFiles\\'
    upload_save_directory = working_directory + '\\UploadFiles\\'
    daily_trade_directory = working_directory + '\\DailyTradeFiles\\'

    preds_tier1_df = pd.read_csv(pred_save_directory + predict_date_str_mm_dd_yyyy + '_RAW_PREDS_TIER1_' +model_type+'_' + iso + '.csv', index_col=['Date','HE'],parse_dates=True)
    # preds_tier1_df = pd.read_csv(pred_save_directory + predict_date_str_mm_dd_yyyy + '_RAW_PREDS_TIER1_' + iso + '.csv',index_col=['Date', 'HE'], parse_dates=True)


    try:
        preds_tier2_df = pd.read_csv(pred_save_directory + predict_date_str_mm_dd_yyyy + '_RAW_PREDS_TIER2_'+model_type+'_' + iso + '.csv', index_col=['Date','HE'],parse_dates=True)
        sd_tier2_df = preds_tier2_df[[col for col in preds_tier2_df.columns if 'sd' in col]].copy()
        preds_tier2_df = preds_tier2_df[[col for col in preds_tier2_df.columns if 'pred' in col]].copy()
        preds_tier2_df.columns = [col.replace('_pred', '') for col in preds_tier2_df.columns]
        sd_tier2_df.columns = [col.replace('_sd', '') for col in sd_tier2_df.columns]
    except:
        print('Tier2 Preds Not Available And/Or Not Used')

    # Read in daily trade variables for locations and ISOs
    trade_variables = pd.ExcelFile(daily_trade_directory+daily_trade_file_name + '.xlsx')
    all_locations_variables_df = pd.read_excel(trade_variables, 'Locations')
    all_locations_variables_df = all_locations_variables_df[all_locations_variables_df['ISO']==iso]
    all_locations_variables_df = all_locations_variables_df[all_locations_variables_df['active_trading_location'] == 1]
    all_locations_variables_df.reset_index(inplace=True,drop=True)

    all_ISOs_variables_df = pd.read_excel(trade_variables, 'ISOs')
    all_ISOs_variables_df = all_ISOs_variables_df[all_ISOs_variables_df['ISO'] == iso]
    all_ISOs_variables_df = all_ISOs_variables_df[all_ISOs_variables_df['model_type'] == model_type]

    all_ISOs_variables_df.reset_index(inplace=True, drop=True)
    sd_tier1_df = preds_tier1_df[[col for col in preds_tier1_df.columns if 'sd' in col]].copy()
    preds_tier1_df = preds_tier1_df[[col for col in preds_tier1_df.columns if 'pred' in col]].copy()
    preds_tier1_df.columns = [col.replace('_pred','') for col in preds_tier1_df.columns]
    sd_tier1_df.columns = [col.replace('_sd', '') for col in sd_tier1_df.columns]


    # Apply tier 2 waive-off
    try:
        for location in preds_tier1_df.columns:
            location_variables_df = all_locations_variables_df[all_locations_variables_df['Location']==location].reset_index(drop=True)
            tier2_daily_PnL_cutoff = location_variables_df['tier2_daily_PnL_cutoff'][0]
            preds_tier1_df.loc[(preds_tier2_df[location] < tier2_daily_PnL_cutoff), location] = 0
    except:
        print('Tier2 Cutoffs Not Enabled')


    if len(preds_tier1_df.columns)<1:
        print('No Predictions for Any Location For ISO: '+iso)
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

    # Only take top nodes per hour on INC and DEC side each
    top_hourly_locs = all_ISOs_variables_df['max_hourly_trades'][0]

    preds_tier1_df = preds_tier1_df.mask(abs(preds_tier1_df).rank(axis=1, method='min', ascending=False) > top_hourly_locs, 0)

    inc_sd_band = all_ISOs_variables_df['INC_sd_band'][0]
    dec_sd_band = all_ISOs_variables_df['DEC_sd_band'][0]
    inc_mean_band_peak = all_ISOs_variables_df['INC_pred_band_peak'][0]
    dec_mean_band_peak = all_ISOs_variables_df['DEC_pred_band_peak'][0]
    inc_mean_band_offpeak = all_ISOs_variables_df['INC_pred_band_offpeak'][0]
    dec_mean_band_offpeak = all_ISOs_variables_df['DEC_pred_band_offpeak'][0]

    # Apply tier 1 bands
    for location in preds_tier1_df.columns:

        # Tier 1 SD bands set preds to 0
        preds_tier1_df.loc[(preds_tier1_df[location] > 0) & (abs(preds_tier1_df[location]) < (sd_tier1_df[location] * inc_sd_band)), location] = 0
        preds_tier1_df.loc[(preds_tier1_df[location] < 0) & (abs(preds_tier1_df[location]) < (sd_tier1_df[location] * dec_sd_band)), location] = 0

        for hour in preds_tier1_df.index.get_level_values('HE').unique():
            if hour in [1, 2, 3, 4, 5, 6, 23, 24]:
                preds_tier1_df.loc[(preds_tier1_df[location] > 0) & (preds_tier1_df.index.get_level_values('HE') == hour) & (preds_tier1_df[location] < inc_mean_band_offpeak), location] = 0
                preds_tier1_df.loc[(preds_tier1_df[location] < 0) & (preds_tier1_df.index.get_level_values('HE') == hour) & (preds_tier1_df[location] > -dec_mean_band_offpeak), location] = 0
            else:
                preds_tier1_df.loc[(preds_tier1_df[location] > 0) & (preds_tier1_df.index.get_level_values('HE') == hour) & (preds_tier1_df[location] < inc_mean_band_peak), location] = 0
                preds_tier1_df.loc[(preds_tier1_df[location] < 0) & (preds_tier1_df.index.get_level_values('HE') == hour) & (preds_tier1_df[location] > -dec_mean_band_peak), location] = 0

    # Set MW df to pred/pred
    mw_tier1_df = (preds_tier1_df/preds_tier1_df).fillna(0)

    # Scale to get max daily MWs. If max net hour cap reached pop off that hour to a new df and reduce the daily target by that amount. Repeat popping off and redoing until finished
    trades_df = pd.DataFrame()

    target_mws = all_ISOs_variables_df['target_daily_mws'][0]

    # if (model_type=='SPREAD') and (iso != 'ERCOT'):
    #     target_mws = target_mws/2

    max_trade_mws = all_ISOs_variables_df['max_trade_size_mws'][0]

    max_hourly_inc_mws = all_ISOs_variables_df['max_hourly_inc_mws'][0]
    max_hourly_dec_mws = all_ISOs_variables_df['max_hourly_dec_mws'][0]
    min_trade_mws = min(max_hourly_dec_mws / top_hourly_locs, max_hourly_inc_mws / top_hourly_locs)

    counter=0
    hour_counter=1
    while ((counter<24) and (target_mws>0) and (hour_counter!=25)):
        mws = mw_tier1_df.sum().sum()
        scaling_factor = min(target_mws / mws, max_trade_mws)
        mw_tier1_df = mw_tier1_df * scaling_factor

        for location in mw_tier1_df.columns:
            mw_tier1_df.loc[(mw_tier1_df[location] > max_trade_mws), location] = max_trade_mws

        mw_tier1_df['INC_Hourly_Total_MW'] = mw_tier1_df[preds_tier1_df > 0].sum(axis=1)
        mw_tier1_df['DEC_Hourly_Total_MW'] = mw_tier1_df[preds_tier1_df < 0].sum(axis=1)

        hour_counter = 1
        for hour in mw_tier1_df.index.get_level_values('HE'):
            hourly_mw_df = mw_tier1_df[mw_tier1_df.index.get_level_values('HE')==hour]
            inc_hourly_total = hourly_mw_df['INC_Hourly_Total_MW'][0]
            dec_hourly_total = hourly_mw_df['DEC_Hourly_Total_MW'][0]

            if (inc_hourly_total>=max_hourly_inc_mws) or (dec_hourly_total>=max_hourly_dec_mws):
                if inc_hourly_total == 0: inc_hourly_total=max_hourly_inc_mws
                if dec_hourly_total == 0: dec_hourly_total = max_hourly_dec_mws

                inc_ratio = max_hourly_inc_mws/inc_hourly_total
                dec_ratio = max_hourly_dec_mws/dec_hourly_total
                smallest_ratio = min(inc_ratio, dec_ratio)

                # Preserve the ratio of INC to DEC trades within the hour but ensure neither breech their respective hourly caps
                hourly_mw_df = hourly_mw_df * smallest_ratio

                # Ensure no trades are below the minimum trade size
                for trade in hourly_mw_df.columns:
                    if (hourly_mw_df[trade][0] < min_trade_mws/2) and (hourly_mw_df[trade][0] > 0): hourly_mw_df[trade][0] = 0
                    if (hourly_mw_df[trade][0] < min_trade_mws) and (hourly_mw_df[trade][0] > 0): hourly_mw_df[trade][0] = min_trade_mws
                    if (hourly_mw_df[trade][0] > max_trade_mws) and (hourly_mw_df[trade][0] > 0): hourly_mw_df[trade][0] = max_trade_mws


                hourly_mws = hourly_mw_df.sum().sum() - hourly_mw_df['INC_Hourly_Total_MW'].sum() - hourly_mw_df['DEC_Hourly_Total_MW'].sum()
                # Reduce target MWs by the number of MWs in the 'full' hour and add the full hour to the final trades df
                target_mws = target_mws-hourly_mws
                trades_df = pd.concat([trades_df,hourly_mw_df])

                # Drop the 'full' hour from the mw matrix
                mw_tier1_df.drop(index = hourly_mw_df.index,inplace=True)
                hour_counter -=1

            hour_counter+=1

        mw_tier1_df.drop(columns=['INC_Hourly_Total_MW', 'DEC_Hourly_Total_MW'], inplace=True)
        counter +=1

    trades_df = pd.concat([mw_tier1_df,trades_df], sort=True)
    trades_df['INC_Hourly_Total_MW'] = trades_df[preds_tier1_df > 0].sum(axis=1)
    trades_df['DEC_Hourly_Total_MW'] = trades_df[preds_tier1_df < 0].sum(axis=1)
    trades_df = trades_df.sort_values(['Date', 'HE']).round(1)

    # Make DECs negative MWs
    trades_df[preds_tier1_df<0] = -trades_df

    trades_df.to_csv(pred_save_directory+predict_date_str_mm_dd_yyyy + '_PREDS_FINAL_' +model_type+'_' + iso + '.csv')

    yes_df, upload_df = create_trade_file(input_mw_df=trades_df,
                                          iso=iso,
                                          all_ISOs_variables_df=all_ISOs_variables_df,
                                          working_directory=working_directory,
                                          model_type=model_type)

    yes_df.to_csv(upload_save_directory+predict_date_str_mm_dd_yyyy + '_YES_FILE_' + model_type+'_'+ name_adder + '_' + iso + '.csv', index=False)
    # upload_df.to_csv(upload_save_directory+predict_date_str_mm_dd_yyyy + '_UPLOAD_FILE_' + name_adder + '_'  + iso + '.csv', index=False)

    print('')
    print('Predictions, post-processing, and upload files completed for '+iso)
    print('')
    print('**********************************************************************************')

    return trades_df, yes_df, upload_df

def create_trade_summary(predict_date_str_mm_dd_yyyy, isos, do_printcharts, name_adder, working_directory, static_directory,model_type):
    print('')
    print('Creating Final Trade Summary For Date' +predict_date_str_mm_dd_yyyy+'...')
    print('')
    print('**********************************************************************************')
    yes_dfs_dict = {}
    failed_locations_dict = {}
    pred_save_directory = working_directory + '\\PredFiles\\'
    upload_save_directory = working_directory + '\\UploadFiles\\'

    summary_df = pd.DataFrame({'ISO': [], 'Trades_Total': [], 'MWs/Trade': [], 'MW_INC': [], 'MW_DEC': [], 'MW_Total': []})

    for iso in isos:
        try:
            yes_dfs_dict[iso] = pd.read_csv(upload_save_directory + predict_date_str_mm_dd_yyyy + '_YES_FILE_' +model_type+'_' + name_adder + '_'  + iso + '.csv')
            failed_locations_dict[iso] = pd.read_csv(pred_save_directory + predict_date_str_mm_dd_yyyy + '_FAILED_LOCATIONS_' +model_type+'_' + iso + '.csv')

            # yes_dfs_dict[iso] = pd.read_csv(upload_save_directory + predict_date_str_mm_dd_yyyy + '_YES_FILE_'  + name_adder + '_'  + iso + '.csv')
            # failed_locations_dict[iso] = pd.read_csv(pred_save_directory + predict_date_str_mm_dd_yyyy + '_FAILED_LOCATIONS_' + iso + '.csv')

        except:
            print('No Trades for '+iso+ 'for trade date '+predict_date_str_mm_dd_yyyy)
            isos.remove(iso)




    date = predict_date_str_mm_dd_yyyy

    figures_dict = {}
    for iso, df in yes_dfs_dict.items():

        failed_locations_df = failed_locations_dict[iso]
        failed_locations_df=failed_locations_df.rename(columns={'Unnamed: 0':'FailedCount'})
        if len(df)==0:
            print('No Trades for ISO: ' + iso)
            figures_dict[iso] = [go.Bar(), go.Bar(), go.Table() ,go.Table() , go.Table()]
            continue
        date = df['targetdate'][0]
        x = range(1,25,1)
        x_df = pd.DataFrame(index=x)
        x_df['MW']=0
        df = df.set_index(['Hour'])
        df_inc_master = df[df['Trade Type']=='INC']
        df_dec_master = df[df['Trade Type']=='DEC']
        df_master = df.set_index(['Trade Type'])
        df = df_master.groupby(level=[0]).sum()
        df_count = df_master.groupby(level=[0]).count()
        df = pd.DataFrame(df['MW'])
        df['Trades'] = df_count['MW']

        df_inc = df_inc_master.groupby(level=[0]).sum()
        df_dec = df_dec_master.groupby(level=[0]).sum()
        df_inc = pd.DataFrame(df_inc['MW'])
        df_dec = pd.DataFrame(df_dec['MW'])
        df_inc_count = df_inc_master.groupby(level=[0]).count()
        df_dec_count = df_dec_master.groupby(level=[0]).count()
        df_inc_count = pd.DataFrame(df_inc_count['MW'])
        df_dec_count = pd.DataFrame(df_dec_count['MW'])
        df_inc['Trades'] =df_inc_count['MW']
        df_dec['Trades'] =df_dec_count['MW']
        df_inc = df_inc.reindex(x_df.index,fill_value=0)
        df_dec = df_dec.reindex(x_df.index, fill_value=0)
        df_inc.columns = [col+'_INC' for col in df_inc.columns]
        df_dec.columns = [col + '_DEC' for col in df_dec.columns]
        df_inc_dec = pd.concat([df_inc, df_dec],axis=1)
        df_inc_dec.reset_index(inplace=True)
        df_inc_dec.rename(columns={'index':'HE'},inplace=True)
        df = df.append(df.sum().rename('<b>Total<b>'))
        df['MW/Trade'] = df['MW'] / df['Trades']
        df_inc_dec['MW_Total'] = df_inc_dec['MW_INC']+df_inc_dec['MW_DEC']
        df_inc_dec['Trades_Total'] = df_inc_dec['Trades_INC'] + df_inc_dec['Trades_DEC']
        df_inc_dec['MWs/Trade'] = df_inc_dec['MW_Total'] / df_inc_dec['Trades_Total']
        df_inc_dec.loc['<b>Total<b>'] = df_inc_dec.sum()
        df_inc_dec['MWs/Trade'] = df_inc_dec['MW_Total'] / df_inc_dec['Trades_Total']
        df_inc_dec.loc[df_inc_dec['HE'] == 300, 'HE'] = '<b>Total<b>'
        df_inc_dec = df_inc_dec[['HE','Trades_Total','MWs/Trade','MW_INC','MW_DEC','MW_Total']]
        df_inc_dec = df_inc_dec.round(1)
        df_inc_dec['MW_Total'] = df_inc_dec['MW_Total'].round(1)
        df_inc_dec['MW_INC'] = df_inc_dec['MW_INC'].round(1)
        df_inc_dec['MW_DEC'] = df_inc_dec['MW_DEC'].round(1)
        df = df.round(1)
        df['MW'] = df['MW'].round(1)
        df = df[['Trades','MW/Trade','MW']]

        df.reset_index(inplace=True)

        iso_summary_df = pd.DataFrame(df_inc_dec.drop(df_inc_dec.tail(1).index).sum()[1:]).T
        iso_summary_df['MWs/Trade'] = iso_summary_df['MW_Total'] / iso_summary_df['Trades_Total']
        iso_summary_df['ISO'] = iso
        iso_summary_df.set_index('ISO', inplace=True)

        iso_summary_df.reset_index(inplace=True)
        if summary_df.empty:
            summary_df = iso_summary_df
        else:
            summary_df = pd.concat([summary_df, iso_summary_df], axis=0, ignore_index=True)

        incbar = go.Bar(name='INC_'+iso, x=df_inc_dec['HE'], y=df_inc_dec['MW_INC'].values[:-1], marker_color = '#D32D41')

        decbar = go.Bar(name='DEC_'+iso, x=df_inc_dec['HE'], y=df_inc_dec['MW_DEC'].values[:-1], marker_color='#1F3F49')

        table1 = go.Table(
                header=dict(
                    values=df_inc_dec.columns,
                font=dict(size=14, color='white'),
                align="left",
                fill=dict(color=['#B3C100'])
                ),
                cells=dict(
                    values=[df_inc_dec[k].tolist() for k in df_inc_dec.columns[0:]],
                    align="left",
                format=[None,",d", None, ",.1f", ",.1f", ",.1f"],
                fill=dict(color=['#23282D','#CED2CC','#CED2CC','#CED2CC','#CED2CC','#23282D']),
                font=dict(color=['white', 'black', 'black', 'black', 'black', 'white']))
            )

        table2 = go.Table(
            header=dict(
                values=failed_locations_df.columns,
                font=dict(size=14, color='white'),
                align="left",
                fill=dict(color=['#4CB5F5'])
            ),
            cells=dict(
                values=[failed_locations_df[k].tolist() for k in failed_locations_df.columns[0:]],
                align="left",
                format=[None,None,None],
                fill=dict(color=['#23282D','#CED2CC','#CED2CC']),
                font=dict(color=['white', 'black', 'black']))
        )

        figures_dict[iso] = [incbar, decbar, table1  ,table2]

    if summary_df.empty==False:
        summary_df.set_index('ISO', inplace=True)
        summary_df.loc['<b>Total<b>'] = summary_df.sum()
        summary_df.reset_index(inplace=True)
        summary_df['MWs/Trade'] = summary_df['MW_Total'] / summary_df['Trades_Total']


    summary_table = go.Table(
        header=dict(
            values=summary_df.columns,
            font=dict(size=14, color='white'),
            align="left",
            fill=dict(color=['#B3C100'])
        ),
        cells=dict(
            values=[summary_df[k].tolist() for k in summary_df.columns[0:]],
            align="left",
            format=[None, ",d", ".1f", ",.1f", ",.1f", ",.1f"],
            fill=dict(color=['#23282D', '#CED2CC', '#CED2CC', '#CED2CC', '#CED2CC', '#23282D']),
            font=dict(color=['white', 'black', 'black', 'black', 'black', 'white']))
    )
    inc_sum_bar = go.Bar(name='INC', x=summary_df['ISO'], y=summary_df['MW_INC'].values,marker_color='#D32D41')

    dec_sum_bar = go.Bar(name='DEC', x=summary_df['ISO'], y=summary_df['MW_DEC'].values,marker_color='#1F3F49')


    specs1 = [[{"type": "bar"},{"type": "bar"}],
               [{"type": "table"},{"type": "table"}],
               [None,{"type": "table"}]]

    specs2 = [[{"type": "bar"},{"type": "bar"},{"type": "bar"}],
              [{"type": "table"},{"type": "table"},{"type": "table"}],
              [None,{"type": "table"},{"type": "table"}]]

    specs3 = [[{"type": "bar"},{"type": "bar"},{"type": "bar"},{"type": "bar"}],
              [{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"}],
              [None,{"type": "table"},{"type": "table"},{"type": "table"}]]

    specs4 = [[{"type": "bar"},{"type": "bar"},{"type": "bar"},{"type": "bar"},{"type": "bar"}],
              [{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"}],
              [None,{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"}]]

    specs5 = [[{"type": "bar"},{"type": "bar"},{"type": "bar"},{"type": "bar"},{"type": "bar"},{"type": "bar"}],
              [{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"}],
              [None,{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"}]]

    specs6 = [[{"type": "bar"},{"type": "bar"},{"type": "bar"},{"type": "bar"},{"type": "bar"},{"type": "bar"},{"type": "bar"}],
              [{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"}],
              [None,{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"},{"type": "table"}]]

    specs_dict = {'1':specs1, '2':specs2, '3':specs3, '4':specs4, '5':specs5, '6':specs6}


    if len(isos)==1:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[0] + ' GBM Trades ' + date+'<b>',
                   None,None,
                   '<b>'+isos[0] + ' Failed Locations<b>')
    elif len(isos)==2:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[0] + ' GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[1] + ' GBM Trades ' + date+'<b>',
                   None,None, None,
                   '<b>'+isos[0] + ' Failed Locations<b>','<b>'+ isos[1] + ' Failed Locations<b>')
    elif len(isos) == 3:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[0] + ' GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+ isos[1] + ' GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[2] + ' GBM Trades ' + date+'<b>',
                   None,None, None, None,
                   '<b>'+isos[0] + ' Failed Locations<b>', '<b>'+model_type+' '+name_adder+' '+isos[1] + ' Failed Locations<b>','<b>'+model_type+' '+name_adder+' '+ isos[2] + ' Failed Locations<b>')
    elif len(isos) == 4:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[0] + ' GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+ isos[1] + ' GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+ isos[2] + ' GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[3] + ' GBM Trades ' + date+'<b>',
                   None,None, None, None, None,
                   '<b>'+isos[0] + ' Failed Locations<b>','<b>'+model_type+' '+name_adder+' '+isos[1] + ' Failed Locations<b>','<b>'+model_type+' '+name_adder+' '+isos[2] + ' Failed Locations<b>', '<b>'+model_type+' '+name_adder+' '+isos[3] + ' Failed Locations<b>')
    elif len(isos) == 5:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[0] + ' GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[1] + ' GBM Trades ' + date+'<b>', '<b>'+model_type+' '+name_adder+' '+isos[2] + ' GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[3] + ' GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[4] + ' GBM Trades ' + date+'<b>',
                   None,None, None, None, None,None,
                  '<b>' + isos[0] + ' Failed Locations<b>','<b>'+model_type+' '+name_adder+' '+ isos[1] + ' Failed Locations<b>', '<b>'+model_type+' '+name_adder+' '+isos[2] + ' Failed Locations<b>','<b>'+model_type+' '+name_adder+' '+isos[3] + ' Failed Locations<b>','<b>'+model_type+' '+name_adder+' '+isos[4] + ' Failed Locations<b>')
    elif len(isos) == 6:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[0] + ' GBM Trades ' + date+'<b>', '<b>'+model_type+' '+name_adder+' '+isos[1] + ' GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+ isos[2] + ' GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[3] + ' GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[4] + ' GBM Trades ' + date+'<b>','<b>'+model_type+' '+name_adder+' '+isos[5] + ' GBM Trades ' + date+'<b>',
                   None,None, None, None, None,None,None,
                   '<b>'+isos[0] + ' Failed Locations<b>', '<b>'+model_type+' '+name_adder+' '+isos[1] + ' Failed Locations<b>','<b>'+model_type+' '+name_adder+' '+ isos[2] + ' Failed Locations<b>','<b>'+model_type+' '+name_adder+' '+isos[3] + ' Failed Locations<b>','<b>'+model_type+' '+name_adder+' '+isos[4] + ' Failed Locations<b>','<b>'+model_type+' '+name_adder+' '+isos[5] + ' Failed Locations<b>')


    fig = make_subplots(
        rows=3, cols=len(isos)+1,
        shared_xaxes=True,
        row_heights=[0.45, .9, 0.3],
        vertical_spacing=0.03,
        specs=specs_dict[str(len(isos))],
        subplot_titles=titles
    )

    fig.add_trace(inc_sum_bar, 1, 1)
    fig.add_trace(dec_sum_bar, 1, 1)
    fig.add_trace(summary_table, 2, 1)

    for iso_num in range(0,len(isos)):
        fig.add_trace(figures_dict[isos[iso_num]][0], 1, iso_num+2)
        fig.add_trace(figures_dict[isos[iso_num]][1], 1, iso_num+2)
        fig.add_trace(figures_dict[isos[iso_num]][2], 2, iso_num+2)
        fig.add_trace(figures_dict[isos[iso_num]][3], 3, iso_num+2)

    fig.update_layout(
        height=1250,
        width=800+800*len(isos),
        paper_bgcolor = 'white',
        plot_bgcolor = '#CED2CC',
        font = dict(color="black")
    )

    # fig.update_yaxes(title_text='MWs', tickvals=list(range(0, 170, 10)), row=1, col=1)

    for col in range(len(isos)+3):
        fig.update_xaxes(tickmode='linear', row=1, col=col)

    if do_printcharts:
        auto_open=True
    else:
        auto_open=False

    url = plotly.offline.plot(fig,filename=upload_save_directory + 'DailyTrades_' + predict_date_str_mm_dd_yyyy + '_' + model_type+'_'+ name_adder+ '.html',auto_open=auto_open)

    pass

def create_trade_file(input_mw_df, iso , all_ISOs_variables_df, working_directory, model_type):
    if iso == 'SPP': iso='SPPISO'
    if iso == 'ISONE': iso = 'NEISO'
    daily_trade_directory = working_directory + '\\DailyTradeFiles\\'

    input_mw_df.drop(columns=['INC_Hourly_Total_MW', 'DEC_Hourly_Total_MW'], inplace=True)

    trades_tall_df = input_mw_df.stack()
    trades_tall_df = pd.DataFrame({'MW':trades_tall_df[:]},index=trades_tall_df.index)
    trades_tall_df.reset_index(inplace=True)
    trades_tall_df.rename(columns={'level_2':'Node Name', 'HE':'Hour','Date':'targetdate'},inplace=True)
    trades_tall_df = trades_tall_df[trades_tall_df['MW'] != 0]
    trades_tall_df.reset_index(inplace=True,drop=True)
    trades_tall_df['iso'] = iso
    trades_tall_df['portfolioname'] = trades_tall_df['targetdate'].dt.strftime('%m/%d/%Y')+'_'+iso+'_GBM_NoTrd'
    trades_tall_df['Trade Type'] = 'INC'
    trades_tall_df.loc[trades_tall_df['MW']<0, 'Trade Type'] = 'DEC'
    trades_tall_df['Bid'] = None
    trades_tall_df['Node ID'] = None
    trades_tall_df['Bookname'] = ''
    trades_tall_df['BidSegment'] = 1

    alt_names_df = pd.read_excel(daily_trade_directory+'NodeAlternateNames.xlsx')

    if model_type=='DART':
        alt_names_df['Model_Name'] = alt_names_df['Model_Name'] + '_DART'

    alt_names_dict = dict(zip(alt_names_df['Model_Name'],alt_names_df['YES_Name']))
    alt_names_dict_spread = dict(zip(alt_names_df['NodeNameNoSpaces'], alt_names_df['Short_Model_Name_Num']))

    inc_bid = all_ISOs_variables_df['inc_offer_price'][0]
    dec_bid = all_ISOs_variables_df['dec_offer_price'][0]


    if model_type=='SPREAD':
        spread_bid = inc_bid

        trades_tall_df['Orig Sink ID'] = trades_tall_df['Node Name'].apply(lambda row: row.split('$')[0].replace('_SPREAD',''))
        trades_tall_df['Orig Source ID'] = trades_tall_df['Node Name'].apply(lambda row: row.split('$')[1].replace('_SPREAD',''))
        trades_tall_df['Bid'] = spread_bid

        for location in trades_tall_df['Orig Source ID'].unique():
            trades_tall_df.loc[(trades_tall_df['Orig Source ID'] == location) & (trades_tall_df['Trade Type']=='INC'), 'Source ID'] = alt_names_dict[location]
            trades_tall_df.loc[(trades_tall_df['Orig Source ID'] == location) & (trades_tall_df['Trade Type'] == 'DEC'), 'Sink ID'] = alt_names_dict[location]


        for location in trades_tall_df['Orig Sink ID'].unique():
            trades_tall_df.loc[(trades_tall_df['Orig Sink ID'] == location) & (trades_tall_df['Trade Type'] == 'INC') , 'Sink ID'] = alt_names_dict[location]
            trades_tall_df.loc[(trades_tall_df['Orig Sink ID'] == location) & (trades_tall_df['Trade Type'] == 'DEC'), 'Source ID'] = alt_names_dict[location]


        for location in trades_tall_df['Source ID'].unique():
            trades_tall_df.loc[(trades_tall_df['Source ID'] == location), 'Source Name'] = alt_names_dict_spread[location.replace(' ','')]

        for location in trades_tall_df['Sink ID'].unique():
            trades_tall_df.loc[(trades_tall_df['Sink ID'] == location), 'Sink Name'] = alt_names_dict_spread[location.replace(' ','')]

        try:
            trades_tall_df['Source Name'] = trades_tall_df['Source Name'].astype('int', errors='ignore')
            trades_tall_df['Sink Name'] = trades_tall_df['Sink Name'].astype('int', errors='ignore')
            trades_tall_df['Source Name'] = trades_tall_df['Source Name'].astype('str')
            trades_tall_df['Sink Name'] = trades_tall_df['Sink Name'].astype('str')
        except:
            pass


        trades_tall_df['Node Name']=trades_tall_df['Node Name'].apply(lambda row: row.replace('_SPREAD', ''))

        trades_tall_df['Node ID'] = trades_tall_df['Node Name']

    elif model_type=='DART':
        for location in trades_tall_df['Node Name'].unique():
            trades_tall_df.loc[(trades_tall_df['Node Name'] == location) & (trades_tall_df['Trade Type']=='INC'), 'Bid'] = inc_bid
            trades_tall_df.loc[(trades_tall_df['Node Name'] == location) & (trades_tall_df['Trade Type'] == 'DEC'), 'Bid'] = dec_bid
            trades_tall_df.loc[(trades_tall_df['Node Name'] == location), 'Node ID'] = alt_names_dict[location]

    trades_tall_df['Node Name'] = trades_tall_df['Node Name'].str.replace('_DART','').str.replace(iso+'_','').str.replace('ISONE_','').str.replace('SPP_','')
    trades_tall_df.sort_values(['Node ID','Hour'],inplace=True)
    trades_tall_df.reset_index(inplace=True,drop=True)
    trades_tall_df['MW'] = abs(trades_tall_df['MW'])

    #Format upload files for DART models
    if model_type=='DART':
        yes_df = trades_tall_df[['Node ID','Node Name','Trade Type','Bookname','iso','targetdate','portfolioname','Hour','MW','Bid']].copy()

        if iso == 'NEISO':
            upload_df = trades_tall_df[['targetdate', 'Node Name', 'Trade Type', 'BidSegment', 'Hour', 'MW', 'Bid', 'Node ID']].copy()
        elif iso == 'PJM':
            upload_df = trades_tall_df[['targetdate', 'Node Name', 'Trade Type', 'BidSegment', 'Hour','MW','Bid']].copy()
        else:
            upload_df = trades_tall_df[['targetdate', 'Node ID', 'Trade Type', 'BidSegment', 'Hour','MW','Bid']].copy()


    # Format upload files for SPREAD models
    elif model_type == 'SPREAD':

        trades_tall_df['BidSegment']=2

        if iso in ['ERCOT', 'PJM']:  ### Actual spread
            yes_df = trades_tall_df[['Orig Source ID','Orig Sink ID','Node Name','Node ID', 'Source ID', 'Sink ID', 'Source Name', 'Sink Name', 'Trade Type', 'Bookname', 'iso', 'targetdate', 'portfolioname', 'Hour', 'MW','Bid']].copy()
            upload_df = yes_df

        else: ### Syntehtic Spread

            sources_df = trades_tall_df[['Source ID','Source Name','Trade Type','Bookname','iso','targetdate','portfolioname','Hour','MW','Bid','BidSegment']].copy()
            sinks_df = trades_tall_df[['Sink ID', 'Sink Name', 'Trade Type', 'Bookname', 'iso', 'targetdate', 'portfolioname', 'Hour','MW', 'Bid','BidSegment']].copy()
            sources_df['Trade Type'] = 'INC'
            sinks_df['Trade Type'] = 'DEC'

            sources_df['Bid'] = inc_bid  ### Synthetic spreads must clear
            sinks_df['Bid'] = dec_bid  ### Synthetic spreads must clear

            sources_df.rename(columns={'Source ID':'Node ID', 'Source Name':'Node Name'},inplace=True)
            sinks_df.rename(columns={'Sink ID': 'Node ID', 'Sink Name': 'Node Name'}, inplace=True)
            yes_df = pd.concat([sources_df, sinks_df], axis=0)


            ### Sum duplicate trades
            yes_df = yes_df.groupby(['Node ID', 'Node Name', 'Trade Type', 'Bookname', 'iso', 'targetdate', 'portfolioname', 'Hour','Bid', 'BidSegment']).sum()
            yes_df.reset_index(inplace=True)

            ## Net out trades in same hour
            yes_df.loc[yes_df['Trade Type']=='DEC','MW'] = yes_df['MW']*-1
            yes_df.drop(columns=['Trade Type','Bid'],inplace=True)
            yes_df = yes_df.groupby(['Node ID', 'Node Name', 'Bookname', 'iso', 'targetdate', 'portfolioname', 'Hour','BidSegment']).sum()
            yes_df.reset_index(inplace=True)

            yes_df.loc[yes_df['MW'] < 0,'Bid'] = dec_bid
            yes_df.loc[yes_df['MW'] > 0, 'Bid'] = inc_bid

            yes_df.loc[yes_df['MW'] < 0,'Trade Type'] = 'DEC'
            yes_df.loc[yes_df['MW'] > 0, 'Trade Type'] = 'INC'

            ### drop trades that netted to 0
            yes_df = yes_df.drop(yes_df[yes_df['MW'] == 0].index)

            yes_df['MW'] = abs(yes_df['MW'])

            try:
                yes_df['Node Name'] = yes_df['Node Name'].astype('int', errors='ignore')
            except:
                pass

            yes_df['Node Name'] = yes_df['Node Name'].astype('str')


            if iso == 'NEISO':
                upload_df = yes_df[['targetdate', 'Node Name', 'Trade Type', 'BidSegment', 'Hour', 'MW', 'Bid', 'Node ID']].copy()
            elif iso == 'PJM':
                upload_df = yes_df[['targetdate', 'Node Name', 'Trade Type', 'BidSegment', 'Hour', 'MW', 'Bid']].copy()
            else:
                upload_df = yes_df[['targetdate', 'Node ID', 'Trade Type', 'BidSegment', 'Hour', 'MW', 'Bid']].copy()

            yes_df.drop(columns=['BidSegment'],inplace=True)



    return yes_df, upload_df

def daily_PnL(predict_date_str_mm_dd_yyyy,isos, name_adder, working_directory, static_directory, do_printcharts, backtest_pnl_filename, model_type,spread_files_name):
    print('**********************************************************************************')
    print('')
    print('Running Daily PnL For: '+predict_date_str_mm_dd_yyyy + '...')
    print('')
    print('**********************************************************************************')
    print('')

    yes_file_directory = working_directory + '\\UploadFiles\\'
    lmp_directory = working_directory+ '\\InputFiles\\'
    save_directory = working_directory + '\\DailyPnL\\'
    spread_files_directory = static_directory + '\ModelUpdateData\\'


    timezone_dict = {'MISO': 'EST', 'PJM': 'EPT', 'ISONE': 'EPT', 'NYISO': 'EPT', 'ERCOT': 'CPT', 'SPP': 'CPT'}

    predict_date = datetime.datetime.strptime(predict_date_str_mm_dd_yyyy, '%m_%d_%Y')
    predict_date = predict_date+datetime.timedelta(days=2)

    # Read LMP Data
    yes_pricetable_dict = process_YES_daily_price_tables(predict_date=predict_date,
                                                         input_timezone='CPT',
                                                         working_directory=working_directory,
                                                         dart_only=False)

    # Read Previous Days' Trades For Each ISO and Concat Into One Large DF
    trades_dict = dict()
    all_trades_df = pd.DataFrame()

    for iso in isos:
        try:
            temp_trades_df = pd.read_csv(yes_file_directory + predict_date_str_mm_dd_yyyy + '_YES_FILE_' + model_type+'_'+ name_adder + '_'  + iso + '.csv')
        except:
            print('No trades on this date for ' +iso+' (or file named wrong)')
            trades_dict[iso]=pd.DataFrame()
            continue



        try:
            temp_trades_df['Node Name'] = temp_trades_df['Node Name'].astype('int',errors='ignore')
        except:
            pass


        temp_trades_df['Node Name'] = temp_trades_df['Node Name'].astype('str')
        temp_trades_df['Node Name'] = temp_trades_df['Node Name'].str.replace(iso+'_','')
        temp_trades_df.set_index(['targetdate','Hour', 'Node Name'],inplace=True)
        temp_trades_df.index.names = ['Date','HourEnding','Node Name']

        temp_lmp_df = yes_pricetable_dict[timezone_dict[iso]]
        temp_lmp_df = temp_lmp_df[[col for col in temp_lmp_df.columns if iso in col]]

        temp_lmp_df = temp_lmp_df.stack().reset_index()
        temp_lmp_df.set_index(['Date','HourEnding'],drop=True,inplace=True)
        temp_lmp_df.columns = ['Descript','Value']
        temp_lmp_df['ISO'] = temp_lmp_df['Descript'].apply(lambda name: name.split('_')[0])
        temp_lmp_df['Node Name'] = temp_lmp_df['Descript'].apply(lambda name: '_'.join(name.split('_')[1:-1]))
        temp_lmp_df['Type'] = temp_lmp_df['Descript'].apply(lambda name: name.split('_')[-1])
        temp_lmp_df.reset_index(inplace=True)
        temp_lmp_df.set_index(['Date','HourEnding','Node Name','Type'],drop=True,inplace=True)
        temp_lmp_df.drop(columns=['ISO','Descript'],inplace=True)
        temp_lmp_df = temp_lmp_df.unstack(3)
        temp_lmp_df.columns = temp_lmp_df.columns.get_level_values(1)
        temp_lmp_df['TOT_DART'] = temp_lmp_df['DALMP'] - temp_lmp_df['RTLMP']
        temp_lmp_df.reset_index(inplace=True)

        try:
            temp_lmp_df['CONG_DART'] = temp_lmp_df['DACONG'] - temp_lmp_df['RTCONG']
            temp_lmp_df['LOSS_DART'] = temp_lmp_df['DALOSS'] - temp_lmp_df['RTLOSS']
            temp_lmp_df['ENERGY_DART'] = temp_lmp_df['DAENERGY'] - temp_lmp_df['RTENERGY']
        except:
            temp_lmp_df['CONG_DART'] = 0
            temp_lmp_df['LOSS_DART'] = 0
            temp_lmp_df['ENERGY_DART'] = temp_lmp_df['TOT_DART']


        temp_lmp_df.drop(columns=['DART'],inplace=True)
        temp_lmp_df.reset_index(inplace=True)
        temp_lmp_df['Date']=temp_lmp_df['Date'].astype('datetime64[ns]')
        temp_lmp_df['Node Name'] = temp_lmp_df['Node Name'].astype('str')


        temp_trades_df.reset_index(inplace=True)
        temp_trades_df['Date']=temp_trades_df['Date'].astype('datetime64[ns]')
        temp_trades_df['Node Name'] = temp_trades_df['Node Name'].astype('str')
        temp_trades_df.loc[temp_trades_df['iso']=='SPPISO','iso']='SPP'
        temp_trades_df.loc[temp_trades_df['iso'] == 'NEISO', 'iso'] = 'ISONE'


        if model_type == 'DART':
            temp_trades_df = pd.merge(temp_trades_df,temp_lmp_df, on=['Date','HourEnding','Node Name'])

            temp_trades_df.set_index(['Date', 'HourEnding'], drop=True, inplace=True)
            temp_trades_df['ClearedTrade'] = 1

            temp_trades_df.loc[(temp_trades_df['Trade Type'] == 'INC') & (temp_trades_df['Bid'] > temp_trades_df['DALMP']), 'ClearedTrade'] = 0
            temp_trades_df.loc[(temp_trades_df['Trade Type'] == 'DEC') & (temp_trades_df['Bid'] < temp_trades_df['DALMP']), 'ClearedTrade'] = 0

            temp_trades_df['INCDEC_MULT'] = 1

            temp_trades_df.loc[temp_trades_df['Trade Type'] == 'DEC', 'INCDEC_MULT'] = -1


            temp_trades_df['MW'] = temp_trades_df['ClearedTrade'] * temp_trades_df['MW']
            temp_trades_df['ENERGY_PnL'] = temp_trades_df['MW'] * temp_trades_df['INCDEC_MULT'] * temp_trades_df['ENERGY_DART']
            temp_trades_df['CONG_PnL'] = temp_trades_df['MW'] * temp_trades_df['INCDEC_MULT'] * temp_trades_df['CONG_DART']
            temp_trades_df['LOSS_PnL'] = temp_trades_df['MW'] * temp_trades_df['INCDEC_MULT'] * temp_trades_df['LOSS_DART']
            temp_trades_df['TOT_PnL'] = temp_trades_df['MW'] * temp_trades_df['INCDEC_MULT'] * temp_trades_df['TOT_DART']
            temp_trades_df['SUCCESS_TRADE'] = 1
            temp_trades_df.loc[temp_trades_df['TOT_PnL'] < 0, 'SUCCESS_TRADE'] = 0

            trades_dict[iso] = temp_trades_df



        if model_type == 'SPREAD':

            if iso == 'ERCOT':
                source_lmp_df = temp_lmp_df.copy()
                source_lmp_df.set_index(['Date', 'HourEnding','Node Name'], drop=True, inplace=True)
                source_lmp_df.columns = [col+'_SOURCE' for col in source_lmp_df.columns]
                source_lmp_df.reset_index(inplace=True)
                source_lmp_df.rename(columns={'Node Name': 'Source Name'},inplace=True)

                sink_lmp_df = temp_lmp_df.copy()
                sink_lmp_df.set_index(['Date', 'HourEnding', 'Node Name'], drop=True, inplace=True)
                sink_lmp_df.columns = [col + '_SINK' for col in sink_lmp_df.columns]
                sink_lmp_df.reset_index(inplace=True)
                sink_lmp_df.rename(columns={'Node Name': 'Sink Name'}, inplace=True)

                source_df = temp_trades_df[['HourEnding','Date', 'Source Name']]
                sink_df = temp_trades_df[['HourEnding', 'Date', 'Sink Name']]

                source_df = pd.merge(source_df, source_lmp_df, on=['Date', 'HourEnding', 'Source Name'])
                sink_df = pd.merge(sink_df, sink_lmp_df, on=['Date', 'HourEnding', 'Sink Name'])

                temp_trades_df = pd.merge(temp_trades_df, source_df, on=['Date','HourEnding','Source Name'])
                temp_trades_df = pd.merge(temp_trades_df, sink_df, on=['Date', 'HourEnding', 'Sink Name'])

                temp_trades_df.set_index(['Date', 'HourEnding'], drop=True, inplace=True)

                temp_trades_df['DALMP_SPREAD'] = temp_trades_df['DALMP_SINK'] - temp_trades_df['DALMP_SOURCE']
                temp_trades_df['TOT_SPREAD'] = temp_trades_df['TOT_DART_SOURCE'] - temp_trades_df['TOT_DART_SINK']
                temp_trades_df['ENERGY_SPREAD'] = temp_trades_df['ENERGY_DART_SOURCE'] - temp_trades_df['ENERGY_DART_SINK']
                temp_trades_df['CONG_SPREAD'] = temp_trades_df['CONG_DART_SOURCE'] - temp_trades_df['CONG_DART_SINK']
                temp_trades_df['LOSS_SPREAD'] = temp_trades_df['LOSS_DART_SOURCE'] - temp_trades_df['LOSS_DART_SINK']

                temp_trades_df['ClearedTrade'] = 1

                temp_trades_df.loc[temp_trades_df['DALMP_SPREAD'] > temp_trades_df['Bid'], 'ClearedTrade'] = 0

                temp_trades_df['MW'] = temp_trades_df['MW'] * temp_trades_df['ClearedTrade']

                temp_trades_df['ENERGY_PnL'] =  temp_trades_df['MW'] * temp_trades_df['ENERGY_SPREAD']
                temp_trades_df['CONG_PnL'] = temp_trades_df['MW'] * temp_trades_df['CONG_SPREAD']
                temp_trades_df['LOSS_PnL'] = temp_trades_df['MW'] * temp_trades_df['LOSS_SPREAD']
                temp_trades_df['TOT_PnL'] =  temp_trades_df['MW'] * temp_trades_df['TOT_SPREAD']
                temp_trades_df['SUCCESS_TRADE'] = 1
                temp_trades_df.loc[temp_trades_df['TOT_PnL'] < 0, 'SUCCESS_TRADE'] = 0
                temp_trades_df = temp_trades_df.drop_duplicates()

                trades_dict[iso] = temp_trades_df

            else: #SYntethic spread ISOs
                temp_trades_df['Node Name'] = temp_trades_df['Node Name'].astype('str')
                temp_lmp_df['Node Name'] = temp_lmp_df['Node Name'].astype('str')

                temp_trades_df = pd.merge(temp_trades_df, temp_lmp_df, on=['Date', 'HourEnding', 'Node Name'])

                temp_trades_df.set_index(['Date', 'HourEnding'], drop=True, inplace=True)
                temp_trades_df['ClearedTrade'] = 1

                temp_trades_df.loc[(temp_trades_df['Trade Type'] == 'INC') & (temp_trades_df['Bid'] >= temp_trades_df['DALMP']), 'ClearedTrade'] = 0
                temp_trades_df.loc[(temp_trades_df['Trade Type'] == 'DEC') & (temp_trades_df['Bid'] <= temp_trades_df['DALMP']), 'ClearedTrade'] = 0

                temp_trades_df['INCDEC_MULT'] = 1

                temp_trades_df.loc[temp_trades_df['Trade Type'] == 'DEC', 'INCDEC_MULT'] = -1

                temp_trades_df['MW'] = temp_trades_df['ClearedTrade'] * temp_trades_df['MW']
                temp_trades_df['ENERGY_PnL'] = temp_trades_df['MW'] * temp_trades_df['INCDEC_MULT'] * temp_trades_df['ENERGY_DART']
                temp_trades_df['CONG_PnL'] = temp_trades_df['MW'] * temp_trades_df['INCDEC_MULT'] * temp_trades_df['CONG_DART']
                temp_trades_df['LOSS_PnL'] = temp_trades_df['MW'] * temp_trades_df['INCDEC_MULT'] * temp_trades_df['LOSS_DART']
                temp_trades_df['TOT_PnL'] = temp_trades_df['MW'] * temp_trades_df['INCDEC_MULT'] * temp_trades_df['TOT_DART']
                temp_trades_df['SUCCESS_TRADE'] = 1
                temp_trades_df.loc[temp_trades_df['TOT_PnL'] < 0, 'SUCCESS_TRADE'] = 0

                trades_dict[iso] = temp_trades_df



    if bool(trades_dict) == False:
        print('')
        print('')
        print('**********************************************************************************')
        print('')
        print('')
        print('No Trades For Any ISO Exist for PnL Date: ' + predict_date_str_mm_dd_yyyy)
        print('')
        print('')
        print('**********************************************************************************')
        return None


    for iso, df in trades_dict.items():
        if all_trades_df.empty:
            all_trades_df=df
        else:
            all_trades_df=pd.concat([all_trades_df,df],axis=0,sort=False)

    all_trades_df.to_csv(save_directory + predict_date_str_mm_dd_yyyy + '_DAILY_PnL_'+model_type+'_' + name_adder + '.csv')

    master_trades_df = pd.read_csv(save_directory+'2020_MASTER_PnL_'+model_type+'_'+name_adder+'.csv', index_col=['Date','HourEnding'], parse_dates=True)
    master_trades_df = pd.concat([master_trades_df,all_trades_df],axis=0, sort=True)
    master_trades_df.reset_index(inplace=True)
    master_trades_df.set_index(['Date','HourEnding','Node Name'],inplace=True,drop=True)
    master_trades_df = master_trades_df.loc[~master_trades_df.index.duplicated(keep='first')]
    master_trades_df.reset_index(inplace=True)
    master_trades_df.set_index(['Date','HourEnding'],inplace=True)

    master_trades_df.to_csv(save_directory+'2020_MASTER_PnL_'+model_type+'_'+name_adder+'.csv')

    backtest_start_date = predict_date - datetime.timedelta(days=60)
    backtest_end_date = predict_date + datetime.timedelta(days=30)
    actuals_start_date = predict_date - datetime.timedelta(days=30)



    daily_actuals_dict = {}
    monthly_actuals_dict = {}
    yearly_actuals_dict = {}

    for iso in master_trades_df['iso'].unique():
        temp_master_trades_df = master_trades_df[master_trades_df['iso']==iso]
        temp_master_trades_df = temp_master_trades_df.rename(columns={'ENERGY_PnL': 'Ene_$', 'CONG_PnL': 'Con_$', 'LOSS_PnL': 'Los_$', 'TOT_PnL': 'Tot_$'})

        temp_daily_df = temp_master_trades_df.groupby(pd.Grouper(freq='D',level='Date')).sum()
        temp_monthly_df = temp_master_trades_df.groupby(pd.Grouper(freq='M',level='Date')).sum()
        temp_yearly_df = temp_master_trades_df.groupby(pd.Grouper(freq='Y',level='Date')).sum()

        temp_daily_df['$/MW'] =  temp_daily_df['Tot_$'] / temp_daily_df['MW']
        temp_monthly_df['$/MW'] = temp_monthly_df['Tot_$'] / temp_monthly_df['MW']
        temp_yearly_df['$/MW'] = temp_yearly_df['Tot_$'] / temp_yearly_df['MW']

        temp_daily_df = temp_daily_df[['Ene_$', 'Con_$', 'Los_$', 'MW', '$/MW', 'Tot_$']]
        temp_monthly_df = temp_monthly_df[['Ene_$', 'Con_$', 'Los_$', 'MW', '$/MW', 'Tot_$']]
        temp_yearly_df = temp_yearly_df[['Ene_$', 'Con_$', 'Los_$', 'MW', '$/MW', 'Tot_$']]

        temp_monthly_df.index.names = ['Month']
        temp_yearly_df.index.names = ['Year']

        temp_daily_df.reset_index(inplace=True)
        temp_monthly_df.reset_index(inplace=True)
        temp_yearly_df.reset_index(inplace=True)

        temp_daily_df = temp_daily_df.sort_values(['Date'],ascending=False)
        temp_monthly_df = temp_monthly_df.sort_values(['Month'], ascending=False)
        temp_yearly_df = temp_yearly_df.sort_values(['Year'], ascending=False)

        temp_daily_df.set_index('Date',inplace=True,drop=True)
        temp_monthly_df.set_index('Month', inplace=True, drop=True)
        temp_yearly_df.set_index('Year', inplace=True, drop=True)

        temp_daily_df.index.names = ['Date']
        temp_monthly_df.index.names = ['Month']
        temp_yearly_df.index.names = ['Year']


        temp_daily_df = temp_daily_df.round({'Ene_$': 0, 'Con_$': 0, 'Los_$': 0, 'Tot_$': 0, 'MW': 0, '$/MW': 2})
        temp_monthly_df = temp_monthly_df.round({'Ene_$': 0, 'Con_$': 0, 'Los_$': 0, 'Tot_$': 0, 'MW': 0, '$/MW': 2})
        temp_yearly_df = temp_yearly_df.round({'Ene_$': 0, 'Con_$': 0, 'Los_$': 0, 'Tot_$': 0, 'MW': 0, '$/MW': 2})

        daily_actuals_dict[iso]=temp_daily_df
        monthly_actuals_dict[iso]=temp_monthly_df
        yearly_actuals_dict[iso]=temp_yearly_df


    actuals_pnl_df = master_trades_df[master_trades_df.index.get_level_values('Date')>=actuals_start_date]
    actuals_pnl_df = actuals_pnl_df[['iso','Node ID','Node Name','TOT_PnL']]
    actuals_pnl_df = actuals_pnl_df.rename(columns = {'iso':'ISO'})

    iso_actuals_pnl_df = actuals_pnl_df.groupby([actuals_pnl_df.index.get_level_values('Date'),actuals_pnl_df.index.get_level_values('HourEnding'),'ISO']).sum()
    iso_actuals_pnl_df.reset_index(inplace=True)
    iso_actuals_pnl_df.set_index(['Date','HourEnding'],inplace=True,drop=True)

    node_name_node_id_dict = dict(zip(actuals_pnl_df['Node Name'], actuals_pnl_df['Node ID']))
    node_name_iso_dict = dict(zip(actuals_pnl_df['Node Name'], actuals_pnl_df['ISO']))

    backtest_pnl_df = pd.read_csv(save_directory+backtest_pnl_filename+'.csv',index_col=['Date','HE'],parse_dates=True)
    backtest_pnl_df.index.names = ['Date','HourEnding']
    limited_backtest_pnl_df = pd.DataFrame()

    # Get all hourly backtest data from past 5 years that is +-30 days from the past 30 days' of actuals (90 days total per year))
    for prev_year in range(0,6):
        start_date = backtest_start_date - datetime.timedelta(days=prev_year*365)
        end_date = backtest_end_date - datetime.timedelta(days=prev_year*365)
        temp_backtest_pnl_df = backtest_pnl_df[(backtest_pnl_df.index.get_level_values('Date')>=start_date) & (backtest_pnl_df.index.get_level_values('Date')<=end_date)]
        if limited_backtest_pnl_df.empty:
            limited_backtest_pnl_df=temp_backtest_pnl_df
        else:
            limited_backtest_pnl_df = pd.concat([limited_backtest_pnl_df,temp_backtest_pnl_df],axis=0,sort=True)

    iso_total_backtest_df = limited_backtest_pnl_df[[col for col in limited_backtest_pnl_df.columns if 'Total' in col]]
    iso_total_backtest_df.columns = [col.split('_Total')[0] for col in iso_total_backtest_df.columns]
    iso_total_backtest_df = iso_total_backtest_df.replace(0,np.nan)

    limited_backtest_pnl_df.columns = [col.split('_SD')[0] for col in limited_backtest_pnl_df.columns]
    limited_backtest_pnl_df.columns = [col.split('_',1)[1] for col in limited_backtest_pnl_df.columns]

    compare_df = pd.DataFrame(columns=['Node','ActSamp','ActMed','ActMean','ActMin','ActMax','ActHR','TestSamp','TestMed', 'TestMean','TestMin','TestMax','TestHR','TStat','MeanProb','MWStat','DistProb'])
    iso_compare_df = pd.DataFrame(columns=['ISO', 'ActSamp', 'ActMed', 'ActMean', 'ActMin', 'ActMax','ActHR', 'TestSamp', 'TestMed', 'TestMean','TestMin', 'TestMax','TestHR', 'TStat', 'MeanProb', 'MWStat', 'DistProb'])

    warnings.filterwarnings('ignore')

    for node in actuals_pnl_df['Node Name'].unique():
        actuals_df = actuals_pnl_df[actuals_pnl_df['Node Name']==node]
        backtest_df = limited_backtest_pnl_df[[col for col in limited_backtest_pnl_df.columns if str(node) in str(col)]].dropna()
        actuals = actuals_df['TOT_PnL'].values

        try:
            backtest = backtest_df[backtest_df.columns[0]].values

            act_correct_trades = len([x for x in actuals if x > 0])
            act_trades = len([x for x in actuals if x != 0])
            test_correct_trades = len([x for x in backtest if x > 0])
            test_trades = len([x for x in backtest if x != 0])


            t_stat,t_pval = ttest_ind(actuals,backtest)
            mw_stat, mw_pval = mannwhitneyu(actuals,backtest)


            new_dict = {'Node':node,
                        'ActSamp':len(actuals),
                        'ActMed':np.median(actuals),
                        'ActMean':np.mean(actuals),
                        'ActMin':np.min(actuals),
                        'ActMax':np.max(actuals),
                        'ActHR':act_correct_trades/act_trades,
                        'TestSamp':len(backtest),
                        'TestMed':np.median(backtest),
                        'TestMean':np.mean(backtest),
                        'TestMin':np.min(backtest),
                        'TestMax':np.max(backtest),
                        'TestHR':test_correct_trades/test_trades,
                        'TStat':t_stat,
                        'MeanProb':t_pval,
                        'MWStat':mw_stat,
                        'DistProb':mw_pval}

            compare_df = compare_df.append(new_dict,ignore_index=True)
        except:
            continue

    for iso in iso_actuals_pnl_df['ISO'].unique():
        actuals_df = iso_actuals_pnl_df[iso_actuals_pnl_df['ISO']==iso]
        backtest_df = iso_total_backtest_df[[col for col in iso_total_backtest_df.columns if iso in col]].dropna()
        actuals = actuals_df['TOT_PnL'].values


        try:
            backtest = backtest_df[backtest_df.columns[0]].values
            act_correct_trades = len([x for x in actuals if x > 0])
            act_trades = len([x for x in actuals if x != 0])
            test_correct_trades = len([x for x in backtest if x > 0])
            test_trades = len([x for x in backtest if x != 0])

            t_stat,t_pval = ttest_ind(actuals,backtest)
            mw_stat, mw_pval = mannwhitneyu(actuals,backtest)

            new_dict = {'ISO':iso,
                        'ActSamp':len(actuals),
                        'ActMed':np.median(actuals),
                        'ActMean':np.mean(actuals),
                        'ActMin':np.min(actuals),
                        'ActMax':np.max(actuals),
                        'ActHR': act_correct_trades / act_trades,
                        'TestSamp':len(backtest),
                        'TestMed':np.median(backtest),
                        'TestMean':np.mean(backtest),
                        'TestMin':np.min(backtest),
                        'TestMax':np.max(backtest),
                        'TestHR': test_correct_trades / test_trades,
                        'TStat':t_stat,
                        'MeanProb':t_pval,
                        'MWStat':mw_stat,
                        'DistProb':mw_pval}

            iso_compare_df = iso_compare_df.append(new_dict,ignore_index=True)
        except:
            continue

    warnings.filterwarnings('default')

    compare_df['Node Name'] = compare_df['Node'].apply(lambda row: node_name_node_id_dict[row])
    compare_df['ISO'] = compare_df['Node'].apply(lambda row: node_name_iso_dict[row])
    compare_df['Node'] = compare_df['Node'].astype('str')
    compare_df['%DeltSamp'] = (compare_df['ActSamp'] - compare_df['TestSamp'])/compare_df['ActSamp']
    compare_df['%DeltMed'] = (compare_df['ActMed']-compare_df['TestMed'])/compare_df['ActMed']
    compare_df['%DeltMean'] = (compare_df['ActMean'] - compare_df['TestMean'])/compare_df['ActMean']
    compare_df['%DeltMin'] = (compare_df['ActMin'] - compare_df['TestMin'])/compare_df['ActMin']
    compare_df['%DeltMax'] = (compare_df['ActMax'] - compare_df['TestMax'])/compare_df['ActMax']
    compare_df['%DeltHR'] = (compare_df['ActHR'] - compare_df['TestHR'])/compare_df['ActHR']
    compare_df = compare_df.sort_values(by='DistProb',ascending=True)
    compare_df = compare_df.round(2)
    compare_df.set_index(['Node Name'],inplace=True)

    iso_compare_df['%DeltSamp'] = (iso_compare_df['ActSamp'] - iso_compare_df['TestSamp'])/iso_compare_df['ActSamp']
    iso_compare_df['%DeltMed'] = (iso_compare_df['ActMed']-iso_compare_df['TestMed'])/iso_compare_df['ActMed']
    iso_compare_df['%DeltMean'] = (iso_compare_df['ActMean'] - iso_compare_df['TestMean'])/iso_compare_df['ActMean']
    iso_compare_df['%DeltMin'] = (iso_compare_df['ActMin'] - iso_compare_df['TestMin'])/iso_compare_df['ActMin']
    iso_compare_df['%DeltMax'] = (iso_compare_df['ActMax'] - iso_compare_df['TestMax'])/iso_compare_df['ActMax']
    iso_compare_df['%DeltHR'] = (iso_compare_df['ActHR'] - iso_compare_df['TestHR'])/iso_compare_df['ActHR']

    iso_compare_df = iso_compare_df.sort_values(by='DistProb',ascending=True)
    iso_compare_df = iso_compare_df.round(2)
    iso_compare_df.set_index('ISO', inplace=True)
    iso_compare_df = iso_compare_df[['ActHR','TestHR','%DeltHR','ActMed', 'TestMed', '%DeltMed', 'ActMean', 'TestMean', '%DeltMean', 'MeanProb', 'DistProb']]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        daily_writer = pd.ExcelWriter(save_directory + 'Daily_PnL_Results_'+model_type+'_'+name_adder+'.xlsx', engine='openpyxl')
        for iso, df in daily_actuals_dict.items():
            df.to_excel(daily_writer, sheet_name=iso, index=True)
        daily_writer.save()
        daily_writer.close()

        distr_writer = pd.ExcelWriter(save_directory + 'PnL_Distributions_'+model_type+'_'+name_adder+'.xlsx', engine='openpyxl')
        compare_df.to_excel(distr_writer, sheet_name='Nodal', index=True)
        iso_compare_df.to_excel(distr_writer, sheet_name='ISOs', index=True)
        distr_writer.save()
        distr_writer.close()

    compare_df = compare_df[compare_df['ActSamp']>=5]
    compare_df = compare_df[compare_df['TestSamp']>= 10]
    compare_df = compare_df[compare_df['%DeltMed'] <= 0]
    compare_df = compare_df[compare_df['%DeltMean'] <= 0]
    compare_df = compare_df[['ISO','ActHR','TestHR','%DeltHR','ActMed', 'TestMed', '%DeltMed', 'ActMean', 'TestMean', '%DeltMean', 'MeanProb', 'DistProb']]

    if do_printcharts:
        auto_open=True
    else:
        auto_open=False

    if do_printcharts:
        summary_pnl_fig = print_summary_pnl(isos=isos,
                                            daily_actuals_dict=daily_actuals_dict,
                                            monthly_actuals_dict=monthly_actuals_dict,
                                            yearly_actuals_dict=yearly_actuals_dict,
                                            compare_df=compare_df,
                                            iso_compare_df=iso_compare_df,
                                            name_adder=name_adder,
                                            model_type=model_type)
        url = plotly.offline.plot(summary_pnl_fig,filename=save_directory + 'SummaryPnL_' + predict_date_str_mm_dd_yyyy + '_'+model_type+'_'+name_adder +'.html',auto_open=auto_open)

        daily_pnl_fig = print_daily_pnl(trades_dict=trades_dict,
                                        isos=isos,
                                        date=predict_date_str_mm_dd_yyyy,
                                        name_adder=name_adder,
                                        model_type=model_type)
        url = plotly.offline.plot(daily_pnl_fig,filename=save_directory + 'DailyPnL_' + predict_date_str_mm_dd_yyyy + '_'+model_type+'_'+name_adder +'.html',auto_open=auto_open)

    ## Save a backup copy of the PnL alternating names based on day of year
    day_of_year = datetime.datetime.now().timetuple().tm_yday
    alt_pnl_num = str(day_of_year%2)
    master_trades_df.to_csv(save_directory + '2020_MASTER_PnL_'+model_type +'_'+ name_adder + '_BACKUP_'+alt_pnl_num+'.csv')

    print('')
    print('Daily PnL Complete For: '+predict_date_str_mm_dd_yyyy)
    print('')
    print('')
    print('**********************************************************************************')

    return trades_dict

def print_daily_pnl(trades_dict,isos,date,name_adder, model_type):
    figures_dict = {}
    hit_rate_df = pd.DataFrame({'ISO':[],'Inc_Trds': [], 'Inc_HR': [], 'Dec_Trds': [], 'Dec_HR': [], 'Tot_Trds': [], 'Tot_HR': []})
    summary_df = pd.DataFrame({'ISO':[],'Inc_$': [], 'Dec_$': [], 'Ene_$': [], 'Con_$': [], 'Los_$': [], 'Tot_$': [], 'MW': [], '$/MW': []})
    all_inc_trades = 0
    all_dec_trades = 0
    all_succ_inc_trades = 0
    all_succ_dec_trades = 0

    for iso, df in trades_dict.items():

        if df.empty:
            print('No Trades for ISO: ' + iso)
            figures_dict[iso] = [go.Bar(), go.Bar(), go.Bar(),go.Bar(), go.Table(), go.Table(), go.Table()]
            continue

        df.reset_index(inplace=True)
        df = df.rename(columns={'HourEnding': 'HE'})

        date = df['Date'][0].strftime("%Y_%m_%d")
        x = range(1, 25, 1)
        x_df = pd.DataFrame(index=x)
        x_df['MW'] = 0
        df = df.set_index(['HE'])
        nodal_df = df.reset_index()
        nodal_df['Node ID'] = nodal_df['Node ID'].astype('str').apply(lambda row: row.replace(' ', '')[0:16])
        nodal_df.set_index(['Node ID'], inplace=True)
        nodal_df = nodal_df.groupby(level=[0]).sum()

        nodal_df = nodal_df.rename(
            columns={'ENERGY_PnL': 'Ene_$', 'CONG_PnL': 'Con_$', 'LOSS_PnL': 'Los_$', 'TOT_PnL': 'Tot_$'})
        nodal_df['$/MW'] = nodal_df['Tot_$'] / nodal_df['MW']
        nodal_df = nodal_df[['Ene_$', 'Con_$', 'Los_$', 'MW', '$/MW', 'Tot_$']]

        worst_pnl_df = nodal_df.sort_values('Tot_$', ascending=True)
        worst_locations_df = worst_pnl_df
        worst_locations_df = worst_locations_df.round(
            {'Ene_$': 0, 'Con_$': 0, 'Los_$': 0, 'Tot_$': 0, 'MW': 0, '$/MW': 2})
        worst_locations_df = worst_locations_df.drop_duplicates(keep='first')
        worst_locations_df.reset_index(inplace=True)

        df_inc_master = df[df['Trade Type'] == 'INC']
        df_dec_master = df[df['Trade Type'] == 'DEC']
        inc_succ_trades = df_inc_master['SUCCESS_TRADE'].sum()
        dec_succ_trades = df_dec_master['SUCCESS_TRADE'].sum()
        tot_succ_trades = inc_succ_trades + dec_succ_trades
        inc_trades = len(df_inc_master)
        dec_trades = len(df_dec_master)
        tot_trades = inc_trades + dec_trades

        all_inc_trades = all_inc_trades + inc_trades
        all_dec_trades = all_dec_trades + dec_trades
        all_succ_inc_trades = all_succ_inc_trades + inc_succ_trades
        all_succ_dec_trades = all_succ_dec_trades + dec_succ_trades

        df_inc = df_inc_master.groupby(level=[0]).sum()
        df_dec = df_dec_master.groupby(level=[0]).sum()

        if tot_trades == 0:
            hr = 0
        else:
            hr = tot_succ_trades / tot_trades

        if inc_trades == 0:
            inc_hr = 0
        else:
            inc_hr = inc_succ_trades / inc_trades

        if dec_trades == 0:
            dec_hr = 0
        else:
            dec_hr = dec_succ_trades / dec_trades

        iso_hit_rate_df = pd.DataFrame(
            {'ISO': [iso], 'Inc_Trds': [inc_trades], 'Dec_Trds': [dec_trades], 'Tot_Trds': [tot_trades],
             'Inc_HR': [inc_hr], 'Dec_HR': [dec_hr], 'Tot_HR': [hr]})
        iso_hit_rate_df = iso_hit_rate_df.round(
            {'Inc_Trds': 0, 'Inc_HR': 2, 'Dec_Trds': 0, 'Dec_HR': 2, 'Tot_Trds': 0, 'Tot_HR': 2})
        if hit_rate_df.empty:
            hit_rate_df = iso_hit_rate_df
        else:
            hit_rate_df = pd.concat([hit_rate_df, iso_hit_rate_df], axis=0, ignore_index=True)

        df_inc = pd.DataFrame(df_inc[['SUCCESS_TRADE', 'MW', 'ENERGY_PnL', 'CONG_PnL', 'LOSS_PnL', 'TOT_PnL']])
        df_dec = pd.DataFrame(df_dec[['SUCCESS_TRADE', 'MW', 'ENERGY_PnL', 'CONG_PnL', 'LOSS_PnL', 'TOT_PnL']])
        df_inc = df_inc.reindex(x_df.index, fill_value=0)
        df_dec = df_dec.reindex(x_df.index, fill_value=0)
        df_inc.columns = [col + '_INC' for col in df_inc.columns]
        df_dec.columns = [col + '_DEC' for col in df_dec.columns]
        df_inc = df_inc.rename(columns={'TOT_PnL_INC': 'Inc_$'})
        df_dec = df_dec.rename(columns={'TOT_PnL_DEC': 'Dec_$'})
        df_inc_dec = pd.concat([df_inc, df_dec], axis=1)
        df_inc_dec.reset_index(inplace=True)
        df_inc_dec.rename(columns={'index': 'HE'}, inplace=True)
        df_inc_dec['MW'] = df_inc_dec['MW_INC'] + df_inc_dec['MW_DEC']
        df_inc_dec['Tot_$'] = df_inc_dec['Inc_$'] + df_inc_dec['Dec_$']
        df_inc_dec['Ene_$'] = df_inc_dec['ENERGY_PnL_INC'] + df_inc_dec['ENERGY_PnL_DEC']
        df_inc_dec['Con_$'] = df_inc_dec['CONG_PnL_INC'] + df_inc_dec['CONG_PnL_DEC']
        df_inc_dec['Los_$'] = df_inc_dec['LOSS_PnL_INC'] + df_inc_dec['LOSS_PnL_DEC']
        df_inc_dec['$/MW'] = df_inc_dec['Tot_$'] / df_inc_dec['MW']
        df_inc_dec.loc['<b>Total<b>'] = df_inc_dec.sum()
        df_inc_dec['$/MW'] = df_inc_dec['Tot_$'] / df_inc_dec['MW']
        df_inc_dec.loc[df_inc_dec['HE'] == 300, 'HE'] = '<b>Total<b>'
        df_inc_dec = df_inc_dec[['HE', 'Inc_$', 'Dec_$', 'Ene_$', 'Con_$', 'Los_$', 'MW', '$/MW', 'Tot_$']]

        iso_summary_df = pd.DataFrame(df_inc_dec.drop(df_inc_dec.tail(1).index).sum()[1:]).T
        iso_summary_df['$/MW'] = iso_summary_df['Tot_$'] / iso_summary_df['MW']
        iso_summary_df['ISO'] = iso
        iso_summary_df.set_index('ISO', inplace=True)

        df_inc_dec = df_inc_dec.round(
            {'HE': 0, 'Inc_$': 0, 'Dec_$': 0, 'Ene_$': 0, 'Con_$': 0, 'Los_$': 0, 'Tot_$': 0, 'MW': 0, '$/MW': 2})

        iso_summary_df.reset_index(inplace=True)
        if summary_df.empty:
            summary_df = iso_summary_df
        else:
            summary_df = pd.concat([summary_df, iso_summary_df], axis=0, ignore_index=True)

        ener_bar = go.Bar(name='ENER_' + iso, x=df_inc_dec['HE'][:-1], y=df_inc_dec['Ene_$'].values[:-1],
                          marker_color='#4CB5F5')

        cong_bar = go.Bar(name='CONG_' + iso, x=df_inc_dec['HE'][:-1], y=df_inc_dec['Con_$'].values[:-1],
                          marker_color='#1F3F49')

        loss_bar = go.Bar(name='LOSS_' + iso, x=df_inc_dec['HE'][:-1], y=df_inc_dec['Los_$'].values[:-1],
                          marker_color='#D32D41')

        tot_bar = go.Bar(name='TOT_' + iso, x=df_inc_dec['HE'][:-1], y=df_inc_dec['Tot_$'].values[:-1],
                         marker_color='#6AB187')

        table1 = go.Table(
            header=dict(
                values=df_inc_dec.columns,
                font=dict(size=14, color='white'),
                align="left",
                fill=dict(color=['#B3C100'])
            ),
            cells=dict(
                values=[df_inc_dec[k].tolist() for k in df_inc_dec.columns[0:]],
                align="left",
                format=[None, "$,d", "$,d", "$,d", "$,d", "$,d", ",d", "$.2f", "$,d"],
                fill=dict(color=['#23282D', '#CED2CC', '#CED2CC', '#CED2CC', '#CED2CC', '#CED2CC', '#CED2CC', '#CED2CC',
                                 '#23282D']),
                font=dict(color=['white', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'white']))
        )

        table2 = go.Table(
            columnwidth=[200, 80, 80, 80, 80, 80, 80],
            header=dict(
                values=worst_locations_df.columns,
                font=dict(size=14, color='white'),
                align="left",
                fill=dict(color=['#4CB5F5'])
            ),
            cells=dict(
                values=[worst_locations_df[k].tolist() for k in worst_locations_df.columns[0:]],
                align="left",
                format=[None, "$,d", "$,d", "$,d", ",d", "$.2f", "$,d"],
                fill=dict(color=['#23282D', '#CED2CC', '#CED2CC', '#CED2CC', '#CED2CC', '#CED2CC', '#23282D']),
                font=dict(color=['white', 'black', 'black', 'black', 'black', 'black', 'white']))
        )

        figures_dict[iso] = [loss_bar, cong_bar, ener_bar, tot_bar, table1, table2]

    if summary_df.empty==False:
        summary_df.set_index('ISO', inplace=True)
        summary_df.loc['<b>Total<b>'] = summary_df.sum()
        summary_df['$/MW'] = summary_df['Tot_$'] / summary_df['MW']
        summary_df = summary_df.round(
            {'Inc_$': 0, 'Dec_$': 0, 'Ene_$': 0, 'Con_$': 0, 'Los_$': 0, 'Tot_$': 0, 'MW': 0, '$/MW': 2})
        summary_df.reset_index(inplace=True)

        hit_rate_df.set_index(['ISO'], inplace=True)
        hit_rate_df.loc['<b>Total<b>'] = hit_rate_df.sum()
        hit_rate_df.loc['<b>Total<b>']['Dec_HR'] = all_succ_dec_trades / all_dec_trades
        hit_rate_df.loc['<b>Total<b>']['Inc_HR'] = all_succ_inc_trades / all_inc_trades
        hit_rate_df.loc['<b>Total<b>']['Tot_HR'] = (all_succ_inc_trades + all_succ_dec_trades) / (
                    all_inc_trades + all_dec_trades)
        hit_rate_df = hit_rate_df.round(
            {'Inc_Trds': 0, 'Inc_HR': 2, 'Dec_Trds': 0, 'Dec_HR': 2, 'Tot_Trds': 0, 'Tot_HR': 2})
        hit_rate_df.reset_index(inplace=True)

    summary_table = go.Table(
        header=dict(
            values=summary_df.columns,
            font=dict(size=14, color='white'),
            align="left",
            fill=dict(color=['#B3C100'])
        ),
        cells=dict(
            values=[summary_df[k].tolist() for k in summary_df.columns[0:]],
            align="left",
            format=[None, "$,d", "$,d", "$,d", "$,d", "$,d", ",d", "$.2f", "$,d"],
            fill=dict(color=['#23282D', '#CED2CC', '#CED2CC', '#CED2CC', '#CED2CC', '#CED2CC', '#CED2CC', '#CED2CC',
                             '#23282D']),
            font=dict(color=['white', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'white'])))

    hit_rate_table = go.Table(
        header=dict(
            values=hit_rate_df.columns,
            font=dict(size=14, color='white'),
            align="left",
            fill=dict(color=['#4CB5F5'])
        ),
        cells=dict(
            values=[hit_rate_df[k].tolist() for k in hit_rate_df.columns[0:]],
            align="left",
            format=[None, None, None, None, '%', '%', '%'],
            fill=dict(color=['#23282D', '#CED2CC', '#CED2CC', '#CED2CC', '#CED2CC', '#CED2CC', '#23282D']),
            font=dict(color=['white', 'black', 'black', 'black', 'black', 'black', 'white']))
    )

    ener_sum_bar = go.Bar(name='ENER_$', x=summary_df['ISO'], y=summary_df['Ene_$'].values,
                          marker_color='#4CB5F5')

    cong_sum_bar = go.Bar(name='CONG_$', x=summary_df['ISO'], y=summary_df['Con_$'].values,
                          marker_color='#1F3F49')

    loss_sum_bar = go.Bar(name='LOSS_$', x=summary_df['ISO'], y=summary_df['Los_$'].values,
                          marker_color='#D32D41')

    tot_sum_bar = go.Bar(name='TOT_$', x=summary_df['ISO'], y=summary_df['Tot_$'].values,
                         marker_color='#6AB187')

    specs1 = [[{"type": "table"}, {"type": "bar"}],
              [{"type": "bar"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}]]

    specs2 = [[{"type": "table"}, {"type": "bar"}, {"type": "bar"}],
              [{"type": "bar"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}]]

    specs3 = [[{"type": "table"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
              [{"type": "bar"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}]]

    specs4 = [[{"type": "table"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
              [{"type": "bar"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}]]

    specs5 = [[{"type": "table"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
              [{"type": "bar"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"},
               {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"},
               {"type": "table"}]]

    specs6 = [
        [{"type": "table"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"},
         {"type": "bar"}],
        [{"type": "bar"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"},
         {"type": "table"}],
        [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"},
         {"type": "table"}, {"type": "table"}]]

    specs_dict = {'1': specs1, '2': specs2, '3': specs3, '4': specs4, '5': specs5, '6': specs6}

    if len(isos) == 1:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy Daily PnL Summary ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' PnL ' + date + '<b>',
                  '<b>Daily PnL By ISO', None,
                  '<b>Daily Hit Rate Summary ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Nodal Results<b>')
    elif len(isos) == 2:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy Daily PnL Summary ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' PnL ' + date + '<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' PnL ' + date + '<b>',
                  '<b>Daily PnL By ISO', None, None,
                  '<b>Daily Hit Rate Summary ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Nodal Results<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Nodal Results<b>')
    elif len(isos) == 3:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy Daily PnL Summary ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' PnL ' + date + '<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' PnL ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' PnL ' + date + '<b>',
                  '<b>Daily PnL By ISO', None, None, None,
                  '<b>Daily Hit Rate Summary ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Nodal Results<b>',
                  '<b>' + isos[1] + ' Nodal Results<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Nodal Results<b>')
    elif len(isos) == 4:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy Daily PnL Summary ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' PnL ' + date + '<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' PnL ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' PnL ' + date + '<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' PnL ' + date + '<b>',
                  '<b>Daily PnL By ISO', None, None, None, None,
                  '<b>Daily Hit Rate Summary ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Nodal Results<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Nodal Results<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Nodal Results<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Nodal Resultss<b>')
    elif len(isos) == 5:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy Daily PnL Summary ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' PnL ' + date + '<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' PnL ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' PnL ' + date + '<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' PnL ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[4] + ' PnL ' + date + '<b>',
                  '<b>Daily PnL By ISO', None, None, None, None, None,
                  '<b>Daily Hit Rate Summary ' + date + '<b>', '<b>' + name_adder + ' ' + isos[0] + ' Nodal Results<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Nodal Results<b>', '<b>' +model_type+' '+ name_adder + ' ' +isos[2] + ' Nodal Results<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' +isos[3] + ' Nodal Results<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[4] + ' Nodal Results<b>')
    elif len(isos) == 6:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy Daily PnL Summary ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' PnL ' + date + '<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' PnL ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' PnL ' + date + '<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' PnL ' + date + '<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[4] + ' PnL ' + date + '<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[5] + ' PnL ' + date + '<b>',
                  '<b>Daily PnL By ISO', None, None, None, None, None, None,
                  '<b>Daily Hit Rate Summary ' + date + '<b>', '<b>' + name_adder + ' ' + isos[0] + ' Nodal Results<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Nodal Results<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Nodal Results<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Nodal Results<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[4] + ' Nodal Results<b>',
                  '<b>' +model_type+' '+ name_adder + ' ' + isos[5] + ' Nodal Results<b>')

    fig = make_subplots(
        rows=3, cols=len(isos) + 1,
        shared_xaxes=True,
        row_heights=[0.45, 1.05, 0.45],
        vertical_spacing=0.05,
        specs=specs_dict[str(len(isos))],
        subplot_titles=titles
    )

    fig.add_trace(summary_table, 1, 1)
    fig.add_trace(loss_sum_bar, 2, 1)
    fig.add_trace(cong_sum_bar, 2, 1)
    fig.add_trace(ener_sum_bar, 2, 1)
    fig.add_trace(tot_sum_bar, 2, 1)
    fig.add_trace(hit_rate_table, 3, 1)

    for iso_num in range(0, len(isos)):
        fig.add_trace(figures_dict[isos[iso_num]][0], 1, iso_num + 2)
        fig.add_trace(figures_dict[isos[iso_num]][1], 1, iso_num + 2)
        fig.add_trace(figures_dict[isos[iso_num]][2], 1, iso_num + 2)
        fig.add_trace(figures_dict[isos[iso_num]][3], 1, iso_num + 2)
        fig.add_trace(figures_dict[isos[iso_num]][4], 2, iso_num + 2)
        fig.add_trace(figures_dict[isos[iso_num]][5], 3, iso_num + 2)

    fig.update_layout(
        height=1300,
        width=1000 + (1000 * len(isos)),
        barmode='group',
        paper_bgcolor='white',
        plot_bgcolor='#CED2CC',
        font=dict(
            color="black")
    )

    for col in range(len(isos) + 2):
        fig.update_xaxes(tickmode='linear', row=1, col=col)
        fig.update_yaxes(tickformat="$", row=1, col=col)
        fig.update_yaxes(tickformat="$", row=2, col=col)


    return fig

def print_summary_pnl(isos,daily_actuals_dict,monthly_actuals_dict,yearly_actuals_dict,compare_df,iso_compare_df,name_adder, model_type):
    figures_dict = {}
    summary_daily_pnl_df = pd.DataFrame()
    summary_monthly_pnl_df = pd.DataFrame()
    summary_yearly_pnl_df = pd.DataFrame()

    for iso, df in daily_actuals_dict.items():
        daily_df = daily_actuals_dict[iso]
        monthly_df = monthly_actuals_dict[iso]
        yearly_df = yearly_actuals_dict[iso]


        indv_iso_compare_df = compare_df[compare_df['ISO']==iso].copy()
        indv_iso_compare_df.drop(columns='ISO',inplace=True)
        indv_iso_compare_df.sort_values(['DistProb'],inplace=True,ascending=True)
        indv_iso_compare_df.reset_index(inplace=True)
        indv_iso_compare_df['Node Name'] = indv_iso_compare_df['Node Name'].astype('str').apply(lambda row: row.replace(' ', '')[0:16])
        indv_iso_compare_df.set_index('Node Name',inplace=True,drop=True)

        temp_daily_df = daily_df.copy()
        temp_monthly_df = monthly_df.copy()
        temp_yearly_df = yearly_df.copy()

        temp_daily_df = temp_daily_df.sort_index(ascending=False)
        temp_monthly_df = temp_monthly_df.sort_index(ascending=False)
        temp_yearly_df = temp_yearly_df.sort_index(ascending=False)

        temp_daily_df.index = temp_daily_df.index.get_level_values('Date').strftime('%m/%d/%y')
        temp_monthly_df.index = temp_monthly_df.index.get_level_values('Month').strftime('%b%Y')
        temp_yearly_df.index = temp_yearly_df.index.get_level_values('Year').strftime('%Y')

        temp_daily_df.index.names = ['Date']
        temp_monthly_df.index.names = ['Month']
        temp_yearly_df.index.names = ['Year']

        table1 = go.Table(
            header=dict(
                values=temp_daily_df.reset_index().columns,
                font=dict(size=12, color='white'),
                align="left",
                fill=dict(color=['#1C4E80'])
            ),
            cells=dict(
                values=[temp_daily_df.reset_index()[k].tolist() for k in temp_daily_df.reset_index().columns[0:]],
                align="left",
                format=[None, "$,d", "$,d", "$,d", ",d", "$.2f", "$,d"],
                fill=dict(color=['#202020', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA']),
                font=dict(color=['white', 'black', 'black', 'black', 'black', 'black', 'black']))
        )

        table2 = go.Table(
            header=dict(
                values=temp_monthly_df.reset_index().columns,
                font=dict(size=12, color='white'),
                align="left",
                fill=dict(color=['#1C4E80'])
            ),
            cells=dict(
                values=[temp_monthly_df.reset_index()[k].tolist() for k in temp_monthly_df.reset_index().columns[0:]],
                align="left",
                format=[None, "$,d", "$,d", "$,d", ",d", "$.2f", "$,d"],
                fill=dict(color=['#202020', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA']),
                font=dict(color=['white', 'black', 'black', 'black', 'black', 'black', 'black']))
        )

        table3 = go.Table(
            header=dict(
                values=temp_yearly_df.reset_index().columns,
                font=dict(size=12, color='white'),
                align="left",
                fill=dict(color=['#1C4E80'])
            ),
            cells=dict(
                values=[temp_yearly_df.reset_index()[k].tolist() for k in temp_yearly_df.reset_index().columns[0:]],
                align="left",
                format=[None, "$,d", "$,d", "$,d", ",d", "$.2f", "$,d"],
                fill=dict(color=['#202020', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA']),
                font=dict(color=['white', 'black', 'black', 'black', 'black', 'black', 'black']))
        )

        table4 = go.Table(
            columnwidth=[200, 80, 80, 80, 80, 80, 80,80,80,80,80,80],
            header=dict(
                values=indv_iso_compare_df.reset_index().columns,
                font=dict(size=12, color='white'),
                align="left",
                fill=dict(color=['#0091D5'])
            ),
            cells=dict(
                values=[indv_iso_compare_df.reset_index()[k].tolist() for k in indv_iso_compare_df.reset_index().columns[0:]],
                align="left",
                format=[None, "%", "%","%", "$.0f", "$.0f","%", "$.0f", "$.0f","%","%","%"],
                fill=dict(color=['#202020', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA', '#DADADA', '#DADADA', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA',
                                 '#DADADA']),
                font=dict(color=['white', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black'])))

        daily_df = daily_df[['MW', 'Tot_$', '$/MW']]
        monthly_df = monthly_df[['MW',  'Tot_$','$/MW']]
        yearly_df = yearly_df[['MW', 'Tot_$', '$/MW']]

        daily_df=daily_df.T.set_index(np.repeat(iso,daily_df.shape[1]), append=True).T
        monthly_df=monthly_df.T.set_index(np.repeat(iso,monthly_df.shape[1]), append=True).T
        yearly_df= yearly_df.T.set_index(np.repeat(iso,yearly_df.shape[1]), append=True).T

        daily_df.columns = daily_df.columns.swaplevel(0, 1)
        monthly_df.columns = monthly_df.columns.swaplevel(0, 1)
        yearly_df.columns = yearly_df.columns.swaplevel(0, 1)

        if summary_daily_pnl_df.empty:
            summary_daily_pnl_df = daily_df
            summary_monthly_pnl_df = monthly_df
            summary_yearly_pnl_df = yearly_df

        else:
            summary_daily_pnl_df = pd.concat([summary_daily_pnl_df,daily_df],join='outer',axis=1)
            summary_monthly_pnl_df = pd.concat([summary_monthly_pnl_df,monthly_df],join='outer',axis=1)
            summary_yearly_pnl_df = pd.concat([summary_yearly_pnl_df,yearly_df],join='outer',axis=1)

        figures_dict[iso] = [table1, table2, table3, table4]

    summary_daily_pnl_df['Total','MW'] = summary_daily_pnl_df[[col for col in summary_daily_pnl_df.columns if (('MW' in col[1]) & ('$' not in col[1]))]].sum(axis=1)
    summary_monthly_pnl_df['Total','MW'] = summary_monthly_pnl_df[[col for col in summary_monthly_pnl_df.columns if (('MW' in col[1]) & ('$' not in col[1]))]].sum(axis=1)
    summary_yearly_pnl_df['Total','MW'] = summary_yearly_pnl_df[[col for col in summary_yearly_pnl_df.columns if (('MW' in col[1]) & ('$' not in col[1]))]].sum(axis=1)

    summary_daily_pnl_df['Total','Tot_$'] = summary_daily_pnl_df[[col for col in summary_daily_pnl_df.columns if 'Tot_$' in col[1]]].sum(axis=1)
    summary_monthly_pnl_df['Total','Tot_$'] = summary_monthly_pnl_df[[col for col in summary_monthly_pnl_df if 'Tot_$' in col[1]]].sum(axis=1)
    summary_yearly_pnl_df['Total','Tot_$'] = summary_yearly_pnl_df[[col for col in summary_yearly_pnl_df.columns if 'Tot_$' in col[1]]].sum(axis=1)

    summary_daily_pnl_df['Total','$/MW'] = (summary_daily_pnl_df['Total','Tot_$']/summary_daily_pnl_df['Total','MW']).round(2)
    summary_monthly_pnl_df['Total','$/MW'] = (summary_monthly_pnl_df['Total','Tot_$']/summary_monthly_pnl_df['Total','MW']).round(2)
    summary_yearly_pnl_df['Total','$/MW'] = (summary_yearly_pnl_df['Total','Tot_$']/summary_yearly_pnl_df['Total','MW']).round(2)


    summary_daily_pnl_df = summary_daily_pnl_df.sort_index(ascending=False)
    summary_monthly_pnl_df = summary_monthly_pnl_df.sort_index(ascending=False)
    summary_yearly_pnl_df = summary_yearly_pnl_df.sort_index(ascending=False)

    summary_daily_pnl_df.index = summary_daily_pnl_df.index.get_level_values('Date').strftime('%m/%d/%y')
    summary_monthly_pnl_df.index = summary_monthly_pnl_df.index.get_level_values('Month').strftime('%b%Y')
    summary_yearly_pnl_df.index = summary_yearly_pnl_df.index.get_level_values('Year').strftime('%Y')

    summary_daily_pnl_df.index.names = ['Date']
    summary_monthly_pnl_df.index.names = ['Month']
    summary_yearly_pnl_df.index.names = ['Year']

    summary_daily_pnl_df = summary_daily_pnl_df.head(21)


    summary_table_1 = go.Table(
        columnwidth= [130]+[110]*(3*(len(isos)+1)),
        header=dict(
            values=summary_daily_pnl_df.reset_index().columns,
            font=dict(size=12, color='white'),
            align="left",
            fill=dict(color='#1C4E80')
        ),
        cells=dict(
            values=[summary_daily_pnl_df.reset_index()[k].tolist() for k in summary_daily_pnl_df.reset_index().columns[0:]],
            align="left",
            format=[None, ",d","$,d", "$.2f", ",d","$,d", "$.2f", ",d","$,d", "$.2f", ",d","$,d", "$.2f", ",d","$,d", "$.2f", ",d","$,d", "$.2f" ],
            fill=dict(color=['#202020','#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA', '#DADADA', '#DADADA','#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA', '#DADADA', '#DADADA','#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA', '#DADADA', '#DADADA']),
            font=dict(color=['white','black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black','black', 'black', 'black', 'black', 'black', 'black'])))

    summary_table_2 = go.Table(
        columnwidth=[110] + [110] * (3 * (len(isos) + 1)),
        header=dict(
            values=summary_monthly_pnl_df.reset_index().columns,
            font=dict(size=12, color='white'),
            align="left",
            fill=dict(color='#1C4E80')
        ),
        cells=dict(
            values=[summary_monthly_pnl_df.reset_index()[k].tolist() for k in summary_monthly_pnl_df.reset_index().columns[0:]],
            align="left",
            format=[None, ",d","$,d", "$.2f", ",d","$,d", "$.2f", ",d","$,d", "$.2f", ",d","$,d", "$.2f", ",d","$,d", "$.2f", ",d","$,d", "$.2f" ],
            fill=dict(color=['#202020','#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA', '#DADADA', '#DADADA','#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA', '#DADADA', '#DADADA','#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA', '#DADADA', '#DADADA']),
            font=dict(color=['white','black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black','black', 'black', 'black', 'black', 'black', 'black'])))

    summary_table_3 = go.Table(
        columnwidth=[110] + [110] * (3 * (len(isos) + 1)),
        header=dict(
            values=summary_yearly_pnl_df.reset_index().columns,
            font=dict(size=12, color='white'),
            align="left",
            fill=dict(color='#1C4E80')
        ),
        cells=dict(
            values=[summary_yearly_pnl_df.reset_index()[k].tolist() for k in summary_yearly_pnl_df.reset_index().columns[0:]],
            align="left",
            format=[None, ",d","$,d", "$.2f", ",d","$,d", "$.2f", ",d","$,d", "$.2f", ",d","$,d", "$.2f", ",d","$,d", "$.2f", ",d","$,d", "$.2f" ],
            fill=dict(color=['#202020','#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA', '#DADADA', '#DADADA','#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA', '#DADADA', '#DADADA','#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA', '#DADADA', '#DADADA']),
            font=dict(color=['white','black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black','black', 'black', 'black', 'black', 'black', 'black'])))

    summary_table_4 = go.Table(
        header=dict(
            values=iso_compare_df.reset_index().columns,
            font=dict(size=12, color='white'),
            align="left",
            fill=dict(color=['#0091D5'])
        ),
        cells=dict(
            values=[iso_compare_df.reset_index()[k].tolist() for k in iso_compare_df.reset_index().columns[0:]],
            align="left",
            format=[None, "%", "%", "%", "$.0f", "$.0f", "%", "$.0f", "$.0f", "%", "%", "%"],
            fill=dict(color=['#202020', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#DADADA', '#DADADA', '#DADADA', '#F1F1F1',
                             '#F1F1F1', '#F1F1F1', '#DADADA',
                             '#DADADA']),
            font=dict(color=['white', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black',
                             'black', 'black'])))



    specs1 = [[{"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}]]

    specs2 = [[{"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}]]

    specs3 = [[{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}]]

    specs4 = [[{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],]

    specs5 = [[{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}]]

    specs6 = [[{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"},{"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}],
              [{"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}, {"type": "table"}]]

    specs_dict = {'1': specs1, '2': specs2, '3': specs3, '4': specs4, '5': specs5, '6': specs6}


    if len(isos) == 1:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Daily PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Axon Energy Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' +isos[0] + ' Monthly PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Axon Energy Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Yearly PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Hourly PnL Comparison To Backtest<b>')
    elif len(isos) == 2:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Daily PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Axon Energy Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Monthly PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Axon Energy Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Yearly PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Hourly PnL Comparison To Backtest<b>')
    elif len(isos) == 3:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Daily PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Axon Energy Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Monthly PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Axon Energy Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Yearly PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Hourly PnL Comparison To Backtest<b>')
    elif len(isos) == 4:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Daily PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Axon Energy Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Monthly PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Axon Energy Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Yearly PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Hourly PnL Comparison To Backtest<b>')
    elif len(isos) == 5:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[4] + ' Daily PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Axon Energy Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[4] + ' Monthly PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Axon Energy Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[4] + ' Yearly PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[4] + ' Hourly PnL Comparison To Backtest<b>')
    elif len(isos) == 6:
        titles = ('<b>'+model_type+' '+name_adder+' Axon Energy Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[4] + ' Daily PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[5] + ' Daily PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Axon Energy Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[4] + ' Monthly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[5] + ' Monthly PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Axon Energy Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' +isos[4] + ' Yearly PnL Summary<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[5] + ' Yearly PnL Summary<b>',
                  '<b>'+model_type+' '+name_adder+' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[0] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[1] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[2] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[3] + ' Hourly PnL Comparison To Backtest<b>', '<b>' +model_type+' '+ name_adder + ' ' + isos[4] + ' Hourly PnL Comparison To Backtest<b>', '<b>'+model_type+' '+ name_adder + ' ' + isos[5] + ' Hourly PnL Comparison To Backtest<b>')



    fig = make_subplots(
        rows=4, cols=len(isos) + 1,
        shared_xaxes=True,
        row_heights=[.5,.3,.15,.5],
        column_widths=[1.4] + [0.8]*len(isos),
        vertical_spacing=0.05,
        horizontal_spacing = 0.01,
        specs=specs_dict[str(len(isos))],
        subplot_titles=titles
    )

    fig.add_trace(summary_table_1, 1, 1)
    fig.add_trace(summary_table_2, 2, 1)
    fig.add_trace(summary_table_3, 3, 1)
    fig.add_trace(summary_table_4, 4, 1)

    for iso_num in range(0, len(isos)):
        fig.add_trace(figures_dict[isos[iso_num]][0], 1, iso_num + 2)
        fig.add_trace(figures_dict[isos[iso_num]][1], 2, iso_num + 2)
        fig.add_trace(figures_dict[isos[iso_num]][2], 3, iso_num + 2)
        fig.add_trace(figures_dict[isos[iso_num]][3], 4, iso_num + 2)

    fig.update_layout(
        height=1300,
        width=1300 + (800* len(isos)),
        paper_bgcolor='white',
        plot_bgcolor='#CED2CC',
        font=dict(
            color="black")
    )

    return fig

def print_var(var_dataframes_dict,predict_date_str_mm_dd_yyyy,name_adder,model_type):
    figures_dict = {}

    for type, df in var_dataframes_dict.items():

        table1 = go.Table(
            header=dict(
                values=df.reset_index().columns,
                font=dict(size=12, color='white'),
                align="left",
                fill=dict(color=['#1C4E80'])
            ),
            cells=dict(
                values=[df.reset_index()[k].tolist() for k in df.reset_index().columns[0:]],
                align="left",
                format=[None, "$,d", "$,d", "$,d", "", "$,d","$,d", "$,d"],
                fill=dict(color=['#202020', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#F1F1F1', '#F1F1F1']),
                font=dict(color=['white', 'red', 'red', 'red','black', 'green', 'green', 'green']))
        )
        figures_dict[type]=table1

    specs1 = [[{"type": "table"}],
              [{"type": "table"}],
              [{"type": "table"}],
              [{"type": "table"}]]

    titles = ('<b>'+model_type+' '+ name_adder + ' ' + predict_date_str_mm_dd_yyyy + ' Axon Energy 3-Year 90 Day Rolling VAR<b>',
              '<b>'+model_type+' '+ name_adder + ' ' + predict_date_str_mm_dd_yyyy + ' Axon Energy 1-Year VAR<b>',
              '<b>'+model_type+' '+ name_adder + ' ' + predict_date_str_mm_dd_yyyy + ' Axon Energy 2-Year VAR<b>',
              '<b>'+model_type+' '+ name_adder + ' ' + predict_date_str_mm_dd_yyyy + ' Axon Energy 3-Year VAR<b>',
)

    fig = make_subplots(
        rows=len(figures_dict), cols=1,
        shared_xaxes=True,
        row_heights=[.5, .5, .5,.5],
        vertical_spacing=0.05,
        horizontal_spacing=0.01,
        specs=specs1,
        subplot_titles=titles
    )

    counter=1
    for type, table in figures_dict.items():
        fig.add_trace(table, counter,1)
        counter+=1

    fig.update_layout(
        height=1300,
        width=1300,
        paper_bgcolor='white',
        plot_bgcolor='#CED2CC',
        font=dict(
            color="black")
    )

    return fig

def create_VAR(preds_dict, VAR_ISOs, historic_var_file_name, working_directory, static_directory, model_type,predict_date_str_mm_dd_yyyy,name_adder):
    VAR_files_directory = static_directory + '\ModelUpdateData\\'
    save_directory = working_directory + '\\DailyPnL\\'
    iso_daily_PnL = pd.DataFrame()

    # Find correct VAR file
    VAR_dict = load_obj(VAR_files_directory+historic_var_file_name)
    timezone_dict = {'MISO': 'EST', 'PJM': 'EPT', 'ISONE': 'EPT', 'NYISO': 'EPT', 'ERCOT': 'CPT', 'SPP': 'CPT'}


    for iso in VAR_ISOs:
        VAR_df = VAR_dict[timezone_dict[iso]]
        VAR_df = VAR_df[[col for col in VAR_df.columns if iso in col]]
        VAR_df = VAR_df.astype('float')
        pred_df = preds_dict[iso]

        # put preds in right format and make decs negative volumne
        mw_df = pred_df.pivot(index= 'Hour',columns='Node Name', values='MW')
        mw_df.fillna(0,inplace=True)
        trade_type_df = pred_df.pivot(index='Hour', columns='Node Name', values='Trade Type')
        trade_type_df.fillna(0, inplace=True)

        for col in trade_type_df.columns:
            trade_type_df.loc[trade_type_df[col]=='INC',col] = 1
            trade_type_df.loc[trade_type_df[col]=='DEC',col] = -1

        mw_df = mw_df * trade_type_df
        mw_df.columns = [iso+'_'+col  for col in mw_df.columns]

        if model_type=='SPREAD':
            if iso == 'ERCOT':
                mw_df.columns = [col.split('$')[0]+'$'+iso+'_'+col.split('$')[1] for col in mw_df.columns]

        # Combine VAR df and pred df

        VAR_df.reset_index(inplace=True)
        mw_df.reset_index(inplace=True)
        mw_df = mw_df.rename(columns={'Hour':'HourEnding'})

        full_pred_df = pd.merge(VAR_df,mw_df,on='HourEnding')
        full_pred_df.set_index(['Date','HourEnding'],inplace=True)
        full_pred_df.sort_values(['Date','HourEnding'],inplace=True)
        full_pred_df.fillna(0,inplace=True)

        hourly_PnL_df = pd.DataFrame(index=full_pred_df.index)

        for col in [col for col in full_pred_df.columns if ('DART' not in col) and ('SPREAD' not in col)]:
            if model_type == 'SPREAD':
                try:
                    pred_df = pd.DataFrame(full_pred_df[[col2 for col2 in full_pred_df.columns if ('SPREAD' in col2) and (col in col2)]])
                    act_df = pd.DataFrame(full_pred_df[col])
                    hourly_PnL_df[col] =  pred_df[pred_df.columns[0]]*act_df[act_df.columns[0]]
                except:
                    pass

            if model_type =='DART':
                try:
                    pred_df = pd.DataFrame(full_pred_df[[col2 for col2 in full_pred_df.columns if ('DART' in col2) and (col in col2)]])
                    act_df = pd.DataFrame(full_pred_df[col])
                    hourly_PnL_df[col] =  pred_df[pred_df.columns[0]]*act_df[act_df.columns[0]]
                except:
                    pass

        hourly_PnL_df.reset_index(inplace=True)
        hourly_PnL_df.set_index('Date',inplace=True)
        hourly_PnL_df.drop(columns='HourEnding',inplace=True)
        hourly_PnL_df[iso] = hourly_PnL_df.sum(axis=1)
        daily_PnL_df = hourly_PnL_df.groupby(pd.Grouper(freq='D')).sum()
        if iso_daily_PnL.empty:
            iso_daily_PnL = pd.DataFrame(daily_PnL_df[iso])
        else:
            iso_daily_PnL[iso] = daily_PnL_df[iso]

    iso_daily_PnL['Combined_Portfolio'] = iso_daily_PnL.sum(axis=1)

    var_dataframes_dict = {}

    iso_daily_PnL.index = pd.to_datetime(iso_daily_PnL.index)

    current_date = datetime.datetime.today()
    one_year = current_date - datetime.timedelta(1*365)
    two_year = current_date - datetime.timedelta(2 * 365)
    three_year = current_date - datetime.timedelta(3 * 365)
    start_dates = {current_date:'THREE_YEAR_90DAY_WINDOW',one_year:'ONE_YEAR',two_year:'TWO_YEAR',three_year:'THREE_YEAR'}

    for date,date_type in start_dates.items():

        if date == current_date:
            temp_iso_daily_PnL = pd.DataFrame()

            for prev_year in range(0, 4):
                start_date = current_date - datetime.timedelta(days=prev_year * 365+45)
                end_date = current_date - datetime.timedelta(days=prev_year * 365-45)
                temp_backtest_pnl_df = iso_daily_PnL.loc[start_date:end_date]
                if temp_iso_daily_PnL.empty:
                    temp_iso_daily_PnL = temp_backtest_pnl_df
                else:
                    temp_iso_daily_PnL = pd.concat([temp_iso_daily_PnL, temp_backtest_pnl_df], axis=0,sort=True)

        else:
            temp_iso_daily_PnL = iso_daily_PnL.loc[date:current_date]


        summary_df = pd.DataFrame(
            columns=['ISO', 'VAR98', 'CVAR98', 'MAX_LOSS', date_type, 'VAR02', 'CVAR02', 'MAX_GAIN'])

        for col in iso_daily_PnL.columns:
            max_loss = round(temp_iso_daily_PnL[col].min(),0)
            max_gain = round(temp_iso_daily_PnL[col].max(),0)

            var_98 = round(temp_iso_daily_PnL[col].quantile(0.02),0)
            cvar_98_dol = temp_iso_daily_PnL[iso_daily_PnL[col] < var_98][col].sum()
            cvar_98_samps = temp_iso_daily_PnL[temp_iso_daily_PnL[col] < var_98][col].count()
            cvar_98 = round(cvar_98_dol/cvar_98_samps,0)

            var_02 = round(temp_iso_daily_PnL[col].quantile(0.98),0)
            cvar_02_dol = temp_iso_daily_PnL[temp_iso_daily_PnL[col] > var_02][col].sum()
            cvar_02_samps = temp_iso_daily_PnL[temp_iso_daily_PnL[col] > var_02][col].count()
            cvar_02 = round(cvar_02_dol / cvar_02_samps,0)

            dict = {'ISO': col,
                    'VAR98': var_98,
                    'CVAR98': cvar_98,
                    'MAX_LOSS' : max_loss,
                    date_type: date_type,
                    'VAR02': var_02,
                    'CVAR02': cvar_02,
                    'MAX_GAIN': max_gain}

            summary_df = summary_df.append(dict, ignore_index=True)

        summary_df.set_index('ISO', inplace=True)
        var_dataframes_dict[date_type] = summary_df


    var_fig = print_var(var_dataframes_dict=var_dataframes_dict,
                        predict_date_str_mm_dd_yyyy=predict_date_str_mm_dd_yyyy,
                        name_adder=name_adder,
                        model_type=model_type)

    var_writer = pd.ExcelWriter(save_directory + 'Daily_VAR_Report_'+predict_date_str_mm_dd_yyyy+'_'+model_type+'_'+name_adder+'.xlsx', engine='openpyxl')

    for type, df in var_dataframes_dict.items():
        df.to_excel(var_writer, sheet_name=type, index=True)

    var_writer.save()
    var_writer.close()



    url = plotly.offline.plot(var_fig,
                              filename=save_directory + 'Daily_VAR_Report_' + predict_date_str_mm_dd_yyyy + '_'+model_type+'_'+name_adder+ '.html',
                              auto_open=True)
    return