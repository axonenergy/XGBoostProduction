from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import datetime
from ast import literal_eval
from XGBLib import save_obj
from XGBLib import load_obj
from XGBLib import create_features
from XGBLib import std_dev_outlier_remove
from XGBLib import read_clean_data
import shap as shap

static_directory = 'C:\\XGBoostProduction\\'
working_directory = 'X:\\Research\\'

# COMMON PARAMETERS
input_file_name = '2020_06_04_BACKTEST_DATA_DICT_MASTER'                        # Use This If Reading From Dictionary (New Method)
input_file_type = 'dict'                                                        # Use This If Reading From Dictionary
hypergrid_dict_name = 'RFGridsearchDict_12092019_MISOAll_Master_Dataset_Dict_'  # Name Of Hypergrid File
all_best_features_filename = 'FeatImport_2020_02_24_BACKTEST_DATA_DICT_MASTER_SPREAD_ONE_YEAR_SD6_PJM' # Name of Feature Importance File
name_adder = ''                                                                 # Additional Identifier For The Run
add_calculated_features = False                                                 # If True Adds Calcualted Features From A Previous Best Feature Importance Run. Will Error If Matching Non-Calculated Feature Importances Are Not Run First. Used to Determine If Calcualted Features Are Good Or Not
do_all_feats = False                                                             # dont segregate the features into feature types
sd_limit = 6                                                                    # SD Limit For Outlier Removal
gridsearch_iterations = 100                                                      # Gridsearch Iterations
cv_folds = 4                                                                    # CV Folds For Gridsearch


feat_dict = {'SPR_EAD': 2,'DA_RT': 2, 'FLOAD': 8, 'FTEMP': 24, 'OUTAGE': 4,'LMP': 4, 'GAS_PRICE': 4}               # Number Of Top Features To Use If Reading From Dict And Adding Calculated Features

train_end_date = datetime.datetime(int(input_file_name.split(sep='_')[0]),int(input_file_name.split(sep='_')[1]),int(input_file_name.split(sep='_')[2]))
vintage_dict = {'ONE_YEAR':train_end_date-datetime.timedelta(days=365*1), 'THREE_YEAR':train_end_date-datetime.timedelta(days=365*3), 'ALL_YEAR':train_end_date-datetime.timedelta(days=365*10)}

iso_list = ['ERCOT']
model_type = 'DART'
model_arch = 'RF' # options are RF and LGB
shapely = False

feat_types_list = ['SPR_EAD', 'DA_RT','LMP', 'FLOAD','FTEMP','OUTAGE','GAS_PRICE']                            # Feat Types To Run

hypergrids_from_file=False
run_gridsearch = False                                                          # Do A Gridsearch?
run_feature_importances = True                                                  # Do Feature Importances?


def do_gridsearch(input_filename, save_name, iso_list, feat_dict, fit_params_dict, feat_types_list, input_file_type, sd_limit, cv_folds, gridsearch_iterations, add_calculated_features, static_directory, working_directory,model_type,model_arch, verbose=True):
    # COORDINATES THE GRIDSEARCH(S)
    gridsearch_directory = static_directory + '\GridsearchFiles\\'
    model_data_directory = static_directory + '\ModelUpdateData\\'

    # Create Empty Dict to Store Hypergrids
    hypergrids = dict()

    # Define Target For Each ISO
    if model_type.upper() == 'DART':
        target_dict = {'ISONE':'ISONE_10033_DART',
        'SPP': 'SPP_AECC_CSWS_DART',
        'PJM': 'PJM_50390_DART',
        'MISO':'MISO_AECI.ALTW_DART'}
    elif model_type.upper() =='SPREAD':
        target_dict = {'ISONE':['ISONE_10033$ISONE_10037_SPREAD'],
        'ERCOT':'ERCOT_HB_HOUSTON$ERCOT_DC_R_SPREAD',
        'SPP': 'SPP_AECC_CSWS$SPP_CSWSLEEPINGBEAR_SPREAD',
        'PJM': 'PJM_1069452904$PJM_1124361945_SPREAD',
        'MISO':'MISO_AECI.ALTW$MISO_AECI.AMMO_SPREAD'}

    # Train Gridsearches for Each ISO In The List
    for iso in iso_list:
        # Read In Input File
        master_df = read_clean_data(input_filename=model_data_directory+input_filename,
                                    input_file_type=input_file_type,
                                    iso=iso)

        target = target_dict[iso]
        print('Gridsearch Target: '+target)

        # Create Calculated Features (if reading from dict) Or Remove All Non-Target DARTs from CSV
        if input_file_type.upper() == 'DICT':
            if add_calculated_features:
                print('Creating Features...')
                feature_df = create_features(input_df=master_df,
                                             feat_dict=feat_dict,
                                             target_name=target,
                                             iso=iso.upper())
            else: feature_df = master_df

        elif input_file_type.upper() == 'CSV':
            target_col = master_df[target]
            feature_df = master_df[[col for col in master_df if 'DART' not in col]]
            feature_df = pd.concat([feature_df,target_col],axis=1)

        # Train Gridsearches for Each Feature Type In The List
        for feat_type in feat_types_list:

            if model_arch == 'RF':
                hypergrid = rf_gridsearch(train_df=feature_df,
                                          feat_type=feat_type,
                                          target=target,
                                          cv_folds=cv_folds,
                                          gridsearch_iterations=gridsearch_iterations,
                                          sd_limit=sd_limit,
                                          fit_params=fit_params_dict[feat_type])
            elif model_arch=='LGB':
                hypergrid = lgb_gridsearch(train_df=feature_df,
                                          feat_type=feat_type,
                                          target=target,
                                          cv_folds=cv_folds,
                                          gridsearch_iterations=gridsearch_iterations,
                                          sd_limit=sd_limit,
                                          fit_params=fit_params_dict[feat_type])

            hypergrid.to_csv(gridsearch_directory+'RFGridsearch_' + save_name + '_' + iso + '_' + feat_type + '.csv', index=False)
            # hypergrids.update({iso+'_'+feat_type: hypergrid})
            # save_obj(hypergrids, gridsearch_directory+'RFGridsearchDict_' + save_name)

    return hypergrids

def rf_gridsearch(train_df, feat_type, target, cv_folds, gridsearch_iterations, sd_limit, fit_params):
    # Remove Outliers From Train Set and Eval Set
    train_df = std_dev_outlier_remove(input_df=train_df,
                                      target=target,
                                      sd_limit=sd_limit,
                                      verbose=True)

    # Prepare X_train and Y_train
    if feat_type == 'OUTAGE':
        x_train_df = train_df[[col for col in train_df.columns if (('DA_RT' not in col)&('FLOAD' not in col )&('FTEMP' not in col)&('DART' not in col)&('SPREAD' not in col)&('SPR_EAD' not in col))]]
    else:
        x_train_df = train_df[[col for col in train_df.columns if feat_type in col]]
    y_train_df = pd.DataFrame(train_df[target])
    print('Training Gridsearch For Feature: ' + feat_type)
    print('Num Features: '+str(len(x_train_df.columns)))
    print(x_train_df.columns)

    # Build and Fit Model
    model = RandomForestRegressor()
    skf = GroupKFold(n_splits=cv_folds)

    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=fit_params,
                                       n_iter=gridsearch_iterations,
                                       cv=skf.split(x_train_df, y_train_df, groups=x_train_df.index.get_level_values('Date')),
                                       verbose=3,
                                       n_jobs=-1,
                                       scoring='neg_mean_squared_error')

    random_search.fit(x_train_df, y_train_df.values.ravel())
    results = pd.DataFrame(random_search.cv_results_)
    results = results.sort_values(by='rank_test_score', ascending=True)
    print('Best RF Hyper Params Target:: ' + target + ' FeatType: ' + feat_type + ' :\n', results)

    return results

def lgb_gridsearch(train_df, feat_type, target, cv_folds, gridsearch_iterations, sd_limit, fit_params):
    # Remove Outliers From Train Set and Eval Set
    train_df = std_dev_outlier_remove(input_df=train_df,
                                      target=target,
                                      sd_limit=sd_limit,
                                      verbose=True)

    # Prepare X_train and Y_train
    if feat_type == 'OUTAGE':
        x_train_df = train_df[[col for col in train_df.columns if (('DA_RT' not in col)&('FLOAD' not in col )&('FTEMP' not in col)&('DART' not in col)&('SPREAD' not in col)&('SPR_EAD' not in col))]]
    else:
        x_train_df = train_df[[col for col in train_df.columns if feat_type in col]]

    y_train_df = pd.DataFrame(train_df[target])
    print('Training Gridsearch For Feature: ' + feat_type)
    print('Num Features: '+str(len(x_train_df.columns)))
    print(x_train_df.columns)

    # Build and Fit Model
    model = lgb.LGBMRegressor(objective='rmse',
                              boosting_type = 'gbdt',
                             n_estimators=5000,
                             device='cpu')

    skf = GroupKFold(n_splits=cv_folds)

    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=fit_params,
                                       n_iter=gridsearch_iterations,
                                       cv=skf.split(x_train_df, y_train_df, groups=x_train_df.index.get_level_values('Date')),
                                       verbose=3,
                                       scoring='neg_mean_squared_error',
                                       n_jobs=-2)

    random_search.fit(x_train_df, y_train_df.values.ravel())
    results = pd.DataFrame(random_search.cv_results_)
    results = results.sort_values(by='rank_test_score', ascending=True)
    print('Best RF Hyper Params Target:: ' + target + ' FeatType: ' + feat_type + ' :\n', results)

    return results

def do_top_features(input_filename, save_name, iso_list, feat_dict, hypergrid_dict_name,shapely, feat_types_list, input_file_type, sd_limit, hypergrids_from_file,train_end_date, add_calculated_features, static_directory,working_directory,vintage_dict, model_type,model_arch):
    feature_importance_directory = static_directory + '\FeatureImportanceFiles\\'
    model_data_directory = static_directory + '\ModelUpdateData\\'
    gridsearch_directory = static_directory + '\GridsearchFiles\\'

    # Create Empty Dict to Store Feature Importances


    for iso in iso_list:

        importances_df = pd.DataFrame(index=range(0, 1000, 1))

        # Read In Input File
        master_df = read_clean_data(input_filename=model_data_directory+input_filename,
                                    input_file_type=input_file_type,
                                    iso=iso)


        targets_df = master_df[[col for col in master_df.columns if model_type in col]]

        targets_df = targets_df[[col for col in targets_df.columns if iso in col]]

        tot_feats = len(feat_types_list) * len(targets_df.columns) * len(vintage_dict)

        counter = 0

        for target in targets_df.columns:

            # Create Calculated Features (if reading from dict) Or Remove All Non-Target DARTs from CSV
            if input_file_type.upper() == 'DICT':
                if add_calculated_features:

                    all_best_features_df = pd.read_csv(feature_importance_directory+all_best_features_filename + ".csv", dtype=np.str)

                    cat_vars = ['Month', 'Weekday']

                    feature_df = create_features(input_df=master_df,
                                                 feat_dict=feat_dict,
                                                 target_name=target,
                                                 iso=iso.upper(),
                                                 cat_vars = cat_vars,
                                                 static_directory=static_directory,
                                                 all_best_features_df=all_best_features_df)
                else: feature_df=master_df

            elif input_file_type.upper() == 'CSV':
                target_col = targets_df[target]
                feature_df = targets_df[[col for col in targets_df if 'DART' not in col]]
                feature_df = pd.concat([feature_df,target_col],axis=1)

            # Remove Outliers
            feature_df = std_dev_outlier_remove(input_df=feature_df,
                                                target=target,
                                                sd_limit=sd_limit,
                                                verbose=False)

            # Iterate Through All Feature Types and get best features for each
            for feat_type in feat_types_list:

                vintage_df = pd.DataFrame()

                for vintage_string, train_start_date in vintage_dict.items():

                    input_df = feature_df[feature_df.index.get_level_values('Date') < train_end_date]
                    input_df = input_df[input_df.index.get_level_values('Date') > train_start_date]

                    #Use generic hypergrid from this file (below) to run
                    params = param_grid_backtest

                    # # Read Hypergrid From file (Used If Non-Top Params Are Much Faster But Not More Accurage
                    if hypergrids_from_file==True:
                        hypergrid_df = pd.read_csv(gridsearch_directory+'RFGridsearch_2020_05_04_BACKTEST_DATA_DICT_MASTER__DART_LGB_PJM_'+feat_type+'.csv')
                        params = hypergrid_df['params'].iloc[0]
                        params = literal_eval(params)

                    #Not lagged - need to remove
                    input_df = input_df.drop(columns=[col for col in input_df.columns if 'RTLMP' in col])
                    input_df = input_df.drop(columns=[col for col in input_df.columns if 'DALMP' in col])

                    ### Split into training and eval sets
                    exp_cv = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=1337)

                    for train_index, eval_index in exp_cv.split(input_df,groups=input_df.index.get_level_values('Date')):
                        train_df, eval_df = input_df.iloc[train_index], input_df.iloc[eval_index]

                    if do_all_feats:
                        x_train_df = train_df[[col for col in train_df.columns if (('DART' not in col)&('SPREAD'not in col))]]
                        x_eval_df = eval_df[[col for col in eval_df.columns if (('DART' not in col) & ('SPREAD' not in col))]]
                        # x_train_df = pd.get_dummies(x_train_df,columns=['Month','Weekday'])
                    elif feat_type == 'OUTAGE':
                        x_train_df = train_df[[col for col in train_df.columns if(('DA_RT' not in col) & ('FLOAD' not in col) & ('FTEMP' not in col) & ('DART' not in col) & ('Weekday' not in col) & ('Month' not in col) & ('HourEnding' not in col)&('SPREAD' not in col)&('SPR_EAD' not in col)&('DA_LMP' not in col)&('RT_LMP' not in col)&('GAS_PRICE' not in col))]]
                        x_eval_df = eval_df[[col for col in eval_df.columns if (('DA_RT' not in col) & ('FLOAD' not in col) & ('FTEMP' not in col) & ('DART' not in col) & ('Weekday' not in col) & ('Month' not in col) & ('HourEnding' not in col) & ('SPREAD' not in col) & ('SPR_EAD' not in col) & ('DA_LMP' not in col) & ('RT_LMP' not in col)&('GAS_PRICE' not in col))]]
                    else:
                        x_train_df = train_df[[col for col in train_df.columns if feat_type in col]]
                        x_eval_df = eval_df[[col for col in eval_df.columns if feat_type in col]]

                    y_train_df = train_df[target]
                    y_eval_df = eval_df[target]

                    feat_names = x_train_df.columns

                    # Train Feature importances for Each Feature Type In The List

                    ## RandomForest Feature Importances
                    if model_arch=='RF':
                        rf = RandomForestRegressor(**params, n_jobs=-1)

                        print('Creating RF ' + vintage_string+ ' Importances For Target: ' +target+'  |  ISO: '+iso+'  |  FeatType: '+feat_type+'  |  NumFeats: ' +str(len(x_train_df.columns)) + '  |  % Complete:'+  str(round(counter/tot_feats*100,2)))

                        rf.fit(x_train_df, y_train_df.values.ravel())
                        counter +=1

                        if vintage_df.empty:
                            vintage_df = pd.DataFrame(rf.feature_importances_, index=feat_names)
                            vintage_df.columns = [vintage_string]
                        else:
                            temp_df = pd.DataFrame(rf.feature_importances_, index=feat_names)
                            temp_df.columns = [vintage_string]
                            vintage_df = pd.concat([vintage_df, temp_df], axis=1)

                        if shapely == True:
                            explainer = shap.TreeExplainer(rf)
                            shap_values = explainer.shap_values(x_train_df)
                            shap_df = pd.DataFrame(shap_values)
                            shap_df.to_csv('shapoutput.csv')
                            shap.summary_plot(shap_values, x_train_df)


                    ### LightGBM feature importances
                    if model_arch == 'LGB':

                        print('Creating LGB ' + vintage_string + ' Importances For Target: ' + target + '  |  ISO: ' + iso + '  |  FeatType: ' + feat_type + '  |  NumFeats: ' + str(len(x_train_df.columns)) + '  |  % Complete:' + str(round(counter / tot_feats * 100, 2)))

                        dtrain = lgb.Dataset(data=x_train_df, label=y_train_df)
                        deval = lgb.Dataset(data=x_eval_df, label=y_eval_df)
                        watchlist = [deval]

                        params['boosting'] = 'gbdt'
                        params['objective'] = 'rmse'
                        params['verbose'] = '-1'
                        early_stopping = 10
                        nrounds = 5000
                        eval_dict = dict()

                        lbm = lgb.train(params=params,
                                        train_set=dtrain,
                                        num_boost_round=nrounds,
                                        valid_sets=watchlist,
                                        early_stopping_rounds=early_stopping,
                                        verbose_eval=False,
                                        evals_result=eval_dict
                                        )

                        counter += 1

                        if vintage_df.empty:
                            vintage_df = pd.DataFrame(lbm.feature_importance(), index=feat_names)
                            vintage_df.columns = [vintage_string]
                        else:
                            temp_df = pd.DataFrame(lbm.feature_importance(), index=feat_names)
                            temp_df.columns = [vintage_string]
                            vintage_df = pd.concat([vintage_df, temp_df], axis=1)

                        if shapely==True:
                            explainer = shap.TreeExplainer(lbm)
                            shap_values = explainer.shap_values(x_train_df)
                            shap_df = pd.DataFrame(shap_values)
                            shap_df.to_csv('shapoutput.csv')
                            shap.summary_plot(shap_values, x_train_df)


                vintage_df['Average'] = vintage_df.mean(axis=1)

                importances = sorted(zip(map(lambda x: round(x, 4), vintage_df['Average']), vintage_df.index), reverse=True)
                importances_df[iso+'_'+feat_type+'_'+target] = pd.Series(importances, )
                importances_df.to_csv(feature_importance_directory+'FeatImport_' + save_name + '_SD'+ str(sd_limit)+'_'+iso + '.csv',index=False)



######################################################################################################
#                                           Gridsearch                                               #
######################################################################################################


#####FOR RANDOM FOREST
param_grid_backtest = {'bootstrap': True,
                  'max_depth': 30,
                  'max_features': 'auto',
                  'min_samples_leaf': 2,
                  'min_samples_split': 2,
                  'n_estimators': 100}


#####FOR LIGHT GBM
# param_grid_backtest = {'learning_rate': [0.0025],  # no greater than 0.01, doesnt affect fit time below that, sometimes gets slightly better results below that
#                       'num_leaves': [25],  # SLIGHT better RMSE as more leaves used, increases training time substantially
#                       'max_bin': [40],  # Impacts RMSE a bit from 16-100 range, doesnt impact training time
#                       'colsample_bytree': [0.2]}  # Impacts RMSE and training time substantially. Training time goes up as colsample goes up


param_grid_DART = {'learning_rate': [0.0005],  # no greater than 0.01, doesnt affect fit time below that, sometimes gets slightly better results below that
                      'num_leaves': [10,15],  # SLIGHT better RMSE as more leaves used, increases training time substantially
                      'max_bin': [60,70,80],  # Impacts RMSE a bit from 16-100 range, doesnt impact training time
                      'colsample_bytree': [0.05]}  # Impacts RMSE and training time substantially. Training time goes up as colsample goes up

param_grid_OUTAGE = {'learning_rate': [0.005],  # no greater than 0.01, doesnt affect fit time below that, sometimes gets slightly better results below that
                      'num_leaves': [25],  # SLIGHT better RMSE as more leaves used, increases training time substantially
                      'max_bin': [60,70,80],  # Impacts RMSE a bit from 16-100 range, doesnt impact training time
                      'colsample_bytree': [0.3]}  # Impacts RMSE and training time substantially. Training time goes up as colsample goes up

param_grid_FLOAD = {'learning_rate': [0.001],  # no greater than 0.01, doesnt affect fit time below that, sometimes gets slightly better results below that
                      'num_leaves': [15],  # SLIGHT better RMSE as more leaves used, increases training time substantially
                      'max_bin': [40,50],  # Impacts RMSE a bit from 16-100 range, doesnt impact training time
                      'colsample_bytree': [0.2]}  # Impacts RMSE and training time substantially. Training time goes up as colsample goes up

param_grid_FTEMP = {'learning_rate': [0.001],  # no greater than 0.01, doesnt affect fit time below that, sometimes gets slightly better results below that
                      'num_leaves': [40,45,50],  # SLIGHT better RMSE as more leaves used, increases training time substantially
                      'max_bin': [40],  # Impacts RMSE a bit from 16-100 range, doesnt impact training time
                      'colsample_bytree': [0.2]}  # Impacts RMSE and training time substantially. Training time goes up as colsample goes up

param_grid_SPR_EAD = {'learning_rate': [0.001],  # no greater than 0.01, doesnt affect fit time below that, sometimes gets slightly better results below that
                      'num_leaves': [25],  # SLIGHT better RMSE as more leaves used, increases training time substantially
                      'max_bin': [60],  # Impacts RMSE a bit from 16-100 range, doesnt impact training time
                      'colsample_bytree': [0.05]}  # Impacts RMSE and training time substantially. Training time goes up as colsample goes up

param_grid_LMP = {'learning_rate': [0.01],  # no greater than 0.01, doesnt affect fit time below that, sometimes gets slightly better results below that
                      'num_leaves': [30],  # SLIGHT better RMSE as more leaves used, increases training time substantially
                      'max_bin': [60,70,80],  # Impacts RMSE a bit from 16-100 range, doesnt impact training time
                      'colsample_bytree': [0.35,0.4,0.45]}  # Impacts RMSE and training time substantially. Training time goes up as colsample goes up

param_grid_GAS_PRICE = {'learning_rate': [0.0005],  # no greater than 0.01, doesnt affect fit time below that, sometimes gets slightly better results below that
                      'num_leaves': [20,25,30],  # SLIGHT better RMSE as more leaves used, increases training time substantially
                      'max_bin': [40],  # Impacts RMSE a bit from 16-100 range, doesnt impact training time
                      'colsample_bytree': [0.2]}  # Impacts RMSE and training time substantially. Training time goes up as colsample goes up



fit_params_dict = {'FLOAD':param_grid_FLOAD,
                   'DA_RT':param_grid_DART,
                   'FTEMP':param_grid_FTEMP,
                   'OUTAGE':param_grid_OUTAGE,
                   'SPR_EAD': param_grid_SPR_EAD,
                   'LMP': param_grid_LMP,
                   'GAS_PRICE': param_grid_GAS_PRICE
                   }

save_name = input_file_name+'_'+name_adder+'_'+model_type+'_'+model_arch

if run_gridsearch:
    do_gridsearch(input_filename=input_file_name,
                     save_name=save_name,
                     iso_list=iso_list,
                     feat_dict= feat_dict,
                     fit_params_dict=fit_params_dict,
                     feat_types_list=feat_types_list,
                     input_file_type=input_file_type,
                     sd_limit=sd_limit,
                     cv_folds=cv_folds,
                     shapely=shapely,
                     model_arch=model_arch,
                     model_type = model_type,
                     gridsearch_iterations = gridsearch_iterations,
                     add_calculated_features=add_calculated_features,
                     static_directory=static_directory,
                     working_directory=working_directory)

if run_feature_importances:
    do_top_features(input_filename=input_file_name,
                    save_name=save_name,
                    iso_list=iso_list,
                    feat_dict=feat_dict,
                    hypergrid_dict_name=hypergrid_dict_name,
                    feat_types_list=feat_types_list,
                    input_file_type=input_file_type,
                    model_arch=model_arch,
                    hypergrids_from_file=hypergrids_from_file,
                    sd_limit=sd_limit,
                    train_end_date=train_end_date,
                    add_calculated_features=add_calculated_features,
                    static_directory=static_directory,
                    working_directory=working_directory,
                    vintage_dict=vintage_dict,
                    model_type=model_type)


