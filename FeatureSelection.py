from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold
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

static_directory = 'C:\\XGBoostProduction\\'
working_directory = 'X:\\Research\\'

# COMMON PARAMETERS
# input_file_name = '09_11_2019_GBM_DATA_MISO_V8.0_MASTER_159F'                 # Use This If Reading From CSV (Old Method)
# input_file_type = 'csv'                                                       # Use This If Reading From CSV (Old Method)
input_file_name = '2020_02_24_BACKTEST_DATA_DICT_MASTER_SPREAD'                        # Use This If Reading From Dictionary (New Method)
input_file_type = 'dict'                                                        # Use This If Reading From Dictionary
hypergrid_dict_name = 'RFGridsearchDict_12092019_MISOAll_Master_Dataset_Dict_'  # Name Of Hypergrid File
all_best_features_filename = 'FeatImport12092019_Master_Nodes_Dataset_Dict_full_SD6_ALL' # Name of Feature Importance File
name_adder = ''                                                                 # Additional Identifier For The Run
add_calculated_features = False                                                 # If True Adds Calcualted Features From A Previous Best Feature Importance Run. Will Error If Matching Non-Calculated Feature Importances Are Not Run First. Used to Determine If Calcualted Features Are Good Or Not
do_all_feats = False                                                             # dont segregate the features into feature types
sd_limit = 6                                                                    # SD Limit For Outlier Removal
gridsearch_iterations = 100                                                      # Gridsearch Iterations
cv_folds = 4                                                                    # CV Folds For Gridsearch
train_end_date = datetime.datetime(2020, 8, 24)                                 # Backtest Start Date (If Not Doing Cross Validation)
feat_dict = {'DA_RT': 6, 'FLOAD': 8, 'FTEMP': 20, 'OUTAGE': 12}               # Number Of Top Features To Use If Reading From Dict And Adding Calculated Features
iso_list = ['PJM']
feat_types_list = ['SPR_EAD','DA_RT','FLOAD','FTEMP','OUTAGE']                            # Feat Types To Run
run_gridsearch = False                                                          # Do A Gridsearch?
run_feature_importances = True                                                  # Do Feature Importances?


def do_rf_gridsearch(input_filename, save_name, iso_list, feat_dict, fit_params_dict, feat_types_list, input_file_type, sd_limit, cv_folds, gridsearch_iterations, add_calculated_features, static_directory, working_directory, verbose=True):
    # COORDINATES THE GRIDSEARCH(S)
    gridsearch_directory = static_directory + '\GridsearchFiles\\'
    model_data_directory = working_directory + '\ModelUpdateData\\'

    # Create Empty Dict to Store Hypergrids
    hypergrids = dict()

    # Define Target For Each ISO
    target_dict = {'PJM': 'PJM_50390_DART',
                   'MISO': 'MISO_AECI.ALTW_DART',
                   'NEISO': 'ISONE_10033_DART',
                   'NYISO': 'NYISO_61752_DART',
                   'ERCOT': 'ERCOT_AMISTAD_ALL_DART',
                   'SPP': 'SPP_AECC_CSWS_DART'}

    # Train Gridsearches for Each ISO In The List
    for iso in iso_list:
        # Read In Input File
        master_df = read_clean_data(input_filename=model_data_directory+input_filename,
                                    input_file_type=input_file_type,
                                    iso=iso)

        target = target_dict[iso]
        print('RF Gridsearch Target: '+target)

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

            hypergrid = rf_gridsearch(train_df=feature_df,
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

def do_top_features(input_filename, save_name, iso_list, feat_dict, hypergrid_dict_name, feat_types_list, input_file_type, sd_limit, train_end_date, add_calculated_features, static_directory,working_directory):
    feature_importance_directory = working_directory + '\FeatureImportanceFiles\\'
    model_data_directory = static_directory + '\ModelUpdateData\\'
    gridsearch_directory = working_directory + '\GridsearchFiles\\'

    # Create Empty Dict to Store Feature Importances


    for iso in iso_list:
        importances_df = pd.DataFrame(index=range(0, 500, 1))

        # Read In Input File
        master_df = read_clean_data(input_filename=model_data_directory+input_filename,
                                    input_file_type=input_file_type,
                                    iso=iso)

        train_df = master_df[master_df.index.get_level_values('Date') < train_end_date]

        targets_df = train_df[[col for col in train_df.columns if 'DART' in col]]
        targets_df = train_df[[col for col in targets_df.columns if iso in col]]

        for target in targets_df.columns:

            # Create Calculated Features (if reading from dict) Or Remove All Non-Target DARTs from CSV
            if input_file_type.upper() == 'DICT':
                if add_calculated_features:

                    all_best_features_df = pd.read_csv(all_best_features_filename + ".csv", dtype=np.str)

                    feature_df = create_features(input_df=train_df,
                                                 feat_dict=feat_dict,
                                                 target_name=target,
                                                 iso=iso.upper(),
                                                 all_best_features_df=all_best_features_df)
                else: feature_df=train_df

            elif input_file_type.upper() == 'CSV':
                target_col = train_df[target]
                feature_df = train_df[[col for col in train_df if 'DART' not in col]]
                feature_df = pd.concat([feature_df,target_col],axis=1)

            # Remove Outliers
            feature_df = std_dev_outlier_remove(input_df=feature_df,
                                                target=target,
                                                sd_limit=sd_limit,
                                                verbose=False)

            # Iterate Through All Feature Types and get best features for each
            for feat_type in feat_types_list:

                #Use generic hypergrid from this file (below) to run
                params = param_grid_backtest

                # # Read Hypergrid From Dictionary
                # hypergrid_dict = load_obj(gridsearch_directory+hypergrid_dict_name)
                # hypergrid_df = hypergrid_dict[iso+'_'+feat_type]
                # params = hypergrid_df['params'].iloc[0]

                # # Read Hypergrid From file (Used If Non-Top Params Are Much Faster But Not More Accurage
                # hypergrid_df = pd.read_csv(gridsearch_directory+'RFGridsearch_12092019_MISOAll_Master_Dataset_Dict__MISO_'+feat_type+'.csv')
                # params = hypergrid_df['params'].iloc[0]
                # params = literal_eval(params)

                if do_all_feats:
                    x_train_df = feature_df[[col for col in feature_df.columns if (('DART' not in col)&('SPREAD'not in col))]]
                    # x_train_df = pd.get_dummies(x_train_df,columns=['Month','Weekday'])
                elif feat_type == 'OUTAGE':
                    x_train_df = feature_df[[col for col in feature_df.columns if(('DA_RT' not in col) & ('FLOAD' not in col) & ('FTEMP' not in col) & ('DART' not in col) & ('Weekday' not in col) & ('Month' not in col) & ('HourEnding' not in col)&('SPREAD' not in col)&('SPR_EAD' not in col))]]
                else:
                    x_train_df = feature_df[[col for col in feature_df.columns if feat_type in col]]

                y_train_df = feature_df[target]
                feat_names = x_train_df.columns

                # Train Feature importances for Each Feature Type In The List
                rf = RandomForestRegressor(**params, n_jobs=-1)

                print('Creating Importances For Target: ' +target+' ISO: '+iso+' FeatType: '+feat_type+'  NumFeats: ' +str(len(x_train_df.columns)))

                rf.fit(x_train_df, y_train_df.values.ravel())

                importances = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feat_names), reverse=True)
                importances_df[iso+'_'+feat_type+'_'+target] = pd.Series(importances)
                importances_df.to_csv(feature_importance_directory+'FeatImport_' + save_name + '_SD'+ str(sd_limit)+'_'+iso+'.csv',index=False)



######################################################################################################
#                                           Gridsearch                                               #
######################################################################################################

param_grid_backtest = {'bootstrap': True,
                  'max_depth': 30,
                  'max_features': 'auto',
                  'min_samples_leaf': 2,
                  'min_samples_split': 2,
                  'n_estimators': 100}


param_grid_DART = {'bootstrap': [True],  # , False],
                  'max_depth': [30],
                  'max_features': ['auto'],
                  'min_samples_leaf': [2],
                  'min_samples_split': [2],
                  'n_estimators': list(range(100, 100, 40))}

param_grid_OUTAGE = {'bootstrap': [True],  # , False],
                  'max_depth': [30],
                  'max_features': ['auto'],
                  'min_samples_leaf': [2],
                  'min_samples_split': [2],
                  'n_estimators': list(range(100, 100, 40))}

param_grid_FLOAD = {'bootstrap': [True],  # , False],
                  'max_depth': [30],
                  'max_features': ['auto'],
                  'min_samples_leaf': [2],
                  'min_samples_split': [2],
                  'n_estimators': list(range(100, 100, 40))}

param_grid_FTEMP = {'bootstrap': [True],  # , False],
                  'max_depth': [30],
                  'max_features': ['auto'],
                  'min_samples_leaf': [2],
                  'min_samples_split': [2],
                  'n_estimators': list(range(100, 100, 40))}

fit_params_dict = {'FLOAD':param_grid_FLOAD,
                   'DA_RT':param_grid_DART,
                   'FTEMP':param_grid_FTEMP,
                   'OUTAGE':param_grid_OUTAGE}

save_name = input_file_name+'_'+name_adder

if run_gridsearch:
    do_rf_gridsearch(input_filename=input_file_name,
                     save_name=save_name,
                     iso_list=iso_list,
                     feat_dict= feat_dict,
                     fit_params_dict=fit_params_dict,
                     feat_types_list=feat_types_list,
                     input_file_type=input_file_type,
                     sd_limit=sd_limit,
                     cv_folds=cv_folds,
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
                    sd_limit=sd_limit,
                    train_end_date=train_end_date,
                    add_calculated_features=add_calculated_features,
                    static_directory=static_directory,
                    working_directory=working_directory)


