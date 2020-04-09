import pandas as pd
import datetime
import random
import os
import re
from ast import literal_eval
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
import numpy as np

from XGBLib import save_obj
from XGBLib import load_obj
from XGBLib import create_features
from XGBLib import create_tier2_features
from XGBLib import xgb_gridsearch
from XGBLib import xgb_train
from XGBLib import std_dev_outlier_remove
from XGBLib import read_clean_data


pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 5000)
pd.set_option('max_row', 10)

static_directory = 'C:\\XGBoostProduction\\'
working_directory = 'X:\\Research\\'
# working_directory = 'C:\\XGBoostProduction\\'

# COMMON PARAMETERS
# input_file_name = '09_11_2019_GBM_DATA_MISO_V8.0_MASTER_159F'      # Use This If Reading From CSV (Old Method)
# input_file_name = '09_11_2019_GBM_DATA_PJM_V8.0_MASTER_207F'      # Use This If Reading From CSV (Old Method)
# input_file_type = 'csv'                                            # Use This If Reading From CSV (Old Method)
input_file_name = '2020_03_19_BACKTEST_DATA_DICT_MASTER'               # Use This If Reading From Dictionary (New Method)
input_file_type = 'dict'                                             # Use This If Reading From Dictionary (New Method)
cat_vars = ['Month','Weekday']                                       # Categorial Variables
iso = 'ERCOT'                                                          # ISO to Backtest


all_best_features_filename = 'FeatImport_2020_03_19_BACKTEST_DATA_DICT_MASTER__SD6_ALL'  # Name of Feature Importance File
all_best_features_filename = 'FeatImport_2020_03_19_BACKTEST_DATA_DICT_MASTER__SPREAD_SD6_ERCOT'  # Name of Feature Importance File

name_adder = ''                                                        # Additional Identifier For The Run


run_reverse = False
model_type = 'SPREAD'  # Options are DART or SPREAD
run_gridsearch = False                                                # Do A Gridsearch?
run_backtest = True                                                    # Do A Backtest?
run_create_models = False

run_tier2_backtest =False
run_tier2_gridsearch = False
run_tier2_create_models = False

# PARAMETERS FOR BACKTEST
sd_limit_range = [1000]                                       # Range Of Max SDs For Outlier Processing (Large Number = No Outlier Processing)
model_creation_sd = 1000
backtest_start_date = datetime.datetime(2018, 8, 24)              # Backtest Start Date (If Not Doing Cross Validation)
num_targets = 250                                                # Number of Targets To Train (Large Number = Train All Targets In File)
nrounds = 5000                                                    # Max Rounds To Train
early_stopping=10                                                 # Early Train Stopping
exp_folds = 20                                                     # Number of Exps To Do For Each Senario (To Take Median And SD Of)
cv_folds = 4                                                      # Number of Folds If Doing CrossValidated Full Backtest (1 Or Less = No CV, Only Run Most Recent Year)
num_top_grids = 1                                                 # Number of Random Parameter Sets From Gridsearch To Select From (1 = Only Use Top Grid)
gpu_train = False                                                  # Train Using GPU (Default is CPU)

# PARAMETERS FOR GRIDSEARCH
gridsearch_iterations = 100                                                   # Number Of Gridsearch Iterations To Do
gridsearch_sd_limit = 1000                                                       # SD Limit to Use In Gridsearch
gridsearch_nrounds = 5000                                                     # Max Rounds To Train
gridsearch_gpu_train = False                                                   # Train Using GPU (Default is CPU)
gridsearch_cv_folds = 4                                                       # Number of Folds If Doing CrossValidated Full Backtest (1 Or Less = No CV, Only Run Most Recent Year)

hypergrid_dict_name=''

if iso == 'MISO':
    tier2_backtest_filename = 'Backtest_Exps_2020_02_24_BACKTEST_DATA_DICT_MASTER_MISO_EXP20_'
    tier2_PnL_filename = 'PnL_Results_Backtest_2020_02_24_BACKTEST_DATA_DICT_MASTER_MISO_EXP20__SD10001.25_notcapped_'
    tier2_hypergrid_name = 'Gridsearch_Tier2__Backtest_Exps_2020_02_24_BACKTEST_DATA_DICT_MASTER_PJM_EXP20__PJM_1048034_DART_Tier2Target'
    feat_dict = {'SPR_EAD': 6, 'DA_RT': 6, 'FLOAD': 8, 'FTEMP': 28,'OUTAGE': 8}  # Number Of Top Features To Use If Reading From Dict
    gridsearch_feat_dict=feat_dict
elif iso == 'PJM':
    tier2_backtest_filename = 'Backtest_Exps_2020_02_24_BACKTEST_DATA_DICT_MASTER_PJM_EXP20_'
    tier2_PnL_filename = 'PnL_Results_Backtest_2020_02_24_BACKTEST_DATA_DICT_MASTER_PJM_EXP20__SD10000.75_notcapped_'
    tier2_hypergrid_name = 'Gridsearch_Tier2__Backtest_Exps_2020_02_24_BACKTEST_DATA_DICT_MASTER_PJM_EXP20__PJM_1048034_DART_Tier2Target'
    feat_dict = {'SPR_EAD': 2, 'DA_RT': 2, 'FLOAD': 8, 'FTEMP': 24,'OUTAGE': 4}  # Number Of Top Features To Use If Reading From Dict
    gridsearch_feat_dict=feat_dict
elif iso == 'SPP':
    tier2_backtest_filename = 'Backtest_Exps_2020_02_24_BACKTEST_DATA_DICT_MASTER_SPP_EXP20_'
    tier2_PnL_filename = 'PnL_Results_Backtest_2020_02_24_BACKTEST_DATA_DICT_MASTER_SPP_EXP20__SD10001.0_notcapped_'
    tier2_hypergrid_name = 'Gridsearch_Tier2__Backtest_Exps_2020_02_24_BACKTEST_DATA_DICT_MASTER_PJM_EXP20__PJM_1048034_DART_Tier2Target'
    feat_dict = {'SPR_EAD': 2, 'DA_RT': 2, 'FLOAD': 10, 'FTEMP': 28,'OUTAGE': 10}  # Number Of Top Features To Use If Reading From Dict
    gridsearch_feat_dict=feat_dict
elif iso == 'ERCOT':
    tier2_backtest_filename = 'Backtest_Exps_2020_02_24_BACKTEST_DATA_DICT_MASTER_ERCOT_EXP20_'
    tier2_PnL_filename = 'PnL_Results_Backtest_2020_02_24_BACKTEST_DATA_DICT_MASTER_ERCOT_EXP20__SD10001.0_notcapped_'
    tier2_hypergrid_name = 'Gridsearch_Tier2__Backtest_Exps_2020_02_24_BACKTEST_DATA_DICT_MASTER_PJM_EXP20__PJM_1048034_DART_Tier2Target'
    feat_dict = {'SPR_EAD': 4, 'DA_RT': 4, 'FLOAD': 8, 'FTEMP': 24,'OUTAGE': 4}  # Number Of Top Features To Use If Reading From Dict
    gridsearch_feat_dict=feat_dict
elif iso == 'ISONE':
    tier2_backtest_filename = 'Backtest_Exps_2020_02_24_BACKTEST_DATA_DICT_MASTER_ISONE_EXP20_'
    tier2_PnL_filename = 'PnL_Results_Backtest_2020_02_24_BACKTEST_DATA_DICT_MASTER_ISONE_EXP20__SD10001.25_notcapped_'
    tier2_hypergrid_name = 'Gridsearch_Tier2__Backtest_Exps_2020_02_24_BACKTEST_DATA_DICT_MASTER_PJM_EXP20__PJM_1048034_DART_Tier2Target'
    feat_dict = {'SPR_EAD': 2, 'DA_RT': 2, 'FLOAD': 10, 'FTEMP': 16,'OUTAGE': 8}  # Number Of Top Features To Use If Reading From Dict
    gridsearch_feat_dict=feat_dict


tier2_cat_vars = []
tier2_dart_sd_location_filter = 'SD1000'
tier2_sd_limit_range = [1000]
tier2_model_creation_sd = 1000
tier2_exp_folds = 5
tier2_cv_folds = 4
tier2_num_grids = 1
tier2_gpu_train = True
tier2_nrounds = 1
tier2_early_stopping = 10
tier2_gridsearch_sd_limit = 1000
tier2_gridsearch_iterations = 100000


if input_file_type.upper() == 'DICT':
    if iso=='PJM':
        hypergrid_name = 'Gridsearch_2020_01_05_BACKTEST_DATA_DICT_MASTER_PJM_SD1000__PJM_50390_DART' #Filename of Stored Hypergrid From Gridsearch
        hypergrid_dict_name = 'GridsearchDict_2020_01_05_BACKTEST_DATA_DICT_MASTER_PJM_SD1000_'
        hypergrid_name = 'Gridsearch_2020_02_24_BACKTEST_DATA_DICT_MASTER_SPREAD_PJM_SD1000_revisedFeats2_PJM_50390_DART' #Filename of Stored Hypergrid From Gridsearch
        hypergrid_dict_name = 'GridsearchDict_2020_02_24_BACKTEST_DATA_DICT_MASTER_SPREAD_PJM_SD1000_revisedFeats2'


    elif iso=='MISO':
        hypergrid_name = 'Gridsearch_2020_02_24_BACKTEST_DATA_DICT_MASTER_SPREAD_MISO_SD1000_DART_revisedFeats1_MISO_AECI.ALTW_DART' #Filename of Stored Hypergrid From Gridsearch
        hypergrid_dict_name = 'GridsearchDict_2020_02_24_BACKTEST_DATA_DICT_MASTER_SPREAD_MISO_SD1000_DART_revisedFeats1'
    elif iso == 'SPP':
        hypergrid_name = 'Gridsearch_2020_03_19_BACKTEST_DATA_DICT_MASTER_SPP_SD1000_DART_revisedfeats1__SPP_AECC_CSWS_DART'  # Filename of Stored Hypergrid From Gridsearch
    elif iso == 'ERCOT':
        hypergrid_name = 'Gridsearch_2020_02_24_BACKTEST_DATA_DICT_MASTER_SPREAD_ERCOT_SD1000_SPREAD__ERCOT_AMISTAD_ALL$ERCOT_AMOCOOIL_CC1_SPREAD'  # Filename of Stored Hypergrid From Gridsearch
    elif iso == 'ISONE':
        hypergrid_name = 'Gridsearch_2020_02_24_BACKTEST_DATA_DICT_MASTER_SPREAD_ISONE_SD1000_DART_revisedfeats1__ISONE_10033_DART'  # Filename of Stored Hypergrid From Gridsearch
    elif iso == 'NYISO':
        hypergrid_name = 'Gridsearch_12092019_Master_Nodes_Dataset_Dict_SD6_NYISO_61752_DART'  # Filename of Stored Hypergrid From Gridsearch
    else:
        print('Correct Hypergrid Missing')
        exit()

if input_file_type.upper() == 'CSV':
    if iso=='PJM': hypergrid_name = 'Gridsearch_09_11_2019_GBM_DATA_PJM_V8.0_MASTER_207F_RGrid' #Filename of Stored Hypergrid From Gridsearch
    elif iso=='MISO': hypergrid_name = 'Gridsearch_09_11_2019_GBM_DATA_MISO_V8.0_MASTER_159F_RGrid' #Filename of Stored Hypergrid From Gridsearch
    else:
        print('Correct Hypergrid Missing')
        exit()

def do_create_models(input_filename, save_name, iso, feat_dict, input_file_type, cat_vars,  exp_folds, hypergrid_name,hypergrid_dict_name, num_grids, gpu_train, nrounds, early_stopping, model_creation_sd, static_directory, working_directory,model_type, run_reverse):
    # COORDINATES MODEL TRAINING
    model_file_directory = static_directory + '\ModelFiles\\NewModelFiles\\'
    model_data_directory = static_directory + '\ModelUpdateData\\'
    gridsearch_directory = working_directory + '\GridsearchFiles\\'
    featimport_directory = working_directory + '\FeatureImportanceFiles\\'
    gpu_train=False
    train_type='cpu'

    model_list = []

    # Read In Input File
    master_df = read_clean_data(input_filename=model_data_directory+input_filename,
                                input_file_type=input_file_type,
                                iso=iso)

    all_best_features_df = pd.read_csv(featimport_directory+all_best_features_filename + ".csv", dtype=np.str)

    # Make List of Targets And Calc Total Number of Trainings
    targets = [col for col in master_df.columns if (model_type in col)&(iso in col)]
    targets = targets[0:num_targets]

    total_num_trains = exp_folds*len(targets)
    print('Training '+str(total_num_trains) +' Models')


    # Cycle Through Each Target and Create Models For Each
    target_num = 1
    num_train = 1

    if run_reverse == True:
        targets = [ele for ele in reversed(targets)]

    for target in targets:
        print('Creating Models For: '+target+ '  **GPU Compute= '+str(gpu_train))

        # Read In Hypergrid
        try: #try to get custom hypergrid for node
            hypergrid_dict = load_obj(gridsearch_directory+hypergrid_dict_name)
            hypergrid_df = hypergrid_dict[target]
        except: #if none exists, use default hypergrid
            hypergrid_df = pd.read_csv(gridsearch_directory + hypergrid_name + '.csv')

        # Create Calculated Features (if reading from dict) Or Remove All Non-Target DARTs from CSV
        if input_file_type.upper() == 'DICT':
            print('Creating Features...')
            feature_df = create_features(input_df=master_df,
                                        feat_dict=feat_dict,
                                        target_name=target,
                                        iso=iso.upper(),
                                        all_best_features_df=all_best_features_df,
                                        cat_vars=cat_vars,
                                         static_directory=static_directory)

        elif input_file_type.upper() == 'CSV':
            target_col = master_df[target]
            feature_df = master_df[[col for col in master_df if model_type.upper() not in col]]
            feature_df = pd.concat([feature_df,target_col],axis=1)

        # Entire feature set length is the train set (no test set)
        train_df = feature_df

        # Create Folds For Multiple Exps (Done Before SD Range to Keep Folds Consistant Over Ranges
        exp_cv = GroupShuffleSplit(n_splits=exp_folds,test_size=0.20,random_state=1337)

        # Train Exps for Each Exp Fold (Folds Constant Across SD Ranges)
        exp_fold = 1

        for exp_train_index, exp_test_index in exp_cv.split(train_df, groups=train_df.index.get_level_values('Date')):
            train_exp, eval_exp = train_df.iloc[exp_train_index], train_df.iloc[exp_test_index]

            # Get a Random Hypergrid From the Top X Param Sets
            params = hypergrid_df['params'].iloc[random.randint(0, num_grids - 1)]
            try:  # if reading from CSV need literal eval
                params = literal_eval(params)
            except:  # if custom dict hypergrid, no literal eval needed
                pass

            print(target+ ': Creating Model:' + str(exp_fold) + '/' + str(exp_folds) + '  Total Progress: ' + str(round(num_train / total_num_trains * 100, 2)) + '%')

            pred_df, model = xgb_train(train_df=train_exp,
                                       test_df=pd.DataFrame(),
                                       eval_df=eval_exp,
                                       target=target,
                                       sd_limit=model_creation_sd,
                                       fit_params=params,
                                       gpu_train=gpu_train,
                                       nrounds=nrounds,
                                       early_stopping=early_stopping,
                                       verbose=False)

            # Save Model
            save_obj(model,model_file_directory+'ModelFile_'+target+'_'+train_type+'_'+str(exp_fold))
            model_list.append(target+'_'+str(exp_fold))
            pd.DataFrame(model_list, columns=[iso+'_ModelName']).to_csv(model_file_directory+'Model_List_'+save_name+'.csv')

            exp_fold += 1
            num_train += 1

        target_num += 1

    return model_list

def do_backtest(input_filename, save_name, num_targets, iso, feat_dict, input_file_type, cat_vars, start_date, sd_limit_range, exp_folds, cv_folds, hypergrid_name,hypergrid_dict_name, num_grids, gpu_train, nrounds, early_stopping, static_directory, working_directory,model_type, run_reverse, verbose=True):
    # COORDINATES THE BACKTEST. SPLITS AND CLEANS DATA BEFORE SENDING IT TO XGBTRAIN
    backtest_directory = static_directory + '\BacktestFiles\\'
    model_data_directory = static_directory + '\ModelUpdateData\\'
    gridsearch_directory = working_directory + '\GridsearchFiles\\'
    featimport_directory = working_directory + '\FeatureImportanceFiles\\'

    # Read In Input File
    master_df = read_clean_data(input_filename=model_data_directory+input_filename,
                                input_file_type=input_file_type,
                                iso=iso)

    # Make List of Targets And Calc Total Number of Trainings
    targets = [col for col in master_df.columns if (model_type.upper() in col)&(iso in col)]


    targets = targets[0:num_targets]

    total_num_trains = len(sd_limit_range)*cv_folds*exp_folds*len(targets)
    print('Training '+str(total_num_trains) +' Models.')

    # Create Empty DF To Add Preds To. Split Data If CV Is Not Being Run
    preds_df = pd.DataFrame(index=master_df.index)
    preds_exp_df = pd.DataFrame(index=master_df.index)

    # USE THIS INSTEAD TO LOAD AND CONCAT TO AN EXISTING BACKTEST
    # preds_df = pd.read_csv('X:\Research\BacktestFiles\Backtest_2020_02_24_BACKTEST_DATA_DICT_MASTER_MISO_EXP20_.csv', index_col=['Date','HE'],parse_dates=True)
    # preds_exp_df = pd.read_csv('X:\Research\BacktestFiles\Backtest_Exps_2020_02_24_BACKTEST_DATA_DICT_MASTER_MISO_EXP20_.csv', index_col=['Date','HE'],parse_dates=True)

    all_best_features_df = pd.read_csv(featimport_directory + all_best_features_filename + ".csv", dtype=np.str)

    # Cycle Through Each Target and Backtest Each
    target_num = 1
    num_train = 1

    if run_reverse == True:
        targets = [ele for ele in reversed(targets)]

    for target in targets:
        # Read In Hypergrid
        try: #try to get custom hypergrid for node
            hypergrid_dict = load_obj(gridsearch_directory+hypergrid_dict_name)
            hypergrid_df = hypergrid_dict[target]
        except: #if none exists, use default hypergrid
            hypergrid_df = pd.read_csv(gridsearch_directory + hypergrid_name + '.csv')

        print('Training Target: '+target+ '  **GPU Compute= '+str(gpu_train))
        # Create Calculated Features (if reading from dict) Or Remove All Non-Target DARTs from CSV
        if input_file_type.upper() == 'DICT':
            print('Creating Features...')
            feature_df = create_features(input_df=master_df,
                                       feat_dict=feat_dict,
                                       target_name=target,
                                       iso=iso.upper(),
                                       all_best_features_df=all_best_features_df,
                                       cat_vars=cat_vars,
                                       static_directory=static_directory)

        elif input_file_type.upper() == 'CSV':
            target_col = master_df[target]
            feature_df = master_df[[col for col in master_df if 'DART' not in col]]
            feature_df = pd.concat([feature_df,target_col],axis=1)


        ##############################################################################
        ### CV BACKTEST CODE Create Folds for CV Backtest If Number Of CV Folds >0 ###
        ##############################################################################

        if cv_folds>1:
            cv_fold = 1
            kf_cv = GroupKFold(n_splits=cv_folds)
            cv_preds_df = pd.DataFrame()
            cv_exp_preds_df = pd.DataFrame()

            for cv_train_index, cv_test_index in kf_cv.split(feature_df, groups=feature_df.index.get_level_values('Date')):
                train_cv_df, test_cv_df = feature_df.iloc[cv_train_index], feature_df.iloc[cv_test_index]
                sd_preds_df = pd.DataFrame(index=test_cv_df.index)
                sd_exp_preds_df = pd.DataFrame(index=test_cv_df.index)

                exp_cv = GroupShuffleSplit(n_splits=exp_folds,test_size=0.20,random_state=1337)

                # If Multiple SD Ranges Train A Model For Each
                sd_fold = 1

                for sd_limit in sd_limit_range:

                    # Train Exps for Each Exp Fold (Folds Constant Across SD Ranges)
                    exp_fold = 1
                    exp_df = pd.DataFrame(index=test_cv_df.index)
                    for exp_train_index, exp_test_index in exp_cv.split(train_cv_df, groups=train_cv_df.index.get_level_values('Date')):
                        train_cv_exp_df, eval_cv_exp_df = train_cv_df.iloc[exp_train_index], train_cv_df.iloc[exp_test_index]

                        # Get a Random Hypergrid From the Top X Param Sets
                        params = hypergrid_df['params'].iloc[random.randint(0,num_grids-1)]
                        try: # if reading from CSV need literal eval
                            params = literal_eval(params)
                        except: # if custom dict hypergrid, no literal eval needed
                            pass

                        print('Training CVFold: '+str(cv_fold) +'/'+str(cv_folds)+'  ExpFold:'+str(exp_fold)+'/'+str(exp_folds)+ '  StdDev: '+str(sd_limit)+'  Total Progress: '+str(round(num_train/total_num_trains*100,2))+'%' )

                        pred_df, model = xgb_train(train_df=train_cv_exp_df,
                                            test_df=test_cv_df,
                                            eval_df=eval_cv_exp_df,
                                            target = target,
                                            sd_limit=sd_limit,
                                            fit_params=params,
                                            gpu_train=gpu_train,
                                            nrounds=nrounds,
                                            early_stopping=early_stopping,
                                            verbose=False)

                        pred_df.columns = pred_df.columns+str(exp_fold)
                        exp_df = exp_df.join(pred_df,on=['Date','HE'])
                        exp_fold += 1
                        num_train += 1

                    # Calculate Exp Summary Statistics
                    sd_preds_df[target + '_SD' + str(sd_limit) + '_pred'] = exp_df.median(axis=1)
                    sd_preds_df[target + '_SD' + str(sd_limit) + '_sd'] = exp_df.std(axis=1)
                    sd_preds_df[target + '_SD' + str(sd_limit) + '_act'] = test_cv_df[target]

                    exp_df.columns = [col + '_SD' + str(sd_limit) for col in exp_df.columns]
                    exp_df[target + '_SD' + str(sd_limit) + '_act'] = test_cv_df[target]

                    sd_fold += 1

                # Stack the CV Results Together and Sort
                cv_preds_df = pd.concat([cv_preds_df,sd_preds_df],axis=0)
                cv_preds_df.sort_index(axis=0, level=1, ascending=True, inplace=True)
                cv_preds_df.sort_index(axis=0,level=0,ascending=True,inplace=True)

                cv_exp_preds_df = pd.concat([cv_exp_preds_df,exp_df],axis=0)
                cv_exp_preds_df.sort_index(axis=0, level=1, ascending=True, inplace=True)
                cv_exp_preds_df.sort_index(axis=0,level=0,ascending=True,inplace=True)

                cv_fold += 1

            # Add Each Successive Target To The Final Pred DF

            preds_df = pd.merge(preds_df,cv_preds_df,how='outer',on=['Date','HE'])
            preds_exp_df = pd.merge(preds_exp_df,cv_exp_preds_df,how='outer',on=['Date','HE'])

            preds_df.to_csv(backtest_directory+'Backtest_'+save_name+'.csv')
            # preds_exp_df.to_csv(backtest_directory+'Backtest_Exps_'+save_name+'.csv')


            target_num += 1


    return preds_df

def do_gridsearch(input_filename, save_name, iso, feat_dict, input_file_type, cat_vars,hypergrid_dict_name, sd_limit, cv_folds, gpu_train, nrounds, iterations, static_directory, working_directory,model_type, verbose=True):
    # COORDINATES THE GRIDSEARCH(S)
    gridsearch_directory = working_directory + '\GridsearchFiles\\'
    model_data_directory = static_directory + '\ModelUpdateData\\'
    feature_directory = working_directory + '\FeatureImportanceFiles\\'

    try:
        hypergrids = load_obj(gridsearch_directory+hypergrid_dict_name)
    except:
        hypergrids = dict()

    # Read In Input File
    master_df = read_clean_data(input_filename=model_data_directory+input_filename,
                                input_file_type=input_file_type,
                                iso=iso)

    # Make List of Targets And Calc Total Number of Trainings
    targets = [col for col in master_df.columns if (model_type.upper() in col) & (iso in col)]

    ###### Use these for each ISO
    if model_type.upper() == 'DART':
        target_dict = {'ISONE':['ISONE_10033_DART'],
        'NYISO':['NYISO_61752_DART'],
        'ERCOT':['ERCOT_AMISTAD_ALL_DART'],
        'SPP': ['SPP_AECC_CSWS_DART'],
        'PJM': ['PJM_50390_DART'],
        'MISO':['MISO_AECI.ALTW_DART']}
    elif model_type.upper() =='SPREAD':
        target_dict = {'ISONE':[''],
        'NYISO':[''],
        'ERCOT':['ERCOT_AMISTAD_ALL$ERCOT_AMOCOOIL_CC1_SPREAD'],
        'SPP': [''],
        'PJM': [''],
        'MISO':['']}


    targets = target_dict[iso]

    all_best_features_df = pd.read_csv(feature_directory+all_best_features_filename + ".csv", dtype=np.str)

    target_num = 1
    for target in targets:
        # if target in hypergrids.keys():
        #     print('Target ' +target+ ' already has hypergrid. Skipping...')
        #     continue

        print('Gridsearch Target: '+target+ '  ***GPU Compute= '+str(gpu_train))

        # Create Calculated Features (if reading from dict) Or Remove All Non-Target DARTs from CSV
        if input_file_type.upper() == 'DICT':
            print('Creating Features...')
            feature_df = create_features(input_df=master_df,
                                        feat_dict=feat_dict,
                                        target_name=target,
                                        iso=iso.upper(),
                                        all_best_features_df=all_best_features_df,
                                        cat_vars=cat_vars,
                                         static_directory=static_directory)

        elif input_file_type.upper() == 'CSV':
            target_col = master_df[target]
            feature_df = master_df[[col for col in master_df if model_type.upper() not in col]]
            feature_df = pd.concat([feature_df,target_col],axis=1)


        print('Training Gridsearch...')
        hypergrid = xgb_gridsearch(train_df=feature_df,
                                   target=target,
                                   cv_folds=cv_folds,
                                   iterations=iterations,
                                   nrounds = nrounds,
                                   sd_limit = sd_limit,
                                   gpu_train=gpu_train)

        hypergrid.to_csv(gridsearch_directory+'Gridsearch_' + save_name + '_' + target + '.csv', index=False)
        hypergrids.update({target: hypergrid})
        save_obj(hypergrids, gridsearch_directory+'GridsearchDict_' + save_name)

        target_num += 1

    return hypergrids

def do_tier2_gridsearch(backtest_filename, PnL_filename, save_name, cat_vars, sd_limit, cv_folds, gpu_train, nrounds, iterations,dart_sd_location_filter, static_directory, working_directory, verbose=True):
    # COORDINATES THE GRIDSEARCH(S)

    gridsearch_directory = working_directory + '\GridsearchFiles\\'
    backtest_directory = working_directory + '\BacktestFiles\\'
    PnL_directory = working_directory + '\PnLFiles\\'
    hypergrids = dict()
    backtest_df = pd.read_csv(backtest_directory + backtest_filename + '.csv', index_col=['Date', 'HE'],
                              parse_dates=True)
    backtest_df.dropna(axis=0, inplace=True)
    backtest_df = backtest_df[[col for col in backtest_df.columns if dart_sd_location_filter in col]]

    backtest_results = pd.ExcelFile(PnL_directory + PnL_filename + '.xlsx')
    hourly_PnL_df = pd.read_excel(backtest_results, 'Hourly_PnL', index_col=[0, 1], parse_dates=True)
    hourly_PnL_df.index.type = 'datetime64[ns]'

    targets = [col.replace('_act', '').replace('_' + dart_sd_location_filter, '') for col in backtest_df.columns if
               '_act' in col]


    total_num_trains = cv_folds * exp_folds * len(targets)
    print('Training ' + str(total_num_trains) + ' Tier2 Models')

    # Cycle Through Each Target and Backtest Each
    target_num = 1

    for target in targets[0:1]:
        print('Gridsearch Target: '+target+ '  ***GPU Compute= '+str(gpu_train))

        feature_df = create_tier2_features(backtest_df=backtest_df,
                                           hourly_PnL_df=hourly_PnL_df,
                                           target=target)

        target = target + '_Tier2Target'  # Added to differentiate the target column from the similiarly named feature columns

        feature_df.dropna(axis=0,inplace=True)  ##Remove Hours And Days That Wouldnt (Werent) Be Traded Due To Mean or SD Banding
        feature_df = feature_df[feature_df[target]!=0]

        print('Num Features: '+str(len(feature_df.columns)-1))

        # Add Categoricals
        feature_df = pd.get_dummies(feature_df, columns=cat_vars)
        print('Num Features W/Categoricals: ' + str(len(feature_df.columns) - 1))

        print('Training Gridsearch...')

        hypergrid = xgb_gridsearch(train_df=feature_df,
                                   target=target,
                                   cv_folds=cv_folds,
                                   iterations=iterations,
                                   nrounds = nrounds,
                                   sd_limit = sd_limit,
                                   gpu_train=gpu_train)

        hypergrid.to_csv(gridsearch_directory+'Gridsearch_' + save_name + '_' + target + '.csv', index=False)
        # hypergrids.update({target: hypergrid})
        # save_obj(hypergrids, gridsearch_directory+'GridsearchDict_'+hourly_daily+'_'  + save_name)
        target_num += 1
    return hypergrids

def do_tier2_backtest(backtest_filename,PnL_filename, save_name, cat_vars, sd_limit_range, exp_folds, cv_folds, hypergrid_name, num_grids, gpu_train, nrounds, early_stopping,dart_sd_location_filter, working_directory,static_directory, verbose=True):
    backtest_directory = static_directory + '\BacktestFiles\\'
    PnL_directory = working_directory + '\PnLFiles\\'
    gridsearch_directory = working_directory + '\GridsearchFiles\\'

    backtest_df = pd.read_csv(backtest_directory+backtest_filename + '.csv',index_col=['Date','HE'],parse_dates=True)
    backtest_df.dropna(axis=0, inplace=True)
    backtest_df = backtest_df[[col for col in backtest_df.columns if dart_sd_location_filter in col]]

    backtest_results = pd.ExcelFile(PnL_directory+PnL_filename + '.xlsx')
    hourly_PnL_df = pd.read_excel(backtest_results, 'Hourly_PnL',index_col=[0,1],parse_dates=True)
    hourly_PnL_df.index.type = 'datetime64[ns]'

    targets = [col.replace('_act', '').replace('_' + dart_sd_location_filter, '') for col in backtest_df.columns if'_act' in col]
    preds_df = pd.DataFrame(index=hourly_PnL_df.index)

    total_num_trains = cv_folds*exp_folds*len(targets)
    print('Training '+str(total_num_trains) +' Tier2 Models')

    # Cycle Through Each Target and Backtest Each
    target_num = 1
    num_train = 1

    for target in targets:
        print('Training Target: ' + target + '  **GPU Compute= ' + str(gpu_train))
        hypergrid_df = pd.read_csv(gridsearch_directory+hypergrid_name + '.csv')

        feature_df = create_tier2_features(backtest_df=backtest_df,
                                           hourly_PnL_df=hourly_PnL_df,
                                           target=target)


        target = target + '_Tier2Target'  # Added to differentiate the target column from the similiarly named feature columns
        feature_df.dropna(axis=0, inplace=True)  ##Remove Hours And Days That Wouldnt (Werent) Be Traded Due To Mean or SD Banding
        feature_df = feature_df[feature_df[target] != 0]

        num_samples = len(feature_df)
        print('Number of Samples for Tier2 Traning: '+str(num_samples))

        if len(feature_df)<24:
            print('Skipping tier2 target ' + target + ' due to limited number of samples: '+ len(feature_df))
            continue

        print('Num Features: ' + str(len(feature_df.columns) - 1))
        # Add Categoricals
        feature_df = pd.get_dummies(feature_df, columns=cat_vars)
        print('Num Features W/Categoricals: ' + str(len(feature_df.columns) - 1))

        ##############################################################################
        ### CV BACKTEST CODE Create Folds for CV Backtest If Number Of CV Folds >0 ###
        ##############################################################################

        if cv_folds > 1:
            cv_fold = 1
            kf_cv = KFold(n_splits=cv_folds, random_state=1337, shuffle=False)
            cv_preds_df = pd.DataFrame()

            for cv_train_index, cv_test_index in kf_cv.split(feature_df):
                train_cv_df, test_cv_df = feature_df.iloc[cv_train_index], feature_df.iloc[cv_test_index]
                sd_preds_df = pd.DataFrame(index=test_cv_df.index)

                # Create Folds For Multiple Exps (Done Before SD Range to Keep Folds Consistant Over Ranges
                exp_cv = KFold(n_splits=exp_folds, random_state=1337, shuffle=True)

                # If Multiple SD Ranges Train A Model For Each
                sd_fold = 1

                for sd_limit in sd_limit_range:

                    # Train Exps for Each Exp Fold (Folds Constant Across SD Ranges)
                    exp_fold = 1
                    exp_preds_df = pd.DataFrame(index=test_cv_df.index)
                    for exp_train_index, exp_test_index in exp_cv.split(train_cv_df):
                        train_cv_exp_df, eval_cv_exp_df = train_cv_df.iloc[exp_train_index], train_cv_df.iloc[
                            exp_test_index]

                        # Get a Random Hypergrid From the Top X Param Sets
                        params = hypergrid_df['params'].iloc[random.randint(0, num_grids - 1)]
                        params = literal_eval(params)

                        print('Training CVFold: ' + str(cv_fold) + '/' + str(cv_folds) + '  ExpFold:' + str(
                            exp_fold) + '/' + str(exp_folds) + '  StdDev: ' + str(
                            sd_limit) + '  Total Progress: ' + str(
                            round(num_train / total_num_trains * 100, 2)) + '%')

                        pred_df,model = xgb_train(train_df=train_cv_exp_df,
                                            test_df=test_cv_df,
                                            eval_df=eval_cv_exp_df,
                                            target=target,
                                            sd_limit=sd_limit,
                                            fit_params=params,
                                            gpu_train=gpu_train,
                                            nrounds=nrounds,
                                            early_stopping=early_stopping,
                                            verbose=False)
                        pred_df.columns = pred_df.columns + str(exp_fold)

                        exp_preds_df = exp_preds_df.join(pred_df, on=['Date','HE'])

                        exp_fold += 1
                        num_train += 1

                    # Calculate Exp Summary Statistics
                    sd_preds_df[target + '_SD' + str(sd_limit) + '_pred'] = exp_preds_df.median(axis=1)
                    sd_preds_df[target + '_SD' + str(sd_limit) + '_sd'] = exp_preds_df.std(axis=1)
                    sd_preds_df[target + '_SD' + str(sd_limit) + '_act'] = test_cv_df[target]
                    sd_fold += 1

                # Stack the CV Results Together and Sort
                cv_preds_df = pd.concat([cv_preds_df, sd_preds_df], axis=0)
                cv_preds_df.sort_index(axis=0, level=1, ascending=True, inplace=True)
                cv_preds_df.sort_index(axis=0, level=0, ascending=True, inplace=True)
                cv_fold += 1

            # Add Each Successive Target To The Final Pred DF
            preds_df = pd.merge(preds_df, cv_preds_df, how='outer', on=['Date','HE'])
            preds_df.reset_index(inplace=True)
            preds_df.set_index(['Date','HE'],inplace=True)

            preds_df.to_csv(backtest_directory+'Backtest_' + save_name + '.csv')
            target_num += 1
    # preds_df = preds_df.join(daily_mw_df, how='inner')
    # preds_df.to_csv(save_directory + 'Backtest_' + hourly_daily + '_' + save_name + '.csv')

def do_tier2_create_models(backtest_filename,PnL_filename, save_name, cat_vars, exp_folds, hypergrid_name, num_grids, gpu_train, nrounds, early_stopping,dart_sd_location_filter, tier2_model_creation_sd, static_directory):

    model_file_directory = static_directory + '\ModelFiles\\'

    model_list = []

    backtest_df = pd.read_csv(backtest_filename + '.csv',index_col=['Date','HE'],parse_dates=True)
    backtest_df.dropna(axis=0, inplace=True)

    backtest_df = backtest_df[[col for col in backtest_df.columns if dart_sd_location_filter in col]]
    backtest_df.columns = [col.replace('_SD1','').replace('_SD2','').replace('_SD3','').replace('_SD4','').replace('_SD4.5','').replace('_SD5','').replace('_SD5.5','').replace('_SD6','').replace('_SD6.5','').replace('_SD7','').replace('_SD7.5','').replace('_SD1000','') for col in backtest_df.columns]

    backtest_results = pd.ExcelFile(PnL_filename + '.xlsx')
    daily_PnL_df = pd.read_excel(backtest_results, 'Daily_PnL',index_col=[0],parse_dates=True)
    daily_PnL_df.index.name = 'Date'
    daily_PnL_df.index.type = 'datetime64[ns]'
    hourly_PnL_df = pd.read_excel(backtest_results, 'Hourly_PnL',index_col=[0,1],parse_dates=True)
    hourly_PnL_df.index.type = 'datetime64[ns]'
    hourly_PnL_df.to_csv('hourl.csv')

    hourly_PnL_df.columns = [col.replace('_SD1','').replace('_SD2','').replace('_SD3','').replace('_SD4','').replace('_SD4.5','').replace('_SD5','').replace('_SD5.5','').replace('_SD6','').replace('_SD6.5','').replace('_SD7','').replace('_SD7.5','').replace('_SD1000','') for col in hourly_PnL_df.columns]
    targets = [col for col in hourly_PnL_df.columns if 'Total' not in col]

    total_num_trains = exp_folds*len(targets)
    print('Creating '+str(total_num_trains) +' Tier2 Models')

    # Cycle Through Each Target and Backtest Each
    target_num = 1
    num_train = 1
    for target in targets:
        print('Creating Target: ' + target + '  **GPU Compute= ' + str(gpu_train))
        hypergrid_df = pd.read_csv(hypergrid_name + '.csv')

        if hourly_daily == 'daily':
            feature_df = create_tier2daily_features(backtest_df=backtest_df,
                                                    daily_PnL_df=daily_PnL_df,
                                                    target=target)
        elif hourly_daily == 'hourly':
            feature_df = create_tier2hourly_features(backtest_df=backtest_df,
                                                     hourly_PnL_df=hourly_PnL_df,
                                                     target=target)
            cat_vars=['Month','Weekday','HourEnding']

        target = target+'_PnL'   # Added to differentiate the target column from the similiarly named feature columns
        feature_df.dropna(axis=0, inplace=True)  ##Remove Hours And Days That Wouldnt (Werent) Be Traded Due To Mean or SD Banding
        feature_df = feature_df[feature_df[target] != 0]


        print('Num Features: ' + str(len(feature_df.columns) - 1))
        # Add Categoricals
        feature_df = pd.get_dummies(feature_df, columns=cat_vars)
        print('Num Features W/Categoricals: ' + str(len(feature_df.columns) - 1))


        # Create Folds For Multiple Exps (Done Before SD Range to Keep Folds Consistant Over Ranges
        exp_cv = KFold(n_splits=exp_folds, random_state=1337, shuffle=True)

        # Train Exps for Each Exp Fold (Folds Constant Across SD Ranges)
        exp_fold = 1

        for exp_train_index, exp_test_index in exp_cv.split(feature_df):
            train_cv_exp_df, eval_cv_exp_df = feature_df.iloc[exp_train_index], feature_df.iloc[exp_test_index]

            # Get a Random Hypergrid From the Top X Param Sets
            params = hypergrid_df['params'].iloc[random.randint(0, num_grids - 1)]
            params = literal_eval(params)

            print('Creating Model ' + str(exp_fold) + '/' + str(exp_folds) + '  Total Progress: ' + str(round(num_train / total_num_trains * 100, 2)) + '%')

            pred_df ,model = xgb_train(train_df=train_cv_exp_df,
                                       test_df=pd.DataFrame(),
                                       eval_df=eval_cv_exp_df,
                                       target=target,
                                       sd_limit=tier2_model_creation_sd,
                                       fit_params=params,
                                       gpu_train=gpu_train,
                                       nrounds=nrounds,
                                       early_stopping=early_stopping,
                                       verbose=False)

            # Save Model
            save_obj(model,model_file_directory+'ModelFile_Tier2_'+target+'_'+str(exp_fold))
            model_list.append(target+'_'+str(exp_fold))
            pd.DataFrame(model_list, columns=[iso+'_Tier2_ModelName_']).to_csv(model_file_directory+'Tier2_Model_List_'+iso+'.csv')

            exp_fold += 1
            num_train += 1

        target_num += 1

    return model_list

rev=''
if run_reverse==True:
    rev='rev'

if run_gridsearch:
    gridsearch_save_name = input_file_name + '_' + iso + '_SD' + str(gridsearch_sd_limit)  + '_'+model_type+ '_' + name_adder + '_'+rev
    print('RUNNING GRIDSEARCH: ' + gridsearch_save_name)
    do_gridsearch(input_filename=input_file_name,
                  save_name=gridsearch_save_name,
                  iso = iso,
                  feat_dict=gridsearch_feat_dict,
                  hypergrid_dict_name=hypergrid_dict_name,
                  input_file_type = input_file_type,
                  cat_vars = cat_vars,
                  sd_limit = gridsearch_sd_limit,
                  cv_folds = gridsearch_cv_folds,
                  gpu_train= gridsearch_gpu_train,
                  nrounds= gridsearch_nrounds,
                  iterations=gridsearch_iterations,
                  static_directory=static_directory,
                  working_directory=working_directory,
                  model_type=model_type)

if run_backtest:
    backtest_save_name = input_file_name + '_' + iso + '_EXP' + str(exp_folds) + '_'+model_type+ '_' + name_adder + '_'+rev
    print('RUNNING BACKTEST: ' + backtest_save_name)
    output_df = do_backtest(input_filename=input_file_name,
                            save_name=backtest_save_name,
                            num_targets=num_targets,
                            iso = iso,
                            feat_dict=feat_dict,
                            input_file_type = input_file_type,
                            cat_vars = cat_vars,
                            start_date = backtest_start_date,
                            sd_limit_range = sd_limit_range,
                            exp_folds = exp_folds,
                            cv_folds = cv_folds,
                            hypergrid_name = hypergrid_name,
                            hypergrid_dict_name = hypergrid_dict_name,
                            num_grids = num_top_grids,
                            gpu_train=gpu_train,
                            nrounds=nrounds,
                            early_stopping=early_stopping,
                            static_directory=static_directory,
                            working_directory=working_directory,
                            model_type=model_type,
                            run_reverse=run_reverse)

if run_create_models:
    backtest_save_name = input_file_name + '_' + iso + '_'+model_type
    print('CREATING TIER 1 MODELS: ' + backtest_save_name)
    model_list = do_create_models(input_filename=input_file_name,
                                  save_name=backtest_save_name,
                                  iso=iso,
                                  feat_dict=feat_dict,
                                  input_file_type=input_file_type,
                                  cat_vars=cat_vars,
                                  exp_folds=exp_folds,
                                  hypergrid_name=hypergrid_name,
                                  hypergrid_dict_name=hypergrid_dict_name,
                                  num_grids=num_top_grids,
                                  gpu_train=gpu_train,
                                  nrounds=nrounds,
                                  early_stopping=early_stopping,
                                  model_creation_sd = model_creation_sd,
                                  static_directory=static_directory,
                                  working_directory=working_directory
                                  ,model_type=model_type,
                                  run_reverse=run_reverse)

if run_tier2_backtest:
    tier2_save_name = 'Tier2_'+tier2_backtest_filename

    do_tier2_backtest(backtest_filename=tier2_backtest_filename,
                      PnL_filename=tier2_PnL_filename,
                      save_name=tier2_save_name,
                      cat_vars = tier2_cat_vars,
                      sd_limit_range=tier2_sd_limit_range,
                      exp_folds=tier2_exp_folds,
                      cv_folds=tier2_cv_folds,
                      hypergrid_name=tier2_hypergrid_name,
                      num_grids=tier2_num_grids,
                      gpu_train=tier2_gpu_train,
                      nrounds=tier2_nrounds,
                      early_stopping=tier2_early_stopping,
                      dart_sd_location_filter=tier2_dart_sd_location_filter,
                      working_directory=working_directory,
                      static_directory=static_directory)

if run_tier2_gridsearch:
    tier2_save_name = 'Tier2_'+'_'+tier2_backtest_filename
    print('RUNNING GRIDSEARCH TIER2: ' + tier2_save_name)
    do_tier2_gridsearch(backtest_filename=tier2_backtest_filename,
                        PnL_filename=tier2_PnL_filename,
                        save_name=tier2_save_name,
                        cat_vars = tier2_cat_vars,
                        sd_limit = tier2_gridsearch_sd_limit,
                        cv_folds = tier2_cv_folds,
                        gpu_train= tier2_gpu_train,
                        nrounds= tier2_nrounds,
                        dart_sd_location_filter=tier2_dart_sd_location_filter,
                        iterations=tier2_gridsearch_iterations,
                        static_directory=static_directory,
                        working_directory=working_directory)

if run_tier2_create_models:
    backtest_save_name = tier2_backtest_filename + '_' + iso
    print('CREATING TIER2 MODELS: ' + backtest_save_name)
    model_list = do_tier2_create_models(backtest_filename=tier2_backtest_filename,
                                        PnL_filename=tier2_PnL_filename,
                                        save_name=backtest_save_name,
                                        cat_vars = tier2_cat_vars,
                                        exp_folds=tier2_exp_folds,
                                        hypergrid_name=tier2_hypergrid_name,
                                        num_grids=tier2_num_grids,
                                        gpu_train=tier2_gpu_train,
                                        nrounds=tier2_nrounds,
                                        early_stopping=tier2_early_stopping,
                                        dart_sd_location_filter=tier2_dart_sd_location_filter,
                                        hourly_daily=tier2_hourly_daily,
                                        tier2_model_creation_sd=tier2_model_creation_sd,
                                        static_directory=static_directory)
