from API_Lib import get_ISO_api_data
from API_Lib import get_lmps
from API_Lib import get_reference_prices
from API_Lib import add_tall_files_together
from API_Lib import get_spreads
from API_Lib import get_var_dict
from API_Lib import load_obj
from API_Lib import save_obj
from dateutil.parser import parse
import datetime
from API_Lib import process_YES_timeseries
from API_Lib import process_YES_daily_price_tables

import pandas as pd
import os


pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 5000)
save_directory = os.getcwd() + '\ModelUpdateData\\'
working_directory = 'X:\\Research\\'
var_directory = 'X:\\Production\\'
static_directory = 'C:\\XGBoostProduction\\'


####################################################################################################################
#                                           API PULLS FOR HISTORIC DART DATES                                          #
####################################################################################################################

# start_date = '2020_05_20'   # 7/15/2015 start date -
# end_date = '2020_06_04'   # check here for most recent file for end date: https://marketplace.spp.org/pages/rtbm-lmp-by-location# - approx 7 days before current date
#
# previous_data_dict_name = '2020_05_28_BACKTEST_DATA_DICT_RAW'
#
# data_dict = get_ISO_api_data(start_date=start_date,
#                              end_date=end_date,
#                              previous_data_dict_name = save_directory+previous_data_dict_name,
#                              static_directory=static_directory,
#                              working_directory=working_directory,
#                              concat_old_dict=True)


####################################################################################################################
#                                           API PULLS FOR HISTORIC DA LMP DATES                                          #
####################################################################################################################

# start_date = '2020_05_18'   # 7/15/2015 start date -
# end_date = '2020_06_04'   # check here for most recent file for end date: https://marketplace.spp.org/pages/rtbm-lmp-by-location# - approx 7 days before current date
#
# previous_data_dict_name = '2020_05_28_LMP_DATA_DICT_MASTER'
#
# data_dict = get_lmps(start_date=start_date,
#                      end_date=end_date,
#                      previous_data_dict_name = save_directory+previous_data_dict_name,
#                      static_directory=static_directory,
#                      working_directory=working_directory,
#                      concat_old_dict=True)
#


####################################################################################################################
####################################################################################################################

# # Use this code to get reference prices
#
# data_dict_name = '2020_06_04_LMP_DATA_DICT_MASTER'
#
# get_reference_prices(data_dict_name=data_dict_name,
#                      working_directory=working_directory,
#                      static_directory=static_directory)



####################################################################################################################
####################################################################################################################

# start_date = '2020_06_08'   # 7/15/2015 start date -
# end_date = '2020_05_28'   # check here for most recent file for end date: https://marketplace.spp.org/pages/rtbm-lmp-by-location# - approx 7 days before current date
#
# start_date = datetime.datetime.strptime(start_date, '%Y_%m_%d')
# start_date = start_date.strftime('%m/%d/%Y')
# end_date = datetime.datetime.strptime(end_date, '%Y_%m_%d')
# end_date = end_date.strftime('%m/%d/%Y')
#
#
#
# input_dict = load_obj(save_directory+'2020_05_28_BACKTEST_DATA_DICT_RAW')
# input_df = input_dict['EST']
#
# input_df = input_df[[col for col in input_df.columns if '_DART' in col]]
# input_df = input_df[[col for col in input_df.columns if 'SPP_' in col]]
#
# corr_matrix = input_df.corr()
#
# pd.DataFrame(corr_matrix).to_csv('corr_full_spp.csv')
        # orig_cols = max(len(temp_df.columns), 0.00001)
        # for i in range(len(corr_matrix.columns)):
        #     for j in range(i):
        #         if abs(corr_matrix.iloc[i, j]) > perc:
        #              colname = corr_matrix.columns[i]
        #              # if colname not in set(required_locs_df['ModelName']):
        #              correlated_dart_features.add(colname)
        #              # else:
        #              #    print('ignored dropping:' + colname)


####################################################################################################################
####################################################################################################################

###### Use this code for bulk- add of DART data to the VAR dictionary after update.

# var_dict_name = '2020_06_04_VAR_DART_DICT'
#
# predict_dates = [
#     '06_03_2020',
#     '06_04_2020',
#     '06_05_2020',
#     '06_06_2020',
#     '06_07_2020',
#     '06_08_2020',
#     '06_09_2020',
#     '06_10_2020',
#     '06_11_2020',
#     '06_12_2020',
#     '06_13_2020',
#     '06_14_2020',
#     '06_15_2020'
# ]
#
# var_dict = load_obj(save_directory+var_dict_name)
#
# for predict_date in predict_dates:
#     print(predict_date)
#     predict_date = datetime.datetime.strptime(predict_date, '%m_%d_%Y')
#
#     # Get Daily DART data
#     yes_pricetable_dict = process_YES_daily_price_tables(predict_date=predict_date,
#                                                          input_timezone='CPT',
#                                                          working_directory=var_directory,
#                                                          dart_only=False)
#
#
#     ### Add daily DART data to VAR dataframe
#     for timezone, df in var_dict.items():
#         var_df = df[[col for col in df.columns if '_DART' in col]]
#         add_df = yes_pricetable_dict[timezone]
#         add_df = add_df[[col for col in add_df.columns if col in var_df.columns]]
#
#         add_df.drop(add_df.tail(17).index, inplace=True)
#
#
#         var_df = var_df.drop(index=add_df.index,errors='ignore')
#         new_df = pd.concat([var_df,add_df],axis=0,sort=True).sort_index(ascending=True)
#         var_dict[timezone]=new_df
#
# # Save updated dict
# var_dict['EST'].to_csv('output.csv')
#
# #
# save_dict = save_obj(var_dict,save_directory+var_dict_name)

####################################################################################################################
####################################################################################################################

# hard_start_date = '06/01/2020'
# hard_start_date = parse(hard_start_date)
#
# hard_end_date = '06/05/2020'
# hard_end_date = parse(hard_end_date)
#
#
# var_dict = load_obj(save_directory+'2020_06_04_BACKTEST_DATA_DICT_RAW')
# df = var_dict['EPT']
# df = df[[col for col in df.columns if (('DART' not in col)&('LMP' not in col))]]
# df = df[df.index.get_level_values('Date') >= hard_start_date]
# df = df[df.index.get_level_values('Date') <= hard_end_date]
#
# df.to_csv('EPT.csv')