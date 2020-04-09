from API_Lib import get_ISO_api_data
from API_Lib import get_lmps
from API_Lib import get_reference_prices
from API_Lib import add_tall_files_together
from API_Lib import get_spreads
from API_Lib import get_var_dict
from API_Lib import load_obj
from API_Lib import save_obj
from dateutil.parser import parse
from API_Lib import process_YES_timeseries

import pandas as pd
import os


pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 5000)
save_directory = os.getcwd() + '\ModelUpdateData\\'
working_directory = 'X:\\Research\\'
static_directory = 'C:\\XGBoostProduction\\'

####################################################################################################################
#                                           API PULLS FOR HISTORIC DART DATES                                          #
####################################################################################################################

# start_date = '2020_03_18'   # 7/15/2015 start date
# end_date = '2020_03_27'   # check here for most recent file for end date: https://marketplace.spp.org/pages/rtbm-lmp-by-location#
#
# previous_data_dict_name = '2020_03_19_BACKTEST_DATA_DICT_RAW_REVISED'
#
# data_dict = get_ISO_api_data(start_date=start_date,
#                              end_date=end_date,
#                              previous_data_dict_name = save_directory+previous_data_dict_name,
#                              static_directory=static_directory,
#                              working_directory=working_directory,
#                              concat_old_dict=True)

input_dict = load_obj(save_directory+ '2020_03_27_BACKTEST_DATA_DICT_MASTER')
df = input_dict['EPT']
df = df[[col for col in df.columns if 'DART' in col]]
orig_cols = len(df.columns)
orig_len = len(df)

hard_start_date = '03/19/2020'
hard_start_date = parse(hard_start_date)
hard_end_date = '03/21/2020'
hard_end_date = parse(hard_end_date)


df = df[(df.index.get_level_values('Date')>=hard_start_date) & (df.index.get_level_values('Date')<=hard_end_date)]

df.to_csv('EPT_2020.csv')


# output_dict = {'origCols':orig_cols}
#
# output_dict['origLen'] = orig_len
#
# for iso in ['PJM','MISO','ERCOT','ISONE','NYISO','SPP']:
#     temp_df = df[[col for col in df.columns if iso in col]]
#     spread = len([col for col in temp_df.columns if 'SPREAD' in col])
#     spread_lag = len([col for col in temp_df.columns if 'SPR_EAD' in col])
#     dart = len([col for col in temp_df.columns if 'DART' in col])
#     dart_lag = len([col for col in temp_df.columns if 'DA_RT' in col])
#     load = len([col for col in temp_df.columns if 'FLOAD' in col])
#     temp = len([col for col in temp_df.columns if 'FTEMP' in col])
#     outage = len([col for col in temp_df.columns if 'OUTAGE' in col])
#     output_dict[iso+'_spread'] = spread
#     output_dict[iso + '_spread_lag'] = spread_lag
#     output_dict[iso + '_dart'] = dart
#     output_dict[iso + '_dart_lag'] = dart_lag
#     output_dict[iso + '_load'] = load
#     output_dict[iso + '_temp'] = temp
#     output_dict[iso + '_outage'] = outage
#
# output_df = pd.DataFrame(output_dict, index=[0])
# output_df.to_csv('3-87-2020_new_featnums.csv')

#

####################################################################################################################
#                                           API PULLS FOR HISTORIC DA LMP DATES                                          #
####################################################################################################################


# start_date = '2020_03_18'   # 7/15/2015 start date
# end_date = '2020_03_20' # check here for most recent file for end date: https://marketplace.spp.org/pages/rtbm-lmp-by-location#
#
# previous_data_dict_name = '2020_03_19_LMP_DATA_DICT_MASTER_REVISED'
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

#Use this code to get reference prices

# data_dict_name = '2020_03_19_BACKTEST_DATA_DICT_RAW'
#
# get_reference_prices(data_dict_name=data_dict_name,
#                      working_directory=working_directory,
#                      static_directory=static_directory)



####################################################################################################################
####################################################################################################################
