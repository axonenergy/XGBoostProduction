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

start_date = '2020_04_28'   # 7/15/2015 start date -
end_date = '2020_05_28'   # check here for most recent file for end date: https://marketplace.spp.org/pages/rtbm-lmp-by-location# - approx 7 days before current date

previous_data_dict_name = '2020_05_04_BACKTEST_DATA_DICT_RAW_GASLMPs'

data_dict = get_ISO_api_data(start_date=start_date,
                             end_date=end_date,
                             previous_data_dict_name = save_directory+previous_data_dict_name,
                             static_directory=static_directory,
                             working_directory=working_directory,
                             concat_old_dict=True)


####################################################################################################################
#                                           API PULLS FOR HISTORIC DA LMP DATES                                          #
####################################################################################################################

start_date = '2020_04_28'   # 7/15/2015 start date
end_date = '2020_05_28' # check here for most recent file for end date: https://marketplace.spp.org/pages/rtbm-lmp-by-location#

previous_data_dict_name = '2020_05_04_LMP_DATA_DICT_MASTER'

data_dict = get_lmps(start_date=start_date,
                     end_date=end_date,
                     previous_data_dict_name = save_directory+previous_data_dict_name,
                     static_directory=static_directory,
                     working_directory=working_directory,
                     concat_old_dict=True)



####################################################################################################################
####################################################################################################################

#Use this code to get reference prices

# data_dict_name = '2020_05_04_LMP_DATA_DICT_MASTER'
#
# get_reference_prices(data_dict_name=data_dict_name,
#                      working_directory=working_directory,
#                      static_directory=static_directory)
#


####################################################################################################################
####################################################################################################################

# start_date = '2020_04_25'   # 7/15/2015 start date -
# end_date = '2020_05_28'   # check here for most recent file for end date: https://marketplace.spp.org/pages/rtbm-lmp-by-location# - approx 7 days before current date
#
# start_date = datetime.datetime.strptime(start_date, '%Y_%m_%d')
# start_date = start_date.strftime('%m/%d/%Y')
# end_date = datetime.datetime.strptime(end_date, '%Y_%m_%d')
# end_date = end_date.strftime('%m/%d/%Y')
#
# utc_spread_locs_df = pd.read_csv(save_directory + 'PJM_UTC_Locs.csv')
#
#
# input_dict = load_obj(save_directory+'2020_05_04_BACKTEST_DATA_DICT_MASTER')
# input_df = input_dict['EST']
#
# input_df = input_df[[col for col in input_df.columns if 'SPREAD' in col]]
# input_df = input_df[[col for col in input_df.columns if 'ERCOT' in col]]
#
# input_df = input_df.dropna(axis=0)


## Code to find all the abs values of spreads
# input_df = input_dict['EST']
#
# # input_df = input_df[[col for col in input_df.columns if col in set(utc_spread_locs_df['ModelName'])]]
#
# input_df = input_df[[col for col in input_df.columns if 'DART' in col]]
# input_df = input_df[[col for col in input_df.columns if 'ERCOT' in col]]
#

#
#
# # input_df = input_df[input_df.columns[0:4]]
#
# input_df = input_df[~input_df.index.duplicated()].copy()
#
# spread_df = pd.DataFrame()
#
# for source_col_num in range(len(input_df.columns)):
#     source_name = input_df.columns[source_col_num]
#     for sink_col_num in range(source_col_num + 1, len(input_df.columns), 1):
#         sink_name = input_df.columns[sink_col_num]
#         spread_df[(source_name + '$' + sink_name + '_SPREAD').replace('_DART', '')] = input_df[source_name] - input_df[sink_name]
#
#
# spread_df = input_df.abs()
#
# avg_df = pd.DataFrame(spread_df.mean())
# avg_df.reset_index(inplace=True)
# avg_df.columns = ['Spread','Avg Spread']
#
#
# avg_df['Source'] =avg_df['Spread'].apply(lambda string: string.split('$')[0].replace('_SPREAD',''))
# avg_df['Sink'] =avg_df['Spread'].apply(lambda string: string.split('$')[1].replace('_SPREAD',''))
#
# avg_df = avg_df.sort_values(by=['Avg Spread'], ascending=False)
#
# avg_df['Avg Spread'] = avg_df['Avg Spread'].round(2)
#
# avg_df.reset_index(inplace=True,drop=True)
#
# avg_df.to_csv('AbsoluteValue_of_currTraded_ERCOT_spread_locs.csv')
#
#
#
#

