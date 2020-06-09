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

start_date = '2020_05_20'   # 7/15/2015 start date -
end_date = '2020_06_04'   # check here for most recent file for end date: https://marketplace.spp.org/pages/rtbm-lmp-by-location# - approx 7 days before current date

previous_data_dict_name = '2020_05_28_BACKTEST_DATA_DICT_RAW'

data_dict = get_ISO_api_data(start_date=start_date,
                             end_date=end_date,
                             previous_data_dict_name = save_directory+previous_data_dict_name,
                             static_directory=static_directory,
                             working_directory=working_directory,
                             concat_old_dict=True)


####################################################################################################################
#                                           API PULLS FOR HISTORIC DA LMP DATES                                          #
####################################################################################################################

start_date = '2020_05_18'   # 7/15/2015 start date -
end_date = '2020_06_04'   # check here for most recent file for end date: https://marketplace.spp.org/pages/rtbm-lmp-by-location# - approx 7 days before current date

previous_data_dict_name = '2020_05_28_LMP_DATA_DICT_MASTER'

data_dict = get_lmps(start_date=start_date,
                     end_date=end_date,
                     previous_data_dict_name = save_directory+previous_data_dict_name,
                     static_directory=static_directory,
                     working_directory=working_directory,
                     concat_old_dict=True)



####################################################################################################################
####################################################################################################################

# Use this code to get reference prices

data_dict_name = '2020_06_04_LMP_DATA_DICT_MASTER'

get_reference_prices(data_dict_name=data_dict_name,
                     working_directory=working_directory,
                     static_directory=static_directory)



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

