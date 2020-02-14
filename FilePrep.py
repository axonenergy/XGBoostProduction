from API_Lib import get_ISO_api_data
from API_Lib import post_process_backtest_data

import pandas as pd
import os
from API_Lib import add_tall_files_together
from API_Lib import get_lmps
from XGBLib import save_obj
from XGBLib import load_obj

pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 5000)
save_directory = os.getcwd() + '\ModelUpdateData\\'
working_directory = 'X:\\Research\\'
static_directory = 'C:\\XGBoostProduction\\'

####################################################################################################################
#                                           API PULLS FOR HISTORIC DART DATES                                          #
####################################################################################################################


start_date = '2020_01_04'   # 7/15/2015 start date
end_date = '2020_01_10'

previous_data_dict_name = '2020_01_05_BACKTEST_DATA_DICT_RAW'

data_dict = get_ISO_api_data(start_date=start_date,
                             end_date=end_date,
                             previous_data_dict_name = save_directory+previous_data_dict_name,
                             static_directory=static_directory,
                             working_directory=working_directory,
                             concat_old_dict=True)
#


####################################################################################################################
#                                           API PULLS FOR HISTORIC DA LMP DATES                                          #
####################################################################################################################


# start_date = '2015_07_15'   # 7/15/2015 start date
# end_date = '2020_01_05'
#
# previous_data_dict_name = '2020_01_05_BACKTEST_DATA_DICT_RAW'
#
# data_dict = get_lmps(start_date=start_date,
#                              end_date=end_date,
#                              previous_data_dict_name = save_directory+previous_data_dict_name,
#                             static_directory=static_directory,
#                             working_directory=working_directory,
#                              concat_old_dict=False)




####################################################################################################################
####################################################################################################################

# #Use this code to add the new tall temp data to the old temp data
# combine_df = add_tall_files_together(filename1=save_directory+'2019_12_09_BACKTEST_INPUT_FILE_YES_DATA_EXTRACT_TALL(Temps)',
#                                      filename2=save_directory+'2020_01_05_BACKTEST_INPUT_FILE_YES_DATA_EXTRACT_TALL(Temps)(newdata)',
#                                      output_save_name=save_directory+'2020_01_05_BACKTEST_INPUT_FILE_YES_DATA_EXTRACT_TALL(Temps)')

####################################################################################################################
####################################################################################################################



