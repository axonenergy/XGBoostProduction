from API_Lib import get_ISO_api_data
from API_Lib import get_lmps
from API_Lib import get_reference_prices
from API_Lib import add_tall_files_together
from API_Lib import get_spreads

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

start_date = '2020_02_20'   # 7/15/2015 start date
end_date = '2020_03_19'   # check here for most recent file for end date: https://marketplace.spp.org/pages/rtbm-lmp-by-location#

previous_data_dict_name = '2020_02_24_BACKTEST_DATA_DICT_RAW'

data_dict = get_ISO_api_data(start_date=start_date,
                             end_date=end_date,
                             previous_data_dict_name = save_directory+previous_data_dict_name,
                             static_directory=static_directory,
                             working_directory=working_directory,
                             concat_old_dict=True)



####################################################################################################################
#                                           API PULLS FOR HISTORIC DA LMP DATES                                          #
####################################################################################################################


start_date = '2020_02_20'   # 7/15/2015 start date
end_date = '2020_03_19' # check here for most recent file for end date: https://marketplace.spp.org/pages/rtbm-lmp-by-location#

previous_data_dict_name = '2020_02_24_LMP_DATA_DICT_MASTER'

data_dict = get_lmps(start_date=start_date,
                     end_date=end_date,
                     previous_data_dict_name = save_directory+previous_data_dict_name,
                     static_directory=static_directory,
                     working_directory=working_directory,
                     concat_old_dict=True)



####################################################################################################################
####################################################################################################################

#Use this code to get reference prices

data_dict_name = '2020_03_24_BACKTEST_DATA_DICT_RAW'

get_reference_prices(data_dict_name=data_dict_name,
                     working_directory=working_directory,
                     static_directory=static_directory)



####################################################################################################################
####################################################################################################################
