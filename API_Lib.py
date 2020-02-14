import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse
import pickle as pickler
from sklearn.feature_selection import VarianceThreshold
import requests
import io
import time
import glob
import os
import pytz
import zipfile
import json
from pandas.io.json import json_normalize


pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 5000)
pd.set_option('max_row', 100)


######################################################################################################
################FUNCTIONS TO READ AND FORMAT DATA FROM APIS AND YES FILES#############################
######################################################################################################

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickler.dump(obj, f, pickler.HIGHEST_PROTOCOL)


def load_obj(file_name):
    with open(file_name + '.pkl', 'rb') as file:
        try:
            return pickler.load(file)
        except:
            print('File Does Not Exist')


def timezone_shift(input_datetime_df, date_col_name, input_tz, output_tz):
    ### Function that shifts time zones for an input dataframe and given input and output timezone.
    ### In the input dataframe, either the date column must have an hours componant, or there needs to be an HourEnding column to go along with the date column

    added_hourending = False
    if input_datetime_df.loc[12,date_col_name].hour >0:
        pass
    else:
        # try:
        input_datetime_df['HourEnding'] = pd.to_numeric(input_datetime_df['HourEnding'], errors='coerce')
        input_datetime_df[date_col_name] = input_datetime_df[date_col_name] + input_datetime_df['HourEnding'].apply(lambda he: datetime.timedelta(hours=he))
        added_hourending = True
        # except:
        #     print(input_datetime_df)
        #     input_datetime_df.to_csv('timezone_error.csv')
        #     print('ERROR: No hour-level detail found in either the Date column or an HourEnding column. Please revise input dataframe sent to timezone shift function.')
        #     exit()

    try:
        input_datetime_df.drop(columns=['HourEnding'],inplace=True)
        removed_hourending = True
    except:
        removed_hourending = False

    input_tz = input_tz.upper()
    output_tz = output_tz.upper()
    tz_dict = {'EST':pytz.timezone('Etc/GMT+5'),
               'CST': pytz.timezone('Etc/GMT+6'),
               'CPT':pytz.timezone('America/Chicago'),
               'EPT':pytz.timezone('America/Indianapolis'),
               'UTC':pytz.timezone('UTC')}

    input_tz = tz_dict[input_tz]
    output_tz = tz_dict[output_tz]

    input_datetime_df[date_col_name] = input_datetime_df[date_col_name].apply(lambda datestamp: input_tz.localize(datestamp))
    input_datetime_df[date_col_name] = input_datetime_df[date_col_name].apply(lambda datestamp: datestamp.astimezone(output_tz))
    input_datetime_df[date_col_name] = input_datetime_df[date_col_name].apply(lambda datestamp: datestamp.replace(tzinfo=None))

    if removed_hourending:
        hourending_df = input_datetime_df[date_col_name].apply(lambda datestamp: datestamp.hour)
        hourending_df.loc[hourending_df == 0] =24
        input_datetime_df.insert(1,'HourEnding',hourending_df)

    if added_hourending:
        hourending_df = input_datetime_df[date_col_name].apply(lambda datestamp: datestamp.hour)
        hourending_df.loc[hourending_df == 0] =24
        input_datetime_df[date_col_name] = input_datetime_df[date_col_name] - hourending_df.apply(lambda he: datetime.timedelta(hours=he))

    input_datetime_df = input_datetime_df.loc[:, ~input_datetime_df.columns.duplicated()]

    return input_datetime_df


def preprocess_data(input_dict, static_directory):
    save_directory = static_directory + '\ModelUpdateData\\Temperature_files\\'
    nan_limit_days = 45  # Must be this high in order to get ISONE points

    #### Removes columns with too much bad data or that dont have temperature normals
    print('Pre-processing data...')
    input_df = input_dict['EST']
    input_df = input_df.astype('float')

    ###drop and fill NA and blank columns
    input_df = input_df.replace(' ', np.nan)
    input_df = input_df.replace('', np.nan)

    initial_columns = len(input_df.columns)
    input_df.dropna(axis=1,thresh=len(input_df)-(nan_limit_days*24),inplace=True)
    new_columns = len(input_df.columns)
    print('Dropped ' + str(initial_columns-new_columns) + ' features (' + str(round(100 * (initial_columns-new_columns) / initial_columns, 2)) + '%) due NaN limit break of > '+str(nan_limit_days)+ ' days of data missing.')
    print('Initial Empty/Null Values: ' + str(input_df.isnull().sum().sum()))


    ### drop duplicate days (from timechanges)
    input_df = input_df.loc[~input_df.index.duplicated(keep='first')]


    # Remove Temperature Locations That Do Not Have Departure Data
    temp_df = input_df[[col for col in input_df.columns if 'FTEMP' in col]]
    all_normals = pd.read_csv(save_directory+'temp_normals_all.csv', index_col=['Month', 'Day', 'HE'])

        # Read In Normals Data And Format
    normals_df = all_normals.loc[:, temp_df.columns[1:].str.replace('_FTEMP', '_TNORM')]
    normals_df.columns = [col.replace('TNORM', 'FTEMP') for col in normals_df.columns]
    nan_by_col_norms = pd.DataFrame(normals_df.isnull().sum())
    drop_cols = list()
    for row in nan_by_col_norms.iterrows():
        if row[1].values > nan_limit_days * 24:
            normals_df.drop(columns=[row[0]], inplace=True)
            drop_cols.append(row[0])

    print('Dropped ' +str(len(drop_cols)) + ' temp locations due to lack of temperature normals data.')

        # Drop FTEMPS that dont have normals
    input_df = input_df[[col for col in input_df.columns if col not in drop_cols]]

    for key, value in input_dict.items():
        df= input_dict[key][input_df.columns]
        # Rename columns that dont have common names
        df.columns = [col.replace('TOTAL_RESOURCE_CAP_OUT', '_OUTAGE').replace('ISO_CAPACITY_OFFLINE', '_OUTAGE').replace('OUTAGES', 'OUTAGE') for col in df.columns]
        input_dict[key]=df

    return input_dict


def drop_correlated_data(input_dict):
    input_df = input_dict['EST']
    input_df = input_df.astype('float')

    ### drop correlated temperature points
    temp_df = input_df[[col for col in input_df.columns if 'FTEMP' in col]]
    correlated_features = set()
    corr_matrix = temp_df.corr()
    orig_cols = max(len(temp_df.columns), 0.00001)
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.975:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)
    input_df.drop(columns=correlated_features, inplace=True)
    print('Dropped Correlated Temp Points: ', correlated_features)
    print('Dropped ' + str(len(correlated_features)) + ' (' + str(
        round(100 * len(correlated_features) / orig_cols, 0)) + '%) correlated (>97.5%) TEMPERATURE locations.')

    ### drop correlated load points
    iso_dict_load = {'NYISO': 0.98,   ###Correlation cutoffs by ISO
                     'PJM': 0.98,
                     'MISO': 0.98,
                     'ERCOT': 0.98,
                     'SPP': 0.98,
                     'ISONE': 0.98}

    for iso, perc in iso_dict_load.items():
        temp_df = input_df[[col for col in input_df.columns if 'FLOAD' in col]]
        temp_df = temp_df[[col for col in temp_df.columns if iso in col]]
        temp_df.dropna(axis=0, inplace=True)
        correlated_features = set()
        corr_matrix = temp_df.corr()
        orig_cols = max(len(temp_df.columns), 0.00001)
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > perc:
                    colname = corr_matrix.columns[i]
                    correlated_features.add(colname)
        input_df.drop(columns=correlated_features, inplace=True)
        print(iso + ' Dropped Correlated Load Points: ', correlated_features)
        print(iso + ' Dropped ' + str(len(correlated_features)) + ' (' + str(
            round(100 * len(correlated_features) / orig_cols, 0)) + '%) correlated (>' + str(perc) + '%) LOAD locations.')

    count_df = input_df[[col for col in input_df.columns if 'FLOAD' in col]]
    for iso, perc in iso_dict_load.items():
        print('Remaining Load Zones in ' + iso + ':' + str(len([col for col in count_df.columns if iso in col])))

    ######## drop correlated dart points
    iso_dict_dart = {'NYISO': 0.995, #.90 ###Correlation cutoffs by ISO
                     'PJM': 0.98, #.95
                     'MISO': 0.94, #.90
                     'ERCOT': 0.995, #.998
                     'SPP': 0.98, #.96
                     'ISONE': 0.995} #.99

    for iso, perc in iso_dict_dart.items():
        temp_df = input_df[[col for col in input_df.columns if 'DART' in col]]
        temp_df = temp_df[[col for col in temp_df.columns if iso in col]]
        temp_df.dropna(axis=0, inplace=True)
        correlated_features = set()
        print('Running Corr Matrix ' + iso)
        corr_matrix = temp_df.corr()
        orig_cols = max(len(temp_df.columns), 0.00001)
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > perc:
                     colname = corr_matrix.columns[i]
                     correlated_features.add(colname)
        input_df.drop(columns=correlated_features, inplace=True)
        print(iso + ' Dropped ' + str(len(correlated_features)) + ' (' + str(
            round(100 * len(correlated_features) / orig_cols, 0)) + '%) correlated (>' + str(perc) + '%'') DART locations.')


    count_df = input_df[[col for col in input_df.columns if 'DART' in col]]
    for iso, perc in iso_dict_dart.items():
        print('Remaining Nodes in ' + iso + ':' + str(len([col for col in count_df.columns if iso in col])))

    for key, value in input_dict.items():
        input_dict[key]=input_dict[key][input_df.columns]

    return input_dict


def lag_data16_40(input_dict, col_type='DART', return_only_lagged = False):
    for timezone, dataframe in input_dict.items():
        temp_df = dataframe[[col for col in dataframe.columns if col_type in col]].copy()
        temp_df.reset_index(inplace=True)
        temp_df['Date'] = temp_df['Date'].astype('datetime64[ns]')
        temp_df.loc[temp_df['HourEnding']<=8, 'Date'] = temp_df['Date'] + datetime.timedelta(days=1)
        temp_df.loc[temp_df['HourEnding']> 8, 'Date'] = temp_df['Date'] + datetime.timedelta(days=2)
        temp_df.set_index(['Date', 'HourEnding'],inplace=True, drop=-True)
        temp_df.columns = [col.replace('DART','DA_RT_LAG') for col in temp_df.columns]

        if not return_only_lagged:
            output_df = dataframe.join(temp_df, how='inner', rsuffix='DELETE').sort_values(by=['Date','HourEnding'],ascending=True)
            input_dict[timezone] = output_df
        else:
            input_dict[timezone] = temp_df.sort_values(by=['Date','HourEnding'],ascending=True)

    return input_dict


def process_YES_data_extract_wide(input_df, input_timezone, start_date, end_date, nan_limit = 21*24):
    ####### THIS FUNCTION READS AND FORMATS ANY YES DATA EXTRACT EXPORT
    input_df['Date'] = input_df['Date'].astype('datetime64[ns]')
    start_date = parse(start_date)
    end_date = parse(end_date)
    output_dict_dataframes = {'EST': None, 'EPT': None, 'CPT': None}

    ### Make date and HE as dual index
    input_df = input_df.iloc[:,4:]
    input_df.set_index(['Date', 'HourEnding'], inplace=True)

    ### add identifying labels for data (FLOAD, TEMP, etc)
    for col in input_df.columns:
        descript = input_df[col][0]
        descript_ref = {'DARTLMP_SELL': 'DART',
                        'FORCTEMP': 'FTEMP',
                        'FORECASTLOAD': 'FLOAD',
                        'ACTUAL_TIE_FLOW': 'AFLOW',
                        'DALMP':'DALMP',
                        'RTLMP':'RTLMP'}
        new_col_name = col + '_' + descript_ref[descript]
        new_col_name = new_col_name.replace(' ', "")
        new_col_name = new_col_name.replace('/', "")
        new_col_name = new_col_name.replace('(', "")
        new_col_name = new_col_name.replace(')', "")
        try:
            input_df.rename(columns={col: new_col_name}, inplace=True)
        except:
            print(
                'Unexpected data type from data extract (forecast load, forecast temp, etc) - code needs to be updated.')

    input_df.columns = ['ERCOT_'+col for col in input_df.columns] ### Currently the only data coming through this feed is ERCOT data
    input_df = input_df.iloc[1:, :]

    input_df.dropna(axis=0,inplace=True)

    for output_timezone in ['EST', 'EPT', 'CPT']:
        timezone_input_df = input_df.copy()
        timezone_input_df.reset_index(inplace=True)
        timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                           date_col_name='Date',
                                           input_tz=input_timezone,
                                           output_tz=output_timezone)
        timezone_input_df = timezone_input_df[(timezone_input_df['Date']>=start_date) & (timezone_input_df['Date']<=end_date)]
        timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)
        input_df = input_df.loc[~input_df.index.duplicated(keep='first')]
        output_dict_dataframes[output_timezone] = timezone_input_df

    return output_dict_dataframes


def process_YES_data_extract_temps_tall(input_df, input_timezone, start_date, end_date, nan_limit = 21*24):
    start_date = parse(start_date)
    end_date = parse(end_date)
    output_dict_dataframes = {'EST': None, 'EPT': None, 'CPT': None}
    input_df = input_df.pivot(index='DateTime', columns='ObjectName', values='FORCTEMP')
    input_df.columns = input_df.columns.get_level_values(0)
    input_df.reset_index(inplace=True)
    input_df['DateTime'] = input_df['DateTime'].astype('datetime64[ns]')
    input_df['Date'] = input_df['DateTime'].apply(
        lambda datestamp: datetime.datetime(year=datestamp.year, month=datestamp.month, day=datestamp.day))
    input_df['HourEnding'] = input_df['DateTime'].dt.hour
    input_df.drop(columns=['DateTime'], inplace=True)
    input_df.loc[input_df['HourEnding'] == 0, 'HourEnding'] = 24
    input_df.loc[input_df['HourEnding'] == 24, 'Date'] = input_df['Date'] - datetime.timedelta(days=1)
    input_df.sort_values(['Date', 'HourEnding'], inplace=True)
    input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)

    input_df.columns = [col+'_FTEMP' for col in input_df.columns ]
    for string in ['/', ' ', '(', ')']:
        input_df.columns = [col.replace(string,'') for col in input_df.columns]

    for output_timezone in ['EST', 'EPT', 'CPT']:
        timezone_input_df = input_df.copy()
        timezone_input_df.reset_index(inplace=True)
        timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                           date_col_name='Date',
                                           input_tz=input_timezone,
                                           output_tz=output_timezone)
        timezone_input_df = timezone_input_df[
            (timezone_input_df['Date'] >= start_date) & (timezone_input_df['Date'] <= end_date)]
        timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)
        timezone_input_df = timezone_input_df.loc[~timezone_input_df.index.duplicated(keep='first')]
        output_dict_dataframes[output_timezone] = timezone_input_df

    return output_dict_dataframes


def process_YES_daily_price_tables(predict_date, input_timezone, working_directory, dart_only):
    input_files_directory = working_directory + '\InputFiles\\'
    alt_nodes_directory = working_directory + '\DailyTradeFiles\\'

    output_dict_dataframes = {'EST': None, 'EPT': None, 'CPT': None}


    T_0_date = predict_date - datetime.timedelta(days=1)
    T_1_date = predict_date - datetime.timedelta(days=2)
    T_2_date = predict_date - datetime.timedelta(days=3)
    predict_date_str_mm_dd_yyyy = predict_date.strftime('%m_%d_%Y')

    try:
        T_0_df = pd.read_excel(input_files_directory+predict_date_str_mm_dd_yyyy + '_DAILY_INPUT_YES_PRICE_TABLE_T-0DAY.xls')
    except:
        print('ERROR: '+input_files_directory+predict_date_str_mm_dd_yyyy + '_DAILY_INPUT_YES_PRICE_TABLE_T-0DAY.xls' + ' file is not found. Please make sure it is in the "InputFiles" directory and is named correctly')
        exit()

    try:
        T_1_df = pd.read_excel(input_files_directory+predict_date_str_mm_dd_yyyy + '_DAILY_INPUT_YES_PRICE_TABLE_T-1DAY.xls')
    except:
        print('ERROR: '+input_files_directory+
            predict_date_str_mm_dd_yyyy + '_DAILY_INPUT_YES_PRICE_TABLE_T-1DAY.xls' + ' file is not found. Please make sure it is in the "InputFiles" directory and is named correctly')
        exit()

    try:
        T_2_df = pd.read_excel(input_files_directory+predict_date_str_mm_dd_yyyy + '_DAILY_INPUT_YES_PRICE_TABLE_T-2DAY.xls')
    except:
        print('ERROR: '+input_files_directory+
            predict_date_str_mm_dd_yyyy + '_DAILY_INPUT_YES_PRICE_TABLE_T-2DAY.xls' + ' file is not found. Please make sure it is in the "InputFiles" directory and is named correctly')
        exit()

    try:
        node_alt_names_df = pd.read_excel(alt_nodes_directory+'NodeAlternateNames.xlsx')
    except:
        print('ERROR: '+alt_nodes_directory+'NodeAlternateNames.csv' + ' file is not found. Please make sure it is in the "InputFiles" directory and is named correctly')
        exit()

    node_alt_names_dict = dict(zip(node_alt_names_df['YES_Name_With_ISO'],node_alt_names_df['Model_Name']))

    T_0_df = format_pricetable_df(pricetable_df=T_0_df, date=T_0_date, dart_only=dart_only, node_alt_names_dict = node_alt_names_dict)
    T_1_df = format_pricetable_df(pricetable_df=T_1_df, date=T_1_date, dart_only=dart_only, node_alt_names_dict = node_alt_names_dict)
    T_2_df = format_pricetable_df(pricetable_df=T_2_df, date=T_2_date, dart_only=dart_only, node_alt_names_dict = node_alt_names_dict)

    pricetable_df = pd.concat([T_2_df,T_1_df,T_0_df])
    pricetable_df.fillna(method='ffill', inplace=True)

    for output_timezone in ['EST', 'EPT', 'CPT']:
        timezone_input_df = pricetable_df.copy()
        timezone_input_df.reset_index(inplace=True)
        timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                           date_col_name='Date',
                                           input_tz=input_timezone,
                                           output_tz=output_timezone)
        timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)
        timezone_input_df = timezone_input_df.loc[~timezone_input_df.index.duplicated(keep='first')]
        output_dict_dataframes[output_timezone] = timezone_input_df

    return output_dict_dataframes


def format_pricetable_df(pricetable_df, date, dart_only,node_alt_names_dict):
    pricetable_df.rename(columns={'Price Node':'Node'},inplace=True)
    pricetable_df['Stat'] = pricetable_df['Stat'].str.replace(' ','').str.upper()
    pricetable_df.drop(columns=['24HR','Peak','WD Peak','WE Peak','Off','Avg'],inplace=True)
    pricetable_df['ISO'] = pricetable_df['ISO'].replace('SPPISO', 'SPP')
    pricetable_df['ISO'] = pricetable_df['ISO'].replace('PJMISO', 'PJM')
    pricetable_df['Node'] = pricetable_df['ISO'] + '_' + pricetable_df['Node']
    pricetable_df['Node'] = pricetable_df['Node'].apply(lambda node: node_alt_names_dict[node])
    pricetable_df['Node'] = pricetable_df['Node'] + '_'+pricetable_df['Stat']
    pricetable_df = pricetable_df.drop(columns=['ISO','Stat'])
    pricetable_df.set_index('Node',inplace=True)
    pricetable_df = pricetable_df.T
    pricetable_df['Date'] = date
    pricetable_df['Date'] = pricetable_df['Date'].astype('datetime64[ns]')
    pricetable_df.reset_index(inplace=True)
    pricetable_df.rename(columns={'index': 'HourEnding'}, inplace=True)
    pricetable_df.set_index(['Date', 'HourEnding'], drop=True, inplace=True)

    if dart_only:
        pricetable_df = pricetable_df[[col for col in pricetable_df if '_DART' in col]]

    return pricetable_df


def add_tall_files_together(filename1,filename2,output_save_name):
    file1_df = pd.read_csv(filename1+'.csv', index_col=['ObjectId', 'DateTime'])
    file2_df = pd.read_csv(filename2+'.csv', index_col=['ObjectId', 'DateTime'])
    file3_df = pd.concat([file1_df,file2_df], axis=0)
    file3_df = file3_df.loc[~file3_df.index.duplicated(keep='first')]
    file3_df.reset_index(inplace=True)
    file3_df.to_csv(output_save_name+'.csv')


def process_YES_timeseries(input_df, input_timezone, start_date, end_date, nan_limit = 21*24):
    ####### THIS FUNCTION READS AND FORMATS ANY YES TIMESERIES EXPORT
    output_dict_dataframes = {'EST': None, 'EPT': None, 'CPT': None}
    start_date = parse(start_date)
    end_date = parse(end_date)
    input_df['Date/Time'] = input_df['Date/Time'].astype('datetime64[ns]')
    input_df['HourEnding'] = input_df['Date/Time'].dt.hour
    input_df['Date'] = input_df['Date/Time'].apply(lambda datestamp: datetime.date(datestamp.year, datestamp.month, datestamp.day))
    input_df.loc[input_df['HourEnding'] == 0, 'HourEnding'] = 24
    input_df.loc[input_df['HourEnding'] == 24, 'Date'] = input_df['Date'] - datetime.timedelta(days=1)

    input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)
    input_df.drop(columns=['Date/Time'], inplace=True)

    ### add identifying labels for data (FLOAD, TEMP, etc)
    descript_ref = {' ' : "",
                    '/' : "",
                    '(': "",
                    ')': "",
                    'ISO_CAPACITY_OFFLINEAverageLatest': '_OUTAGE',
                    'TOTAL_RESOURCE_CAP_OUTAverageLatest': '_OUTAGE',
                    'AverageYesterday0:null':'',
                    'WIND_STWPF_ORIGAverage': '_WIND_STWPF',
                    'WIND_WGRPP_ORIGAverage': '_WIND_WGRPP',
                    'WIND_STWPFAverageLatest': '_WIND_STWPF',
                    'WIND_WGRPPAverageLatest': '_WIND_WGRPP',
                    'BIDCLOSE_LOAD_FORECASTAverage': '_FLOAD',
                    'LOAD_FORECASTAverageLatest': '_FLOAD',
                    'TEMP_NORMAverage': '_TNORM',
                    'FORCTEMP_FAverageLatest' : '_FTEMP'
                    }

    PJM_list = ['AECO','AEP','ATSI','BGE','COMED','DAYTON','DEOK','DOMINION','DPL',
                'DUQUESNE','EKPCPJMISO','JCPL','METED','MID-ATLANTICREGION','PECO','PENELEC','PEPCO',
                'PPL','PSEG','RECO','RTOCOMBINED','SOUTHERNREGION','UGI','WESTERNREGION']

    for orig, replace in descript_ref.items():
        input_df.columns = input_df.columns.str.replace(orig, replace)

    input_df.columns = [col.replace('WZ','ERCOT') for col in input_df.columns]
    for pjm_load in PJM_list:
        input_df.columns = [col.replace(pjm_load,'PJM_'+pjm_load) for col in input_df.columns]

    for output_timezone in ['EST', 'EPT', 'CPT']:
        timezone_input_df = input_df.copy()
        timezone_input_df.reset_index(inplace=True)
        timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                           date_col_name='Date',
                                           input_tz=input_timezone,
                                           output_tz=output_timezone)
        timezone_input_df = timezone_input_df[
            (timezone_input_df['Date'] >= start_date) & (timezone_input_df['Date'] <= end_date)]
        timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)
        output_dict_dataframes[output_timezone] = timezone_input_df
        input_df = input_df.loc[~input_df.index.duplicated(keep='first')]

    return output_dict_dataframes


def get_MISO_load_outage(start_date, end_date):
    ####### THIS FUNCTION RETURNS CURRENT 7-DAY MISO OUTAGE FORECAST FROM THE MISO API
    ####### ALL DAYS PAST THE FORECAST DAY ARE TRUNCATED AT THE END OF THE FUNCION - THEY MAY BE REINCLUDED BY NOT TRUNCATING THEM
    output_dict_dataframes = {'EST':None,'EPT':None,'CPT':None}
    start_date = parse(start_date)
    end_date = parse(end_date)
    end_date = end_date - datetime.timedelta(days=1)
    start_date = start_date - datetime.timedelta(days=1)
    common_url = 'https://docs.misoenergy.org/marketreports/'
    api_name = '_sr_la_rg'
    file_date = start_date

    while file_date <= end_date:
        string_date = file_date.strftime('%Y%m%d')
        print('MISO Load/Outage Data Processing for Date: ' + string_date)
        try:
            new_file_df = pd.read_csv(common_url+string_date+api_name+'.csv', skiprows=3)
            new_file_df['Date'] = file_date + datetime.timedelta(days=1)
            new_file_df['Hourend_EST'] = pd.to_numeric(new_file_df['Hourend_EST'].str.replace('Hour   ',''), errors='coerce')
            new_file_df = new_file_df.rename(columns = {'Hourend_EST':'HourEnding'})
            new_file_df = new_file_df.dropna()
            new_file_df.set_index(['Date', 'HourEnding', 'Region'], drop=True, inplace=True)

            new_columns = []

            for day in range(0,7):
                new_columns.append((file_date+datetime.timedelta(days=day),'FLOAD_Day_'+str(day)))
                new_columns.append((file_date+datetime.timedelta(days=day),'OUTAGE_Day_'+str(day)))
            new_file_df.columns = pd.MultiIndex.from_tuples(new_columns)
            new_file_df = new_file_df.swaplevel(axis=1)
            new_file_df = new_file_df.stack()
            new_file_df.fillna(-10000, inplace=True)
            new_file_df['FLOAD'] = new_file_df[[col for col in new_file_df.columns if 'FLOAD' in col]].apply(lambda row: row.max(), axis=1)
            new_file_df['OUTAGE'] = new_file_df[[col for col in new_file_df.columns if 'OUTAGE' in col]].apply(lambda row: row.max(), axis=1)
            new_file_df = new_file_df[[col for col in new_file_df.columns if 'Day' not in col]]
            new_file_df.reset_index(inplace=True)
            new_file_df.drop(columns=['Date'],inplace=True)
            new_file_df.rename(columns = {'level_3':'Date'}, inplace=True)
            new_file_df.set_index(['Date','HourEnding'], inplace=True, drop=True)
            new_file_df.sort_values(['Date', 'HourEnding'], inplace=True)
            new_file_df.loc[new_file_df['Region']=='MISO', 'Region'] = 'TOTAL'

            unstacked_file_df = pd.DataFrame()
            for region in new_file_df['Region'].unique():
                regional_df = new_file_df[new_file_df['Region']==region].copy()
                regional_df.drop(columns='Region', inplace=True)
                regional_df.columns = ['MISO_' + region + '_' + str(col) for col in regional_df.columns]
                if unstacked_file_df.empty: unstacked_file_df = regional_df
                else: unstacked_file_df = unstacked_file_df.merge(regional_df, on=['Date', 'HourEnding'])


            for output_timezone in ['EST', 'EPT', 'CPT']:
                timezone_input_df = unstacked_file_df.copy()
                timezone_input_df.reset_index(inplace=True)
                timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                                   date_col_name='Date',
                                                   input_tz='EST',
                                                   output_tz=output_timezone)

                timezone_input_df  = timezone_input_df[(timezone_input_df['Date']>=file_date+datetime.timedelta(days=1))&(timezone_input_df['Date']<=file_date+datetime.timedelta(days=2))] ## Only return the next days' forecast
                timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)

                df = output_dict_dataframes[output_timezone]
                if output_timezone not in output_dict_dataframes.keys():
                    df = timezone_input_df
                else:
                    df = pd.concat([df,timezone_input_df], axis=0)
                    df = df.loc[~df.index.duplicated(keep='last')]
                output_dict_dataframes[output_timezone] = df

        except:
            print('MISO Load/Outage File for date '+ string_date + ' not available on website')

        file_date = file_date + datetime.timedelta(days=1)

    return output_dict_dataframes


def get_MISO_LMPs(start_date, end_date, static_directory):
    save_directory = static_directory + '\ModelUpdateData\\MISO_files\\'
    ####### THIS FUNCTION RETURNS MISO DART
    ####### ALL DAYS PAST THE FORECAST DAY ARE TRUNCATED AT THE END OF THE FUNCION - THEY MAY BE REINCLUDED BY NOT TRUNCATING THEM
    raw_dict_LMP_dataframes = dict()
    arch_data = pd.DataFrame()
    output_dict_DALMP_dataframes = {'EST':None,'EPT':None,'CPT':None}
    output_dict_RTLMP_dataframes = {'EST':None,'EPT':None,'CPT':None}
    output_dict_DART_dataframes = {'EST':None,'EPT':None,'CPT':None}
    start_date = parse(start_date)
    end_date = parse(end_date)
    api_dict = {'RT':'_rt_lmp_final',
                'DA':'_da_expost_lmp'}
    common_url = 'https://docs.misoenergy.org/marketreports/'

    for lmp_type, api_name in api_dict.items():
        file_date = start_date
        if file_date < datetime.datetime(2017, 1, 1): file_date = datetime.datetime(2017, 1, 1)
        temp_df = pd.DataFrame()

        while file_date <= end_date:
            string_date = file_date.strftime('%Y%m%d')
            print('MISO '+lmp_type+' Data Processing for Date: ' + string_date)
            try:
                try: #try to get final LMP data
                    new_file_df = pd.read_csv(common_url+string_date+api_name+'.csv', skiprows=4)
                except: # if no final data get prelim data
                    new_file_df = pd.read_csv(common_url + string_date + '_rt_lmp_prelim' + '.csv', skiprows=4)

                new_file_df.columns = [col.replace('HE ','') for col in new_file_df.columns]
                new_file_df.drop(columns = ['Type','Value'],inplace=True)
                new_file_df.set_index('Node',inplace=True, drop=True)
                new_file_df = new_file_df.T
                new_file_df['Date'] = file_date
                new_file_df['Date'] = new_file_df['Date'].astype('datetime64[ns]')
                new_file_df.reset_index(inplace=True)
                new_file_df.rename(columns={'index':'HourEnding'},inplace=True)
                new_file_df.set_index(['Date','HourEnding'],drop=True, inplace=True)
                del new_file_df.columns.name
                new_file_df.columns = ['MISO_' + col+'_'+lmp_type+'LMP' for col in new_file_df.columns]
                for string in ['/', ' ', '(', ')']:
                    new_file_df.columns = [col.replace(string, '') for col in new_file_df.columns]
                new_file_df = new_file_df.loc[:, ~new_file_df.columns.duplicated()]

                if temp_df.empty:
                    temp_df = new_file_df
                    raw_dict_LMP_dataframes[lmp_type] = temp_df
                else:
                    temp_df = pd.concat([temp_df, new_file_df], axis=0)
                    raw_dict_LMP_dataframes[lmp_type] = temp_df

            except:
                print('MISO LMP File for date '+ string_date + ' not available on website')

            file_date = file_date + datetime.timedelta(days=1)
    try:
        if start_date < datetime.datetime(2017, 1, 1):  # need to get archived data
            print('Loading Archive MISO LMP Data...')
            arch_data_rt = pd.read_csv(save_directory+'miso_RTLMP_archive_2015-2016.csv').reset_index(drop=True)
            arch_data_rt['Date'] = arch_data_rt['Date'].astype('datetime64[ns]')
            arch_data_rt = arch_data_rt[(arch_data_rt['Date'] >= start_date) & (arch_data_rt['Date'] <= end_date)]
            arch_data_rt.set_index(['Date', 'HourEnding'], inplace=True, drop=True)
            arch_data_da = pd.read_csv(save_directory+'miso_DALMP_archive_2015-2016.csv').reset_index(drop=True)
            arch_data_da['Date'] = arch_data_da['Date'].astype('datetime64[ns]')
            arch_data_da = arch_data_da[(arch_data_da['Date'] >= start_date) & (arch_data_da['Date'] <= end_date)]
            arch_data_da.set_index(['Date', 'HourEnding'], inplace=True, drop=True)
            if bool(raw_dict_LMP_dataframes)==False:
                raw_dict_LMP_dataframes['RT'] = arch_data_rt
                raw_dict_LMP_dataframes['DA'] = arch_data_da
            else:
                raw_dict_LMP_dataframes['RT'] = pd.concat([arch_data_rt,raw_dict_LMP_dataframes['RT'].copy()])
                raw_dict_LMP_dataframes['DA'] = pd.concat([arch_data_da,raw_dict_LMP_dataframes['DA'].copy()])

        rt_lmp_df = raw_dict_LMP_dataframes['RT'].copy()
        da_lmp_df = raw_dict_LMP_dataframes['DA'].copy()
        rt_lmp_df.columns = [col.replace('RTLMP','') for col in rt_lmp_df.columns]
        da_lmp_df.columns = [col.replace('DALMP', '') for col in da_lmp_df.columns]
        dart_df = da_lmp_df-rt_lmp_df
        dart_df.columns = [col+'DART' for col in dart_df.columns]
        raw_dict_LMP_dataframes['DART'] = dart_df

        for key in raw_dict_LMP_dataframes.keys():
            for output_timezone in ['EST', 'EPT', 'CPT']:
                timezone_input_df = raw_dict_LMP_dataframes[key].copy()
                timezone_input_df.reset_index(inplace=True)
                timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                                   date_col_name='Date',
                                                   input_tz='EST',
                                                   output_tz=output_timezone)

                timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)

                if output_timezone not in output_dict_DART_dataframes.keys():
                    if key == 'DA': output_dict_DALMP_dataframes[output_timezone] = timezone_input_df
                    elif key == 'RT': output_dict_RTLMP_dataframes[output_timezone] = timezone_input_df
                    elif key == 'DART': output_dict_DART_dataframes[output_timezone] = timezone_input_df

                else:
                    if key == 'DA': output_dict_DALMP_dataframes[output_timezone]  = pd.concat([output_dict_DALMP_dataframes[output_timezone], timezone_input_df], axis=0)
                    elif key == 'RT': output_dict_RTLMP_dataframes[output_timezone]  = pd.concat([output_dict_RTLMP_dataframes[output_timezone], timezone_input_df], axis=0)
                    elif key == 'DART': output_dict_DART_dataframes[output_timezone]  = pd.concat([output_dict_DART_dataframes[output_timezone], timezone_input_df], axis=0)
    except:
            print('No MISO LMP Data for Timeframe Selected.')
    return output_dict_DALMP_dataframes, output_dict_RTLMP_dataframes, output_dict_DART_dataframes


def get_SPP_wind(start_date, end_date, static_directory):
    save_directory = static_directory + '\ModelUpdateData\\SPP_files\\'
    ####### THIS FUNCTION RETURNS CURRENT 7-DAY SPP WIND FORECAST FROM THE SPP API
    output_dict_dataframes = {'EST':None,'EPT':None,'CPT':None}
    start_date = parse(start_date)-datetime.timedelta(days=1)
    end_date = parse(end_date)-datetime.timedelta(days=1)
    common_url = 'https://marketplace.spp.org/file-browser-api/download/'
    api_name = 'midterm-resource-forecast?path='
    file_date = start_date

    while file_date <= end_date:
        string_date = file_date.strftime('%Y%m%d')
        string_year = file_date.strftime('%Y')
        string_month = file_date.strftime('%m')
        string_day = file_date.strftime('%d')
        print('SPP Wind Data Processing for Date: ' + string_date)

        try:
            if file_date>=datetime.datetime(2018, 1, 1):  ##Get data from SPP API
                unique_url = '%2F' + string_year + '%2F' + string_month + '%2F' + string_day + '%2FOP-MTRF-' + string_date + '0800.csv'
                url = common_url + api_name + unique_url
                new_file_df = pd.read_csv(url)
            else:  ### Get data from SPP downloaded zip folders
                try:
                    unique_url = save_directory + string_year + '_spp_mtrf/' + string_year + '/' + string_month + '/' + string_day + '/OP-DAWF-' + string_date + '0800.csv'
                    new_file_df = pd.read_csv(unique_url)
                    new_file_df.rename(columns={'Forecast MW': 'Wind Forecast MW'},inplace=True)
                except:
                    unique_url = save_directory + string_year + '_spp_mtrf/' + string_year + '/' + string_month + '/' + string_day + '/OP-MTRF-' + string_date + '0800.csv'
                    new_file_df = pd.read_csv(unique_url)
                    new_file_df.rename(columns={'Forecast MW': 'Wind Forecast MW'},inplace=True)

            new_file_df = new_file_df[['Wind Forecast MW','Interval']]
            new_file_df.rename(columns={'Wind Forecast MW':'SPP_FWIND'}, inplace=True)
            new_file_df['Interval'] = new_file_df['Interval'].astype('datetime64[ns]')
            new_file_df['Date'] = new_file_df['Interval'].apply(lambda datestamp: datetime.datetime(datestamp.year, datestamp.month, datestamp.day))
            new_file_df['HourEnding'] = new_file_df['Interval'].dt.hour
            new_file_df.loc[new_file_df['HourEnding']==0, 'HourEnding'] = 24
            new_file_df.loc[new_file_df['HourEnding']==24, 'Date'] = new_file_df['Date'] - datetime.timedelta(days=1)
            new_file_df.drop(columns=['Interval'], inplace=True)
            new_file_df.dropna(axis=0, inplace=True)
            new_file_df.sort_values(by=['Date', 'HourEnding'], ascending=True, inplace=True)
            new_file_df = new_file_df[new_file_df['Date']>=file_date]
            new_file_df = new_file_df[new_file_df['Date']<= file_date+datetime.timedelta(days=2)]
            new_file_df.set_index(['Date','HourEnding'], drop=True, inplace=True)

            for output_timezone in ['EST', 'EPT', 'CPT']:
                timezone_input_df = new_file_df.copy()
                timezone_input_df.reset_index(inplace=True)
                timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                                   date_col_name='Date',
                                                   input_tz='CPT',
                                                   output_tz=output_timezone)

                timezone_input_df = timezone_input_df[(timezone_input_df['Date'] >= file_date + datetime.timedelta(days=1)) & (timezone_input_df['Date'] <= file_date + datetime.timedelta(days=2))]  ## Only return the next days' forecast
                timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)

                df = output_dict_dataframes[output_timezone]
                if output_timezone not in output_dict_dataframes.keys():
                    df = timezone_input_df
                else:
                    df = pd.concat([df, timezone_input_df], axis=0)
                    df = df.loc[~df.index.duplicated(keep='last')]
                output_dict_dataframes[output_timezone] = df
        except:
            print('SPP Wind File for date '+ string_date + ' not available on website')

        file_date = file_date + datetime.timedelta(days=1)

    return output_dict_dataframes


def get_SPP_load(start_date, end_date,static_directory):
    save_directory = static_directory + '\ModelUpdateData\\SPP_files\\'
    ####### THIS FUNCTION RETURNS CURRENT 7-DAY SPP WIND FORECAST FROM THE SPP API
    output_dict_dataframes = {'EST':None,'EPT':None,'CPT':None}
    start_date = parse(start_date)-datetime.timedelta(days=1)
    end_date = parse(end_date)-datetime.timedelta(days=1)
    common_url = 'https://marketplace.spp.org/file-browser-api/download/'
    api_name = 'mtlf-vs-actual?path='
    file_date = start_date

    while file_date <= end_date:
        string_date = file_date.strftime('%Y%m%d')
        string_year = file_date.strftime('%Y')
        string_month = file_date.strftime('%m')
        string_day = file_date.strftime('%d')
        print('SPP Load Data Processing for Date: ' + string_date)

        try:
            if file_date >= datetime.datetime(2018, 1, 1):  ##Get data from SPP API
                unique_url = '%2F' + string_year + '%2F' + string_month + '%2F' + string_day + '%2FOP-MTLF-' + string_date + '0800.csv'
                url = common_url + api_name + unique_url
                new_file_df = pd.read_csv(url)

            else:  ### Get data from SPP downloaded zip folders
                unique_url = save_directory + string_year + '_spp_load/' + string_year + '/' + string_month + '/' + string_day + '/OP-MTLF-' + string_date + '0800.csv'
                new_file_df = pd.read_csv(unique_url)

            new_file_df.drop(columns=['GMTIntervalEnd', 'Averaged Actual'], inplace=True)
            new_file_df.rename(columns={'MTLF':'SPP_FLOAD'}, inplace=True)
            new_file_df['Interval'] = new_file_df['Interval'].astype('datetime64[ns]')
            new_file_df['Date'] = new_file_df['Interval'].apply(lambda datestamp: datetime.datetime(datestamp.year, datestamp.month, datestamp.day))
            new_file_df['HourEnding'] = new_file_df['Interval'].dt.hour
            new_file_df.loc[new_file_df['HourEnding']==0, 'HourEnding'] = 24
            new_file_df.loc[new_file_df['HourEnding']==24, 'Date'] = new_file_df['Date'] - datetime.timedelta(days=1)
            new_file_df.drop(columns=['Interval'], inplace=True)
            new_file_df.dropna(axis=0, inplace=True)
            new_file_df.sort_values(by=['Date', 'HourEnding'], ascending=True, inplace=True)
            new_file_df = new_file_df[new_file_df['Date']>=file_date]
            new_file_df = new_file_df[new_file_df['Date']<= file_date+datetime.timedelta(days=2)]
            new_file_df.set_index(['Date','HourEnding'], drop=True, inplace=True)

            for output_timezone in ['EST', 'EPT', 'CPT']:
                timezone_input_df = new_file_df.copy()
                timezone_input_df.reset_index(inplace=True)
                timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                                   date_col_name='Date',
                                                   input_tz='CPT',
                                                   output_tz=output_timezone)

                timezone_input_df = timezone_input_df[(timezone_input_df['Date'] >= file_date + datetime.timedelta(days=1)) & (timezone_input_df['Date'] <= file_date + datetime.timedelta(days=2))]  ## Only return the next days' forecast
                timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)

                df = output_dict_dataframes[output_timezone]
                if output_timezone not in output_dict_dataframes.keys():
                    df = timezone_input_df
                else:
                    df = pd.concat([df, timezone_input_df], axis=0)
                    df = df.loc[~df.index.duplicated(keep='last')]
                output_dict_dataframes[output_timezone] = df
        except:
            print('SPP Load Forecast File for date '+ string_date + ' not available on website')

        file_date = file_date + datetime.timedelta(days=1)

    return output_dict_dataframes


def get_SPP_outage(start_date, end_date,static_directory):
    save_directory = static_directory + '\ModelUpdateData\\SPP_files\\'
    ####### THIS FUNCTION RETURNS CURRENT 7-DAY SPP OUTAGE FORECAST FROM THE SPP API
    output_dict_dataframes = {'EST':None,'EPT':None,'CPT':None}
    start_date = parse(start_date)-datetime.timedelta(days=1)
    end_date = parse(end_date)-datetime.timedelta(days=1)
    common_url = 'https://marketplace.spp.org/file-browser-api/download/'
    api_name = 'capacity-of-generation-on-outage?path='
    file_date = start_date

    while file_date <= end_date:
        string_date = file_date.strftime('%Y%m%d')
        string_year = file_date.strftime('%Y')
        string_month = file_date.strftime('%m')
        print('SPP Outage Data Processing for Date: ' + string_date)

        try:
            if file_date >= datetime.datetime(2018, 1, 1):  ##Get data from SPP API
                unique_url = '%2F'+string_year+'%2F'+string_month+'%2FCapacity-Gen-Outage-'+string_date+'.csv'
                url = common_url+api_name+unique_url
                new_file_df = pd.read_csv(url)

            else:  ### Get data from SPP downloaded zip folders
                unique_url = save_directory + string_year + '_spp_outage/' + string_year + '/' + string_month + '/Capacity-Gen-Outage-' + string_date + '.csv'
                new_file_df = pd.read_csv(unique_url)


            new_file_df = new_file_df[['Market Hour',' Outaged MW']]
            new_file_df.rename(columns={'Market Hour':'Interval',
                                        ' Outaged MW':'SPP_OUTAGE'
                                        }, inplace=True)
            new_file_df['Interval'] = new_file_df['Interval'].astype('datetime64[ns]')
            new_file_df['Date'] = new_file_df['Interval'].apply(lambda datestamp: datetime.datetime(datestamp.year, datestamp.month, datestamp.day))
            new_file_df['HourEnding'] = new_file_df['Interval'].dt.hour
            new_file_df.loc[new_file_df['HourEnding']==0, 'HourEnding'] = 24
            new_file_df.loc[new_file_df['HourEnding']==24, 'Date'] = new_file_df['Date'] - datetime.timedelta(days=1)
            new_file_df.drop(columns=['Interval'], inplace=True)
            new_file_df.dropna(axis=0, inplace=True)
            new_file_df.sort_values(by=['Date', 'HourEnding'], ascending=True, inplace=True)
            new_file_df = new_file_df[new_file_df['Date']>=file_date+datetime.timedelta(days=0)]
            new_file_df = new_file_df[new_file_df['Date']<=file_date+datetime.timedelta(days=2)]
            new_file_df.set_index(['Date','HourEnding'], drop=True, inplace=True)

            for output_timezone in ['EST', 'EPT', 'CPT']:
                timezone_input_df = new_file_df.copy()
                timezone_input_df.reset_index(inplace=True)
                timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                                   date_col_name='Date',
                                                   input_tz='CPT',
                                                   output_tz=output_timezone)

                timezone_input_df = timezone_input_df[(timezone_input_df['Date'] >= file_date + datetime.timedelta(days=1)) & (timezone_input_df['Date'] <= file_date + datetime.timedelta(days=2))]  ## Only return the next days' forecast
                timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)

                df = output_dict_dataframes[output_timezone]
                if output_timezone not in output_dict_dataframes.keys():
                    df = timezone_input_df
                else:
                    df = pd.concat([df, timezone_input_df], axis=0)
                    df = df.loc[~df.index.duplicated(keep='last')]
                output_dict_dataframes[output_timezone] = df

        except:
            print('SPP Outage File for date ' + string_date + ' not available on website')

        file_date = file_date + datetime.timedelta(days=1)

    return output_dict_dataframes


def get_SPP_LMPs(start_date, end_date,static_directory):
    save_directory = static_directory + '\ModelUpdateData\\SPP_files\\'
    ####### THIS FUNCTION RETURNS DA,RT, and DART PJM LMPS FROM THE PJM DATAMINER2 API
    raw_dict_LMP_dataframes = dict()
    output_dict_DALMP_dataframes = {'EST':None,'EPT':None,'CPT':None}
    output_dict_RTLMP_dataframes = {'EST':None,'EPT':None,'CPT':None}
    output_dict_DART_dataframes = {'EST':None,'EPT':None,'CPT':None}
    start_date = parse(start_date)
    end_date = parse(end_date)
    api_dict = {'DA': ['da-lmp-by-location','%2FBy_Day%2FDA-LMP-SL-','0100'],'RT': ['rtbm-lmp-by-location','%2FBy_Day%2FRTBM-LMP-DAILY-SL-','']
                }

    common_url = 'https://marketplace.spp.org/file-browser-api/download/'

    for lmp_type, api_name in api_dict.items():
        file_date = start_date
        temp_df = pd.DataFrame()

        while file_date <= end_date:
            string_date = file_date.strftime('%Y%m%d')
            string_year = file_date.strftime('%Y')
            string_month = file_date.strftime('%m')
            string_day = file_date.strftime('%d')
            print('SPP '+lmp_type+' Data Processing for Date: ' + string_date)
            try:
                if file_date >= datetime.datetime(2018, 1, 1):  ##Get data from SPP API
                    unique_url = '?path=%2F' + string_year + '%2F' + string_month + api_dict[lmp_type][1] + string_date + api_dict[lmp_type][2]+'.csv'
                    url = common_url + api_name[0] + unique_url
                    new_file_df = pd.read_csv(url)

                else:  ### Get data from SPP downloaded zip folders
                    if lmp_type== 'DA':
                        try:
                            unique_url = save_directory + string_year + '_spp_dalmp/' + string_year + '/' + string_month + '/By_Day/DA-LMP-SL-' + string_date + '0100.csv'
                            new_file_df = pd.read_csv(unique_url)
                        except:
                            unique_url = save_directory + string_year + '_spp_dalmp/' + string_year + '/' + string_month + '/DA-LMP-SL-' + string_date + '0100.csv'
                            new_file_df = pd.read_csv(unique_url)
                    else:
                        new_file_df = pd.DataFrame(columns=['Interval','GMTIntervalEnd','Settlement Location','Pnode','LMP','MLC','MCC','MEC'])
                        try:
                            unique_url = save_directory + string_year + '_spp_rtlmp/' + string_year + '/' + string_month + '/By_Day/RTBM-LMP-DAILY-SL-' + string_date + '.csv'
                            new_file_df = pd.read_csv(unique_url)
                        except:
                            directory = save_directory + string_year + '_spp_rtlmp/' + string_year + '/' + string_month + '/' + string_day +'/'
                            for file_name in glob.glob(directory+'*.csv'):
                                file = pd.read_csv(file_name)
                                new_file_df = pd.concat([new_file_df,file],axis=0)
                            new_file_df.rename(columns={'GMTIntervalEnd':'GMTInterval', 'Pnode':' PNODE Name','Settlement Location':' Settlement Location Name','LMP':' LMP'})
                            new_file_df.reset_index(inplace=True,drop=True)

                new_file_df.columns = [col.replace(" ","") for col in new_file_df.columns]
                new_file_df.columns = [col.replace("End", "") for col in new_file_df.columns]
                new_file_df.columns = [col.replace("Name", "") for col in new_file_df.columns]
                new_file_df.columns = [col.upper() for col in new_file_df.columns]
                new_file_df.drop(columns=['GMTINTERVAL', 'MLC','MCC','MEC','PNODE'], inplace=True)
                new_file_df.rename(columns={' LMP': lmp_type+'LMP'}, inplace=True)
                new_file_df['INTERVAL'] = new_file_df['INTERVAL'].astype('datetime64[ns]')+datetime.timedelta(hours=1, minutes=-1) #change from HB to HE
                new_file_df['Date'] = new_file_df['INTERVAL'].apply(
                    lambda datestamp: datetime.datetime(datestamp.year, datestamp.month, datestamp.day))
                new_file_df['HourEnding'] = new_file_df['INTERVAL'].dt.hour
                new_file_df.loc[new_file_df['HourEnding'] == 0, 'HourEnding'] = 24
                new_file_df.loc[new_file_df['HourEnding'] == 24, 'Date'] = new_file_df['Date'] - datetime.timedelta(days=1)
                new_file_df.drop(columns=['INTERVAL'], inplace=True)
                new_file_df.dropna(axis=0, inplace=True)
                new_file_df.sort_values(by=['Date', 'HourEnding'], ascending=True, inplace=True)
                new_file_df = new_file_df[new_file_df['Date'] >= file_date]
                new_file_df = new_file_df[new_file_df['Date'] <= file_date + datetime.timedelta(days=2)]
                new_file_df = new_file_df.groupby(['Date','HourEnding','SETTLEMENTLOCATION']).mean()
                new_file_df.reset_index(inplace=True)

                new_file_df = new_file_df.pivot_table(index=['Date','HourEnding'], columns='SETTLEMENTLOCATION', values='LMP')
                del new_file_df.columns.name
                new_file_df.columns = ['SPP_'+str(col)+'_'+lmp_type+'LMP' for col in new_file_df.columns]

                for string in ['/', ' ', '(', ')']:
                    new_file_df.columns = [col.replace(string, '') for col in new_file_df.columns]

                if temp_df.empty:
                    temp_df = new_file_df
                    raw_dict_LMP_dataframes[lmp_type] = temp_df
                else:
                    temp_df = pd.concat([temp_df, new_file_df], axis=0)
                    raw_dict_LMP_dataframes[lmp_type] = temp_df
            except:
                print('SPP LMP File for date ' + string_date + ' not available on website')

            file_date = file_date + datetime.timedelta(days=1)
    try:
        rt_lmp_df = raw_dict_LMP_dataframes['RT'].copy()
        da_lmp_df = raw_dict_LMP_dataframes['DA'].copy()
        rt_lmp_df.columns = [col.replace('RTLMP', '') for col in rt_lmp_df.columns]
        da_lmp_df.columns = [col.replace('DALMP', '') for col in da_lmp_df.columns]
        dart_df = da_lmp_df - rt_lmp_df
        dart_df.columns = [col + 'DART' for col in dart_df.columns]
        raw_dict_LMP_dataframes['DART'] = dart_df

        for key in raw_dict_LMP_dataframes.keys():
            for output_timezone in ['EST', 'EPT', 'CPT']:
                timezone_input_df = raw_dict_LMP_dataframes[key].copy()
                timezone_input_df.reset_index(inplace=True)
                timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                                   date_col_name='Date',
                                                   input_tz='CPT',
                                                   output_tz=output_timezone)

                timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)

                if output_timezone not in output_dict_DART_dataframes.keys():
                    if key == 'DA':
                        output_dict_DALMP_dataframes[output_timezone] = timezone_input_df
                    elif key == 'RT':
                        output_dict_RTLMP_dataframes[output_timezone] = timezone_input_df
                    elif key == 'DART':
                        output_dict_DART_dataframes[output_timezone] = timezone_input_df

                else:
                    if key == 'DA':
                        output_dict_DALMP_dataframes[output_timezone] = pd.concat(
                            [output_dict_DALMP_dataframes[output_timezone], timezone_input_df], axis=0)
                    elif key == 'RT':
                        output_dict_RTLMP_dataframes[output_timezone] = pd.concat(
                            [output_dict_RTLMP_dataframes[output_timezone], timezone_input_df], axis=0)
                    elif key == 'DART':
                        output_dict_DART_dataframes[output_timezone] = pd.concat(
                            [output_dict_DART_dataframes[output_timezone], timezone_input_df], axis=0)
    except:
        print('No SPP LMP Data for Timeframe Selected.')
    return output_dict_DALMP_dataframes, output_dict_RTLMP_dataframes, output_dict_DART_dataframes


def get_PJM_outage(start_date, end_date):
    ####### THIS FUNCTION RETURNS CURRENT 7-DAY PJM OUTAGE FORECAST FROM THE PJM DATAMINER2 API
    output_dict_dataframes = {'EST':None,'EPT':None,'CPT':None}
    start_date = parse(start_date) - datetime.timedelta(days=1)
    end_date = parse(end_date) - datetime.timedelta(days=1)
    root_url = 'https://api.pjm.com/api/v1/'
    common_params = '?download=True&rowCount=50000&startRow=1'
    format_url = 'format=csv'
    key = 'subscription-key=f1d14d3b4ccd4c48a79f3ad52cfcef58'
    feed_type = 'gen_outages_by_type'
    file_date = start_date
    output_df = pd.DataFrame()
    while file_date <= end_date:
        string_date = file_date.strftime('%m/%d/%Y')
        print('PJM Outage Data Processing for Date: ' + string_date)

        try:
            url = root_url+feed_type+common_params+'&forecast_execution_date_ept='+string_date+"&"+key+'&'+format_url
            request = requests.get(url).text
            new_file_df = pd.read_csv(io.StringIO(request))
            new_file_df['forecast_execution_date_ept'] = new_file_df['forecast_execution_date_ept'].astype('datetime64[ns]')
            new_file_df['forecast_date'] = new_file_df['forecast_date'].astype('datetime64[ns]')
            new_file_df.rename(columns = {'forecast_date':'ForecastDate'}, inplace=True)
            new_file_df.rename(columns={'region': 'Region',
                                        'total_outages_mw': 'OUTAGE',
                                        'ForecastDate':'Date'}, inplace=True)
            new_file_df.loc[new_file_df['Region'] == 'Western', 'Region'] = 'WEST'
            new_file_df.loc[new_file_df['Region'] == 'Mid Atlantic - Dominion', 'Region'] = 'MIDATL'
            new_file_df.loc[new_file_df['Region'] == 'PJM RTO', 'Region'] = 'TOTAL'
            new_file_df.drop(columns=['planned_outages_mw','maintenance_outages_mw','forced_outages_mw','forecast_execution_date_ept'], inplace=True)
            new_file_df = new_file_df[(new_file_df['Date']>=file_date+datetime.timedelta(days=1))&(new_file_df['Date']<=file_date+datetime.timedelta(days=2))]
            new_file_df.set_index(['Date'],drop=True, inplace=True)

            unstacked_file_df = pd.DataFrame()
            for region in new_file_df['Region'].unique():
                regional_df = new_file_df[new_file_df['Region']==region].copy()
                regional_df.drop(columns=['Region'], inplace=True)
                regional_df.rename(columns = {'OUTAGE':'PJM_'+region+'_OUTAGE'}, inplace=True)

                if unstacked_file_df.empty:
                    unstacked_file_df = regional_df
                else:
                    unstacked_file_df = unstacked_file_df.merge(regional_df, on=['Date'])

            unstacked_file_df = unstacked_file_df.append([unstacked_file_df]*23)
            unstacked_file_df.sort_values('Date',inplace=True)
            unstacked_file_df['HourEnding'] = list(range(1,25))+list(range(1,25))
            unstacked_file_df.reset_index(inplace=True)
            unstacked_file_df.set_index(['Date','HourEnding'], drop=True, inplace=True)
            if output_df.empty:
                output_df = unstacked_file_df
            else:
                output_df = pd.concat([output_df, unstacked_file_df], axis=0, sort=True)
                output_df = output_df.loc[~output_df.index.duplicated(keep='last')]


        except:
            print('PJM Outage File for date ' + string_date + ' not available on website')

        file_date = file_date + datetime.timedelta(days=1)

        for output_timezone in ['EST', 'EPT', 'CPT']:
            output_dict_dataframes[output_timezone] = output_df ### values for all timezones are the same since reporting is on a daily level

    return output_dict_dataframes


def get_PJM_load(start_date, end_date):
    ####### THIS FUNCTION RETURNS CURRENT 7-DAY PJM LOAD FORECAST FROM THE PJM DATAMINER2 API
    ####### THIS FUNCTION ONLY WORKS FOR THE CURRENT DAY FORECAST - HISTORIC FORECASTS ARE NOT AVAILABLE THROUGH THE PJM API
    ####### HISTORIC FORECASTS MUST COME FROM THE YES LOAD API (TIMESERIES - BIDCLOSE)
    ####### ALL DAYS PAST THE FORECAST DAY ARE TRUNCATED AT THE END OF THE FUNCION - THEY MAY BE REINCLUDED BY NOT TRUNCATING THEM

    start_date = parse(start_date)
    end_date = parse(end_date)

    root_url = 'https://api.pjm.com/api/v1/'
    common_params = '?download=True&rowCount=50000&startRow=1'
    format_url = 'format=csv'
    key = 'subscription-key=f1d14d3b4ccd4c48a79f3ad52cfcef58'
    feed_type = 'load_frcstd_7_day'
    output_dict_dataframes = {}

    file_date = start_date
    output_df = pd.DataFrame()
    while file_date <= end_date:

        string_date_start = file_date.strftime('%m/%d/%Y')
        string_date_end = (file_date+datetime.timedelta(days=7)).strftime('%m/%d/%Y')
        print('PJM Load Data Processing for Date: ' + string_date_start)

        url = root_url + feed_type + common_params + '&forecast_datetime_beginning_ept=' + string_date_start + 'to' + string_date_end + "&" + key + '&' + format_url
        request = requests.get(url).text
        new_file_df = pd.read_csv(io.StringIO(request))

        new_file_df['evaluated_at_datetime_ept'] = new_file_df['evaluated_at_datetime_ept'].astype('datetime64[ns]')
        new_file_df['forecast_datetime_ending_ept'] = new_file_df['forecast_datetime_ending_ept'].astype('datetime64[ns]')
        new_file_df['Date'] = new_file_df['forecast_datetime_ending_ept'].apply(lambda datestamp: datetime.datetime(datestamp.year, datestamp.month, datestamp.day))
        new_file_df['HourEnding'] = new_file_df['forecast_datetime_ending_ept'].dt.hour
        new_file_df.loc[new_file_df['HourEnding']==0, 'HourEnding'] = 24
        new_file_df.loc[new_file_df['HourEnding']==24, 'Date'] = new_file_df['Date'] - datetime.timedelta(days=1)

        new_file_df.drop(columns=['evaluated_at_datetime_utc','forecast_datetime_beginning_utc','forecast_datetime_beginning_ept','forecast_datetime_ending_utc', 'evaluated_at_datetime_ept','forecast_datetime_ending_ept'], inplace=True)

        new_file_df.rename(columns={'forecast_area': 'Region',
                                    'forecast_load_mw': 'FLOAD'}, inplace=True)
        new_file_df.loc[new_file_df['Region'] == 'AE/MIDATL', 'Region'] = 'AE'
        new_file_df.loc[new_file_df['Region'] == 'BG&E/MIDATL', 'Region'] = 'BGE'
        new_file_df.loc[new_file_df['Region'] == 'DP&L/MIDATL', 'Region'] = 'DPL'
        new_file_df.loc[new_file_df['Region'] == 'JCP&L/MIDATL', 'Region'] = 'JCPL'
        new_file_df.loc[new_file_df['Region'] == 'METED/MIDATL', 'Region'] = 'METED'
        new_file_df.loc[new_file_df['Region'] == 'MID_ATLANTIC_REGION', 'Region'] = 'MID_ATL_TOTAL'
        new_file_df.loc[new_file_df['Region'] == 'PECO/MIDATL', 'Region'] = 'PECO'
        new_file_df.loc[new_file_df['Region'] == 'PENELEC/MIDATL', 'Region'] = 'PENELEC'
        new_file_df.loc[new_file_df['Region'] == 'PEPCO/MIDATL', 'Region'] = 'PEPCO'
        new_file_df.loc[new_file_df['Region'] == 'PPL/MIDATL', 'Region'] = 'PPL'
        new_file_df.loc[new_file_df['Region'] == 'PSE&G/MIDATL', 'Region'] = 'PSEG'
        new_file_df.loc[new_file_df['Region'] == 'RECO/MIDATL', 'Region'] = 'RECO'
        new_file_df.loc[new_file_df['Region'] == 'RTO_COMBINED', 'Region'] = 'RTO_TOTAL'
        new_file_df.loc[new_file_df['Region'] == 'SOUTHERN_REGION', 'Region'] = 'SOUTH_TOTAL'
        new_file_df.loc[new_file_df['Region'] == 'UGI/MIDATL', 'Region'] = 'UGI'
        new_file_df.loc[new_file_df['Region'] == 'WESTERN_REGION', 'Region'] = 'WEST_TOTAL'
        new_file_df = new_file_df[new_file_df['Date']>=file_date+datetime.timedelta(days=0)]
        new_file_df = new_file_df[new_file_df['Date']<file_date+datetime.timedelta(days=2)]

        new_file_df.set_index(['Date','HourEnding'], drop=True, inplace=True)


        unstacked_file_df = pd.DataFrame()
        for region in new_file_df['Region'].unique():
            regional_df = new_file_df[new_file_df['Region'] == region].copy()
            regional_df.drop(columns='Region', inplace=True)
            regional_df.columns = ['PJM_' + region + '_' + str(col) for col in regional_df.columns]
            if unstacked_file_df.empty:
                unstacked_file_df = regional_df
            else:
                unstacked_file_df = unstacked_file_df.merge(regional_df, on=['Date', 'HourEnding'])

        if output_df.empty:
            output_df = unstacked_file_df
        else:
            output_df = pd.concat([output_df, unstacked_file_df], axis=0, sort=True)
            output_df = output_df.loc[~output_df.index.duplicated(keep='last')]

        file_date = file_date + datetime.timedelta(days=1)

    for output_timezone in ['EST', 'EPT', 'CPT']:
        timezone_input_df = output_df.copy()
        timezone_input_df.reset_index(inplace=True)
        timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                           date_col_name='Date',
                                           input_tz='EST',
                                           output_tz=output_timezone)

        timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)

        if output_timezone not in output_dict_dataframes.keys():
            output_dict_dataframes[output_timezone] = timezone_input_df


    return output_dict_dataframes


def get_PJM_LMPs(start_date, end_date, static_directory):
    save_directory = static_directory + '\ModelUpdateData\\'
    ####### THIS FUNCTION RETURNS DA,RT, and DART PJM LMPS FROM THE PJM DATAMINER2 API
    raw_dict_LMP_dataframes = dict()
    output_dict_DALMP_dataframes = {'EST':None,'EPT':None,'CPT':None}
    output_dict_RTLMP_dataframes = {'EST':None,'EPT':None,'CPT':None}
    output_dict_DART_dataframes = {'EST':None,'EPT':None,'CPT':None}
    start_date = parse(start_date)
    end_date = parse(end_date)
    api_dict = {'RT':'rt_hrl_lmps',
                'DA':'da_hrl_lmps'}

    tradeable_nodes_df = pd.read_excel(save_directory+'valid-bidding-locations-for-virtual-bids.xls', skiprows=2)
    tradeable_nodes_df = pd.DataFrame(tradeable_nodes_df['PNODEID'])

    tradeable_nodes = ''
    for row in range(0,len(tradeable_nodes_df)):
        tradeable_nodes = tradeable_nodes + (str(tradeable_nodes_df.iloc[row,0])) + ';'
    tradeable_nodes = tradeable_nodes[:-1]

    root_url = 'https://api.pjm.com/api/v1/'
    key = 'f1d14d3b4ccd4c48a79f3ad52cfcef58'
    sFields_DA = ['pnode_id', 'datetime_beginning_ept', 'total_lmp_da']
    sFields_RT = ['pnode_id', 'datetime_beginning_ept', 'total_lmp_rt']
    headers = {'Content-Type': 'application/json'}


    for lmp_type, api_name in api_dict.items():
        file_date = start_date
        temp_df = pd.DataFrame()


        while file_date <= end_date:
            try:
                string_date_start = file_date.strftime('%m/%d/%Y')
                url = root_url + api_name + '?subscription-key=' + key + '&format=json'
                if lmp_type=='RT': sFields=sFields_RT
                else: sFields = sFields_DA
                print('PJM ' + lmp_type + 'LMP Data Processing for Date: ' + string_date_start)

                new_file_df=pd.DataFrame()

                try:
                    data = {'rowCount':50000,
                            'startRow':1,
                            'fields': sFields,
                            'filters': [{'datetime_beginning_ept':string_date_start},
                                        {'row_is_current':'TRUE'},
                                        {'pnode_id': tradeable_nodes}]
                            }
                    data =json.dumps(data)
                    response = requests.post(url, data=data, headers=headers)
                    json_return = json_normalize(response.json())['items']
                    new_file_df = pd.DataFrame(json_return[0])
                except:
                    for node_type in ['GEN','ZONE;AGGREGATE;HUB;INTERFACE;RESIDUAL_METERED_EDC']:
                        data = {'rowCount':50000,
                                'startRow':1,
                                'filters': [{'datetime_beginning_ept':string_date_start},
                                            {'row_is_current':'TRUE'},
                                            {'type':node_type}]
                                }
                        data =json.dumps(data)
                        response = requests.post(url, data=data, headers=headers)
                        json_return = json_normalize(response.json())['items']
                        if new_file_df.empty:
                            new_file_df = pd.DataFrame(json_return[0])
                        else:
                            new_file_df = pd.concat([new_file_df,pd.DataFrame(json_return[0])],axis=0)
                            new_file_df.reset_index(inplace=True, drop=True)

                new_file_df['datetime_beginning_ept'] = new_file_df['datetime_beginning_ept'].astype('datetime64[ns]')
                new_file_df['datetime_ending_ept'] = new_file_df['datetime_beginning_ept'] + datetime.timedelta(hours=1)
                new_file_df.rename(columns={'total_lmp_da':'DALMP','total_lmp_rt':'RTLMP'},inplace=True)
                new_file_df['Date'] = new_file_df['datetime_ending_ept'].apply(lambda datestamp: datetime.datetime(datestamp.year, datestamp.month, datestamp.day))
                new_file_df['HourEnding'] = new_file_df['datetime_ending_ept'].dt.hour
                new_file_df.loc[new_file_df['HourEnding']==0, 'HourEnding'] = 24
                new_file_df.loc[new_file_df['HourEnding']==24, 'Date'] = new_file_df['Date'] - datetime.timedelta(days=1)
                new_file_df.drop(columns=['datetime_beginning_ept','datetime_ending_ept'], inplace=True)
                new_file_df.set_index(['Date','HourEnding'], drop=True, inplace=True)

                new_file_df = new_file_df.pivot_table(index=['Date','HourEnding'], columns='pnode_id', values=lmp_type+'LMP')

                del new_file_df.columns.name
                new_file_df.columns = ['PJM_'+str(col)+'_'+lmp_type+'LMP' for col in new_file_df.columns]

                for string in ['/', ' ', '(', ')']:
                    new_file_df.columns = [col.replace(string, '') for col in new_file_df.columns]

                if temp_df.empty:
                    temp_df = new_file_df
                    raw_dict_LMP_dataframes[lmp_type] = temp_df
                else:
                    temp_df = pd.concat([temp_df, new_file_df], axis=0)
                    raw_dict_LMP_dataframes[lmp_type] = temp_df

                file_date = file_date + datetime.timedelta(days=1)

            except:
                file_date = file_date + datetime.timedelta(days=1)
                print('No PJM LMP Data for Date: '+string_date_start)
    try:
        rt_lmp_df = raw_dict_LMP_dataframes['RT'].copy()
        da_lmp_df = raw_dict_LMP_dataframes['DA'].copy()
        rt_lmp_df.columns = [col.replace('RTLMP', '') for col in rt_lmp_df.columns]
        da_lmp_df.columns = [col.replace('DALMP', '') for col in da_lmp_df.columns]
        dart_df = da_lmp_df - rt_lmp_df
        dart_df.columns = [col + 'DART' for col in dart_df.columns]
        raw_dict_LMP_dataframes['DART'] = dart_df

        for key in raw_dict_LMP_dataframes.keys():
            for output_timezone in ['EST', 'EPT', 'CPT']:
                timezone_input_df = raw_dict_LMP_dataframes[key].copy()
                timezone_input_df.reset_index(inplace=True)
                timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                                   date_col_name='Date',
                                                   input_tz='EPT',
                                                   output_tz=output_timezone)

                timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)

                if output_timezone not in output_dict_DART_dataframes.keys():
                    if key == 'DA':
                        output_dict_DALMP_dataframes[output_timezone] = timezone_input_df
                    elif key == 'RT':
                        output_dict_RTLMP_dataframes[output_timezone] = timezone_input_df
                    elif key == 'DART':
                        output_dict_DART_dataframes[output_timezone] = timezone_input_df

                else:
                    if key == 'DA':
                        output_dict_DALMP_dataframes[output_timezone] = pd.concat(
                            [output_dict_DALMP_dataframes[output_timezone], timezone_input_df], axis=0)
                    elif key == 'RT':
                        output_dict_RTLMP_dataframes[output_timezone] = pd.concat(
                            [output_dict_RTLMP_dataframes[output_timezone], timezone_input_df], axis=0)
                    elif key == 'DART':
                        output_dict_DART_dataframes[output_timezone] = pd.concat(
                            [output_dict_DART_dataframes[output_timezone], timezone_input_df], axis=0)
    except:
        print('No PJM LMP Data for Timeframe Selected.')

    return output_dict_DALMP_dataframes, output_dict_RTLMP_dataframes, output_dict_DART_dataframes


def get_ISONE_outage(start_date, end_date):
    output_dict_dataframes = {'EST':None,'EPT':None,'CPT':None}
    start_date = parse(start_date)
    end_date = parse(end_date)
    root_url = 'https://www.iso-ne.com/transform/csv/'
    feed_type = 'sdf'
    file_date = start_date
    output_df = pd.DataFrame()
    while file_date <= end_date:
        if file_date >= datetime.datetime(2016,4,21): string_date = file_date.strftime('%Y%m%d') ##ISONE changed their file naming methodoloy on this date
        else:  string_date = (file_date-datetime.timedelta(days=1)).strftime('%Y%m%d')

        print('ISONE Outage/Gen Data Processing for Date: ' + string_date)
        try:
            url = root_url + feed_type + '?start=' + string_date
            request = requests.get(url).text
            new_file_df = pd.read_csv(io.StringIO(request), sep=',', skiprows=6)

            new_file_df = new_file_df.iloc[:,1:4]
            new_file_df.dropna(axis=0, inplace=True)
            new_file_df.set_index(['Date'], inplace=True, drop=True)
            new_file_df = new_file_df.T
            descript_ref = {' ': "",
                            '/': "",
                            '(': "",
                            ')': "",
                            'OtherGenerationOutages': 'ISONE_OUTAGE',
                            'AnticipatedColdWeatherOutages': 'ISONE_COLD_OUTAGES',
                            'ProjectedSurplusDeficiency': 'ISONE_SURPLUS_CAPACITY',
                            }
            for orig, replace in descript_ref.items():
                new_file_df.columns = new_file_df.columns.str.replace(orig, replace)

            new_file_df.reset_index(inplace=True)
            new_file_df.rename(columns = {'index':'Date'},inplace=True)
            new_file_df['Date'] = new_file_df['Date'].astype('datetime64[ns]')
            new_file_df.set_index(['Date'], drop=True, inplace=True)

            new_file_df = new_file_df.append([new_file_df] * 23)
            new_file_df.sort_values('Date', inplace=True)
            new_file_df['HourEnding'] = list(range(1, 25)) + list(range(1, 25))
            new_file_df.reset_index(inplace=True)
            new_file_df.set_index(['Date', 'HourEnding'], drop=True, inplace=True)
            new_file_df = new_file_df[[col for col in new_file_df.columns if 'ISONE' in col]]


            if output_df.empty:
                output_df = new_file_df
            else:
                output_df = pd.concat([output_df, new_file_df], axis=0)
                output_df = output_df.loc[~output_df.index.duplicated(keep='last')]
        except:
            print('ISONE Outage File for date ' + string_date + ' not available on website')

        file_date = file_date + datetime.timedelta(days=1)

        for output_timezone in ['EST', 'EPT', 'CPT']:
            output_dict_dataframes[
                output_timezone] = output_df  ### values for all timezones are the same since reporting is on a daily level
    return output_dict_dataframes


def get_ISONE_load_new(start_date, end_date):
    output_dict_dataframes = {'EST':None,'EPT':None,'CPT':None}
    root_url = 'https://www.iso-ne.com/transform/csv/'
    feed_type = 'reliabilityregionloadforecast'
    file_date = start_date
    while file_date <= end_date:
        string_date = file_date.strftime('%Y%m%d')
        print('ISONE New Load Processing for Date: ' + string_date)
        two_day_output_df = pd.DataFrame()

        try:
            for pull_day in [file_date, file_date + datetime.timedelta(days=1)]:  ##Need to pull two files for next two days in order to get timezone outputs for CPT and EST
                string_date = pull_day.strftime('%Y%m%d')
                url = root_url + feed_type + '?start=' + string_date
                request = requests.get(url).text
                new_file_df = pd.read_csv(io.StringIO(request), sep=',', skiprows=5)
                new_file_df.rename(columns={'Date.1':'ModelRunDate', 'Number':'FLOAD', 'String':'Region', 'HE':'HourEnding'},inplace=True)
                new_file_df.dropna(axis=0, inplace=True)
                new_file_df['Date'] = new_file_df['Date'].astype('datetime64[ns]')
                new_file_df['ModelRunDate'] = new_file_df['ModelRunDate'].astype('datetime64[ns]')
                new_file_df = new_file_df[new_file_df['ModelRunDate']<(pull_day+datetime.timedelta(hours=7)-datetime.timedelta(days=1))]  #no forecasts before 7am of day before predict date
                new_file_df = new_file_df[new_file_df['ModelRunDate'] == new_file_df['ModelRunDate'].max()]
                new_file_df.drop(columns=['H', 'Number.1','ModelRunDate'], inplace=True)

                new_file_df = new_file_df[new_file_df['HourEnding'].astype('str') != '02X']  ## remove extra hour for daylight savings
                new_file_df.set_index(['Date', 'HourEnding'], drop=True, inplace=True)
                unstacked_file_df = pd.DataFrame()



                for region in new_file_df['Region'].unique():
                    regional_df = new_file_df[new_file_df['Region'] == region].copy()
                    regional_df.drop(columns='Region', inplace=True)
                    regional_df.columns = ['ISONE_' + region + '_' + str(col) for col in regional_df.columns]
                    if unstacked_file_df.empty:
                        unstacked_file_df = regional_df
                    else:
                        unstacked_file_df = unstacked_file_df.merge(regional_df, on=['Date', 'HourEnding'])

                if two_day_output_df.empty:
                    two_day_output_df = unstacked_file_df
                else:
                    two_day_output_df = pd.concat([two_day_output_df, unstacked_file_df], axis=0)

            two_day_output_df['ISONE_TOTAL_FLOAD'] = two_day_output_df.sum(axis=1)

            for output_timezone in ['EST', 'EPT', 'CPT']:
                timezone_input_df = two_day_output_df.copy()
                timezone_input_df.reset_index(inplace=True)
                timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                                   date_col_name='Date',
                                                   input_tz='EPT',
                                                   output_tz=output_timezone)
                timezone_input_df = timezone_input_df[timezone_input_df['Date'] == file_date]  ## Only return the next days' forecast
                timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)


                if output_timezone not in output_dict_dataframes.keys():
                    output_dict_dataframes[output_timezone] = timezone_input_df
                else:
                    output_dict_dataframes[output_timezone] = pd.concat([output_dict_dataframes[output_timezone], timezone_input_df], axis=0)
        except:
            print('ISONE New Load File for date ' + string_date + ' not available on website')

        file_date = file_date + datetime.timedelta(days=1)

    return output_dict_dataframes


def get_ISONE_load_archive(start_date, end_date):
    output_dict_dataframes = {'EST':None,'EPT':None,'CPT':None}
    root_url = 'https://www.iso-ne.com/static-assets/documents/'
    file_date = start_date
    while file_date <= end_date:
        string_date = file_date.strftime('%Y_%m_%d')
        string_year = file_date.strftime('%Y')
        string_month = file_date.strftime('%m')
        string_day = file_date.strftime('%d')
        print('ISONE Archive Load Processing for Date: ' + string_date)

        try:
            url = root_url + string_year + '/' + string_month + '/rr_lf_' + string_date + '.csv'
            new_file_df = pd.read_csv(url)

            new_file_df.rename(columns={'CT':'ISONE_.Z.CONNECTICUT_FLOAD',
                                        'ME':'ISONE_.Z.MAINE_FLOAD',
                                        'NEMABOS':'ISONE_.Z.NEMASSBOST_FLOAD',
                                        'NH': 'ISONE_.Z.NEWHAMPSHIRE_FLOAD',
                                        'RI': 'ISONE_.Z.RHODEISLAND_FLOAD',
                                        'SEMA': 'ISONE_.Z.SEMASS_FLOAD',
                                        'VT': 'ISONE_.Z.VERMONT_FLOAD',
                                        'WCMA': 'ISONE_.Z.WCMASS_FLOAD',
                                        'Zonal_Total': 'ISONE_TOTAL_FLOAD',
                                        'Hour':'HourEnding'},inplace=True)
            new_file_df.dropna(axis=0, inplace=True)
            new_file_df['Date'] = pd.to_datetime(new_file_df[['Year','Month','Day']])
            new_file_df.drop(columns=['Month', 'Day','Year'], inplace=True)
            new_file_df = new_file_df[[col for col in new_file_df.columns if '%' not in col]]
            new_file_df = new_file_df[new_file_df['HourEnding'] != '02X']  ## remove extra hour for daylight savings
            new_file_df.set_index(['Date', 'HourEnding'], drop=True, inplace=True)

            for output_timezone in ['EST', 'EPT', 'CPT']:
                timezone_input_df = new_file_df.copy()
                timezone_input_df.reset_index(inplace=True)
                timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                                   date_col_name='Date',
                                                   input_tz='EPT',
                                                   output_tz=output_timezone)

                timezone_input_df = timezone_input_df[timezone_input_df['Date'] == file_date]  ## Only return the next days' forecast
                timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)

                if output_timezone not in output_dict_dataframes.keys():
                    output_dict_dataframes[output_timezone] = timezone_input_df
                else:
                    output_dict_dataframes[output_timezone] = pd.concat(
                        [output_dict_dataframes[output_timezone], timezone_input_df], axis=0)

        except:
            print('ISONE Archive Load File for date ' + string_date + ' not available on website')

        file_date = file_date + datetime.timedelta(days=1)

    return output_dict_dataframes


def get_ISONE_load(start_date, end_date):
    start_date = parse(start_date)
    end_date = parse(end_date)
    output_dict_dataframes = {'EST':None,'EPT':None,'CPT':None}
    archive_dict = dict()
    new_dict = dict()

    if start_date <= datetime.datetime(2017,5,24):
        archive_dict = get_ISONE_load_archive(start_date=start_date,
                                              end_date=min(end_date, datetime.datetime(2017,5,24)))
    if end_date >= datetime.datetime(2017,5,25):
        new_dict = get_ISONE_load_new(start_date=max(start_date,datetime.datetime(2017,5,25)),
                                      end_date=end_date)

    if len(archive_dict)==0: output_dict_dataframes = new_dict
    elif len(new_dict)==0: output_dict_dataframes = archive_dict
    else:
        for timezone in archive_dict.keys():
            output_dict_dataframes[timezone] = pd.concat([archive_dict[timezone], new_dict[timezone]], axis=0)

    return output_dict_dataframes


def get_ISONE_LMPs(start_date, end_date):
    raw_dict_LMP_dataframes = dict()
    output_dict_DALMP_dataframes = {'EST':None,'EPT':None,'CPT':None}
    output_dict_RTLMP_dataframes = {'EST':None,'EPT':None,'CPT':None}
    output_dict_DART_dataframes = {'EST':None,'EPT':None,'CPT':None}
    start_date = parse(start_date)
    end_date = parse(end_date)
    api_dict = {'RT': 'rt-lmp/lmp_rt_final_',
                'DA': 'da-lmp/WW_DALMP_ISO_'}
    common_url = 'https://www.iso-ne.com/static-transform/csv/histRpts/'

    for lmp_type, api_name in api_dict.items():
        file_date = start_date
        temp_df = pd.DataFrame()

        while file_date <= end_date:
            string_date = file_date.strftime('%Y%m%d')
            print('ISONE ' + lmp_type + ' Data Processing for Date: ' + string_date)
            try:
                time.sleep(1)   ## ISO NE needs time cause they bitches
                new_file_df = pd.read_csv(common_url + api_name +  string_date+'.csv', skiprows=4)
                new_file_df = new_file_df.drop([new_file_df.index[0]])
                new_file_df.drop(columns=['Location Type', 'Energy Component','Congestion Component','Marginal Loss Component','H','Location Name'], inplace=True)
                new_file_df['Locational Marginal Price'] = pd.to_numeric(new_file_df['Locational Marginal Price'],errors='coerce')
                new_file_df.dropna(inplace=True)
                new_file_df['Date'] = new_file_df['Date'].astype('datetime64[ns]')
                new_file_df.rename(columns={'Hour Ending':'HourEnding', 'Locational Marginal Price':lmp_type+'LMP'},inplace=True)
                new_file_df = new_file_df[new_file_df['HourEnding'] != '02X']  ## remove extra hour for daylight savings
                new_file_df = new_file_df.pivot_table(index=['Date','HourEnding'], columns='Location ID', values=lmp_type+'LMP')
                del new_file_df.columns.name
                new_file_df.columns = ['ISONE_'+str(col)+'_'+lmp_type+'LMP' for col in new_file_df.columns]


                for string in ['/', ' ', '(', ')']:
                    new_file_df.columns = [col.replace(string, '') for col in new_file_df.columns]

                new_file_df = new_file_df.loc[:, ~new_file_df.columns.duplicated()]

                if temp_df.empty:
                    temp_df = new_file_df
                    raw_dict_LMP_dataframes[lmp_type] = temp_df
                else:
                    temp_df = pd.concat([temp_df, new_file_df], axis=0)
                    raw_dict_LMP_dataframes[lmp_type] = temp_df

            except:
                print('ISONE LMP File for date ' + string_date + ' not available on website')

            file_date = file_date + datetime.timedelta(days=1)
    try:
        rt_lmp_df = raw_dict_LMP_dataframes['RT'].copy()
        da_lmp_df = raw_dict_LMP_dataframes['DA'].copy()
        rt_lmp_df.columns = [col.replace('RTLMP', '') for col in rt_lmp_df.columns]
        da_lmp_df.columns = [col.replace('DALMP', '') for col in da_lmp_df.columns]
        dart_df = da_lmp_df - rt_lmp_df
        dart_df.columns = [col + 'DART' for col in dart_df.columns]
        raw_dict_LMP_dataframes['DART'] = dart_df

        for key in raw_dict_LMP_dataframes.keys():
            for output_timezone in ['EST', 'EPT', 'CPT']:
                timezone_input_df = raw_dict_LMP_dataframes[key].copy()
                timezone_input_df.reset_index(inplace=True)
                timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                                   date_col_name='Date',
                                                   input_tz='EPT',
                                                   output_tz=output_timezone)

                timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)

                if output_timezone not in output_dict_DART_dataframes.keys():
                    if key == 'DA':
                        output_dict_DALMP_dataframes[output_timezone] = timezone_input_df
                    elif key == 'RT':
                        output_dict_RTLMP_dataframes[output_timezone] = timezone_input_df
                    elif key == 'DART':
                        output_dict_DART_dataframes[output_timezone] = timezone_input_df

                else:
                    if key == 'DA':
                        output_dict_DALMP_dataframes[output_timezone] = pd.concat(
                            [output_dict_DALMP_dataframes[output_timezone], timezone_input_df], axis=0)
                    elif key == 'RT':
                        output_dict_RTLMP_dataframes[output_timezone] = pd.concat(
                            [output_dict_RTLMP_dataframes[output_timezone], timezone_input_df], axis=0)
                    elif key == 'DART':
                        output_dict_DART_dataframes[output_timezone] = pd.concat(
                            [output_dict_DART_dataframes[output_timezone], timezone_input_df], axis=0)
    except:
        print('No ISONE LMP Data for Timeframe Selected.')
    return output_dict_DALMP_dataframes, output_dict_RTLMP_dataframes, output_dict_DART_dataframes


def get_NYISO_load(start_date, end_date, static_directory):
    save_directory = static_directory + '\ModelUpdateData\\NYISO_files\\'
    output_dict_dataframes = {'EST':None,'EPT':None,'CPT':None}
    start_date = parse(start_date) - datetime.timedelta(days=1)
    end_date = parse(end_date) - datetime.timedelta(days=1)
    root_url = 'http://mis.nyiso.com/public/csv/isolf/'
    download_date = datetime.datetime(year = start_date.year, month = start_date.month, day=1)

    while download_date <=end_date:
        string_date = download_date.strftime('%Y%m%d')
        file_name = string_date + 'isolf_csv.zip'
        url = root_url + file_name
        file_name_loc = save_directory + string_date+ 'zonalBidLoad_csv.zip'
        if os.path.exists(file_name_loc)==False:
            try:
                response = requests.get(url)
                z = zipfile.ZipFile(io.BytesIO(response.content))
                z.extractall(path=save_directory)
            except Exception as x:
                pass
        download_date = download_date + datetime.timedelta(days=1)

    file_date = start_date

    while file_date <= end_date:
        string_date = file_date.strftime('%Y%m%d')

        print('NYISO Load Processing for Date: ' + string_date)

        try:
            url = save_directory + string_date+ 'isolf.csv'
            new_file_df = pd.read_csv(url)
            new_file_df.rename(columns = {'Time Stamp':'Date'},inplace=True)
            new_file_df['Date'] = new_file_df['Date'].astype('datetime64[ns]')
            new_file_df['HourEnding'] = new_file_df['Date'].dt.hour+1
            new_file_df['Date'] = new_file_df['Date'].apply(lambda datestamp: datetime.datetime(year= datestamp.year, month=datestamp.month, day=datestamp.day))
            new_file_df.set_index(['Date', 'HourEnding'], drop=True, inplace=True)
            new_file_df.columns = [('NYISO'+col + '_FLOAD').replace(' ','') for col in new_file_df.columns]

            for output_timezone in ['EST', 'EPT', 'CPT']:
                timezone_input_df = new_file_df.copy()
                timezone_input_df.reset_index(inplace=True)
                timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                                   date_col_name='Date',
                                                   input_tz='EPT',
                                                   output_tz=output_timezone)

                timezone_input_df = timezone_input_df[(timezone_input_df['Date'] >= file_date + datetime.timedelta(days=1)) & (timezone_input_df['Date'] <= file_date + datetime.timedelta(days=2))]  ## Only return the next days' forecast
                timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)

                df = output_dict_dataframes[output_timezone]
                if output_timezone not in output_dict_dataframes.keys():
                    df = timezone_input_df
                else:
                    df = pd.concat([df, timezone_input_df], axis=0)
                    df = df.loc[~df.index.duplicated(keep='last')]
                output_dict_dataframes[output_timezone] = df

        except:
            print('NYISO Load File for date ' + string_date + ' not available on website')

        file_date = file_date + datetime.timedelta(days=1)

    return output_dict_dataframes


def get_NYISO_LMPs(start_date, end_date,static_directory):
    save_directory = static_directory + '\ModelUpdateData\\NYISO_files\\'
    raw_dict_LMP_dataframes = dict()
    output_dict_DALMP_dataframes = {'EST':None,'EPT':None,'CPT':None}
    output_dict_RTLMP_dataframes = {'EST':None,'EPT':None,'CPT':None}
    output_dict_DART_dataframes = {'EST':None,'EPT':None,'CPT':None}
    start_date = parse(start_date)
    end_date = parse(end_date)

    api_dict = {'RT': ['rtlbmp/','rtlbmp_zone'],
                'DA': ['damlbmp/','damlbmp_zone']}

    root_url = 'http://mis.nyiso.com/public/csv/'
    download_date = datetime.datetime(year = start_date.year, month = start_date.month, day=1)

    while download_date <=end_date:
        for lmp_type, api_name in api_dict.items():
            string_date = download_date.strftime('%Y%m%d')
            file_name = string_date + api_dict[lmp_type][1]+'_csv.zip'
            url = root_url +api_dict[lmp_type][0] +  file_name
            file_name_loc =save_directory+ string_date + api_dict[lmp_type][1] + '_csv.zip'
            if os.path.exists(file_name_loc)==False:
                try:
                    response = requests.get(url)
                    z = zipfile.ZipFile(io.BytesIO(response.content))
                    z.extractall(path=save_directory)
                except Exception as x:
                    pass
        download_date = download_date + datetime.timedelta(days=1)


    for lmp_type, api_name in api_dict.items():
        file_date = start_date
        temp_df = pd.DataFrame()

        while file_date <= end_date:
            string_date = file_date.strftime('%Y%m%d')
            print('NYISO ' + lmp_type + ' Data Processing for Date: ' + string_date)
            # try:
            url = save_directory+string_date+api_dict[lmp_type][1]+'.csv'
            new_file_df = pd.read_csv(url)

            new_file_df.drop(columns=[col for col in new_file_df.columns if 'Losses' in col], inplace=True)
            new_file_df.drop(columns=[col for col in new_file_df.columns if 'Congestion' in col], inplace=True)
            new_file_df.rename(columns={'Time Stamp': 'Date','LBMP ($/MWHr)':lmp_type+'LMP'}, inplace=True)

            new_file_df['Date'] = new_file_df['Date'].astype('datetime64[ns]')
            new_file_df['HourEnding'] = new_file_df['Date'].dt.hour + 1
            new_file_df['Date'] = new_file_df['Date'].apply(lambda datestamp: datetime.datetime(year=datestamp.year, month=datestamp.month, day=datestamp.day))


            new_file_df = new_file_df.pivot_table(index=['Date','HourEnding'], columns='PTID', values=lmp_type+'LMP')
            del new_file_df.columns.name
            new_file_df.columns = ['NYISO_'+str(col)+'_'+lmp_type+'LMP' for col in new_file_df.columns]

            for string in ['/', ' ', '(', ')']:
                new_file_df.columns = [col.replace(string, '') for col in new_file_df.columns]

            new_file_df = new_file_df.loc[:, ~new_file_df.columns.duplicated()]

            if temp_df.empty:
                temp_df = new_file_df
                raw_dict_LMP_dataframes[lmp_type] = temp_df
            else:
                temp_df = pd.concat([temp_df, new_file_df], axis=0)
                raw_dict_LMP_dataframes[lmp_type] = temp_df

            # except:
            #     print('NYISO LMP File for date ' + string_date + ' not available on website')
            file_date = file_date + datetime.timedelta(days=1)

    try:
        rt_lmp_df = raw_dict_LMP_dataframes['RT'].copy()
        da_lmp_df = raw_dict_LMP_dataframes['DA'].copy()
        rt_lmp_df.columns = [col.replace('RTLMP', '') for col in rt_lmp_df.columns]
        da_lmp_df.columns = [col.replace('DALMP', '') for col in da_lmp_df.columns]
        dart_df = da_lmp_df - rt_lmp_df
        dart_df.columns = [col + 'DART' for col in dart_df.columns]
        raw_dict_LMP_dataframes['DART'] = dart_df

        for key in raw_dict_LMP_dataframes.keys():
            for output_timezone in ['EST', 'EPT', 'CPT']:
                timezone_input_df = raw_dict_LMP_dataframes[key].copy()
                timezone_input_df.reset_index(inplace=True)
                timezone_input_df = timezone_shift(input_datetime_df=timezone_input_df,
                                                   date_col_name='Date',
                                                   input_tz='EPT',
                                                   output_tz=output_timezone)

                timezone_input_df.set_index(['Date', 'HourEnding'], inplace=True, drop=True)

                if output_timezone not in output_dict_DART_dataframes.keys():
                    if key == 'DA':
                        output_dict_DALMP_dataframes[output_timezone] = timezone_input_df
                    elif key == 'RT':
                        output_dict_RTLMP_dataframes[output_timezone] = timezone_input_df
                    elif key == 'DART':
                        output_dict_DART_dataframes[output_timezone] = timezone_input_df

                else:
                    if key == 'DA':
                        output_dict_DALMP_dataframes[output_timezone] = pd.concat(
                            [output_dict_DALMP_dataframes[output_timezone], timezone_input_df], axis=0)
                    elif key == 'RT':
                        output_dict_RTLMP_dataframes[output_timezone] = pd.concat(
                            [output_dict_RTLMP_dataframes[output_timezone], timezone_input_df], axis=0)
                    elif key == 'DART':
                        output_dict_DART_dataframes[output_timezone] = pd.concat(
                            [output_dict_DART_dataframes[output_timezone], timezone_input_df], axis=0)
    except:
        print('No NYISO LMP Data for Timeframe Selected.')
    return output_dict_DALMP_dataframes, output_dict_RTLMP_dataframes, output_dict_DART_dataframes


def get_ISO_api_data(start_date, end_date, previous_data_dict_name, concat_old_dict, working_directory, static_directory):

    if concat_old_dict ==True: previous_dict = load_obj(previous_data_dict_name)

    output_dict_dataframes = {'EST':None,'EPT':None,'CPT':None}
    input_files_directory = static_directory + '\ModelUpdateData\\'
    end_date_string = end_date
    end_date = datetime.datetime.strptime(end_date,'%Y_%m_%d')
    end_date = end_date.strftime('%m/%d/%Y')
    start_date = datetime.datetime.strptime(start_date, '%Y_%m_%d')
    start_date = start_date.strftime('%m/%d/%Y')


    #### Process and add YES timeseries files
    file_name = end_date_string + '_BACKTEST_INPUT_FILE_YES_TS(PJMloadERCOTdata)'
    time_zone = 'EST'
    print('Processing YES Timeseries File: ' + file_name)
    try:
        ts_input_df = pd.read_excel(input_files_directory + file_name + '.xls')
    except:
        print(
            input_files_directory + file_name + '.xls' + ' file is not found. Please make sure it is in the "InputFiles" directory and is named correctly')
        exit()

    yes_timeseries_dict = process_YES_timeseries(start_date=start_date,
                                                 end_date=end_date,
                                                 input_df=ts_input_df,
                                                 input_timezone=time_zone)


    #### Process and add YES data extract tall (temperatures only) files
    file_name = end_date_string + '_BACKTEST_INPUT_FILE_YES_DATA_EXTRACT_TALL(Temps)'
    time_zone = 'EST'
    print('Processing YES Data Extract Tall (Temperatures) File: ' + file_name)
    try:
        data_extract_input_df = pd.read_csv(input_files_directory + file_name + '.csv')
    except:
        print(input_files_directory + file_name + '.csv' + ' file is not found. Please make sure it is in the "InputFiles" directory and is named correctly')
        exit()

    yes_tall_temps_dict = process_YES_data_extract_temps_tall(start_date=start_date,
                                                              end_date=end_date,
                                                              input_df=data_extract_input_df,
                                                              input_timezone=time_zone)

    #### Process and add YES data extract wide files (ERCOT LMPs)
    file_name = end_date_string + '_BACKTEST_INPUT_FILE_YES_DATA_EXTRACT_WIDE(ERCOTLMPs)'
    time_zone = 'CPT'
    print('')
    print('Processing Data Extract Wide File: ' + file_name)
    try:
        data_extract_input_df = pd.read_csv(input_files_directory+file_name+'.csv')
    except:
        print(input_files_directory + file_name+'.csv' + ' file is not found. Please make sure it is in the "InputFiles" directory and is named correctly')
        exit()

    yes_wide_extract_dict = process_YES_data_extract_wide(start_date=start_date,
                                                          end_date=end_date,
                                                          input_df=data_extract_input_df,
                                                          input_timezone=time_zone)


    #Pull individual ISO API data
    MISO_load_outage_dict = get_MISO_load_outage(start_date=start_date,
                                                 end_date=end_date)

    MISO_DALMP_dict, MISO_RTLMP_dict, MISO_DART_dict = get_MISO_LMPs(start_date=start_date,
                                                                     end_date=end_date,
                                                                     static_directory=static_directory)

    SPP_wind_dict = get_SPP_wind(start_date=start_date,
                                 end_date=end_date,
                                 static_directory=static_directory)

    SPP_outage_dict = get_SPP_outage(start_date=start_date,
                                     end_date=end_date,
                                     static_directory=static_directory)

    SPP_load_dict = get_SPP_load(start_date=start_date,
                                 end_date=end_date,
                                 static_directory=static_directory)

    SPP_DALMP_dict, SPP_RTLMP_dict, SPP_DART_dict = get_SPP_LMPs(start_date=start_date,
                                                                 end_date=end_date,
                                                                 static_directory=static_directory)

    PJM_outage_dict = get_PJM_outage(start_date=start_date,
                                     end_date=end_date)

    PJM_load_dict = {'EST':None,'EPT':None,'CPT':None}
    if (start_date == end_date) and (parse(start_date) > datetime.datetime.today()-datetime.timedelta(days=3)):
        PJM_load_dict = get_PJM_load(current_date=start_date)

    PJM_DALMP_dict, PJM_RTLMP_dict, PJM_DART_dict = get_PJM_LMPs(start_date=start_date,
                                                                 end_date=end_date,
                                                                 static_directory=static_directory)

    ISONE_outage_dict = get_ISONE_outage(start_date=start_date,
                                         end_date=end_date)

    ISONE_load_dict = get_ISONE_load(start_date=start_date,
                                     end_date=end_date)

    ISONE_DALMP_dict, ISONE_RTLMP_dict, ISONE_DART_dict = get_ISONE_LMPs(start_date=start_date,
                                                                     end_date=end_date)


    NYISO_load_dict = get_NYISO_load(start_date=start_date,
                                     end_date=end_date,
                                     static_directory=static_directory)

    NYISO_DALMP_dict, NYISO_RTLMP_dict, NYISO_DART_dict = get_NYISO_LMPs(start_date=start_date,
                                                                     end_date=end_date,
                                                                         static_directory=static_directory)


    list_to_concat = [MISO_load_outage_dict,MISO_DART_dict,
                      SPP_wind_dict,SPP_outage_dict,SPP_load_dict,SPP_DART_dict,
                      PJM_outage_dict,PJM_DART_dict,PJM_load_dict,
                      ISONE_outage_dict,ISONE_load_dict,ISONE_DART_dict,
                      NYISO_load_dict,NYISO_DART_dict,
                      yes_wide_extract_dict,yes_tall_temps_dict,yes_timeseries_dict]


    for dict_to_concat in list_to_concat:
        for timezone in ['EST','EPT','CPT']:
            if dict_to_concat[timezone] is not None:
                try:
                    output_dict_dataframes[timezone] = output_dict_dataframes[timezone].join(dict_to_concat[timezone],how='outer',on=['Date','HourEnding']).sort_values(['Date','HourEnding'],ascending=True)
                except:
                    output_dict_dataframes[timezone] = dict_to_concat[timezone]

    if concat_old_dict == True:
        for timezone in ['EST','EPT','CPT']:
            old_df = previous_dict[timezone]
            new_df = output_dict_dataframes[timezone]
            concat_df = pd.concat([old_df, new_df],join='inner')
            concat_df = concat_df.loc[~concat_df.index.duplicated(keep='last')]
            concat_df = concat_df.sort_index()
            output_dict_dataframes[timezone] = concat_df

    dict_save_name = end_date_string + '_BACKTEST_DATA'
    save_obj(output_dict_dataframes, input_files_directory+dict_save_name+'_DICT_RAW')

    post_process_dict = post_process_backtest_data(input_dict=output_dict_dataframes,
                                                   static_directory=static_directory)


    # Save Final Dict
    save_obj(post_process_dict,input_files_directory + dict_save_name+'_DICT_MASTER')

    # Save CSV Files
    for key, value in post_process_dict.items():
        post_process_dict[key].to_csv(input_files_directory+ dict_save_name+'_MASTER'+'_'+key+'.csv')

    #Create Max Min Limits For Daily Trade File
    max_min_save_name = end_date_string + '_MAX_MIN_LIMITS'
    max_min_df = create_max_min_limits(input_dict=output_dict_dataframes)
    max_min_df.to_csv(input_files_directory+max_min_save_name+'.csv')

    return post_process_dict


def read_clean_data_daily_file(input_df, verbose=True):
    # Create a Multi-Index to Keep Things Orderly. Date and HE remain as multi-index throughout code.
    input_df.reset_index(inplace=True)
    input_df['HE'] = input_df['HourEnding']
    input_df['Date'] = input_df['Date'].astype('datetime64[ns]')
    input_df.set_index(['Date', 'HE'], inplace=True, drop=True)

    # Remove Rows With NA Values and Duplicated Rows
    initial_rows = len(input_df)
    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    master_df = input_df.loc[~input_df.index.duplicated(keep='first')]

    orig_cols = set(master_df.columns)
    inactive_nodes = ['MISO_AMIL.COFFEEN1_DART','MISO_AMIL.HAVANA86_DART','MISO_AMIL.HENNEPN82_DART','MISO_DMGEN3.AGG_DART','MISO_EES.CC.WPEC_DART','MISO_EES.WRPP1_DART',
                      'MISO_AMIL.COFFEEN1_DA_RT_LAG','MISO_AMIL.HAVANA86_DA_RT_LAG','MISO_AMIL.HENNEPN82_DA_RT_LAG','MISO_DMGEN3.AGG_DA_RT_LAG','MISO_EES.CC.WPEC_DA_RT_LAG','MISO_EES.WRPP1_DA_RT_LAG']
    master_df = master_df[[col for col in master_df if col not in inactive_nodes]]
    new_cols = set(master_df.columns)
    if verbose:
        print("Duplicated rows dropped = " + str(initial_rows - len(master_df)))
        print('While reading and cleaning data dropped inactive nodes:')
        print(orig_cols-new_cols)

    return master_df


def post_process_backtest_data(static_directory,dict_filename=None, input_dict=None):
    input_files_directory = static_directory + '\ModelUpdateData\\'
    # # Load Dict if needed
    if input_dict==None:
        input_dict = load_obj(input_files_directory+dict_filename)
    else:
        input_dict = input_dict

    # Preprocess Data (Remove Bad and Missing Data
    input_dict = preprocess_data(input_dict=input_dict,
                                 static_directory=static_directory)
    input_dict = drop_correlated_data(input_dict=input_dict)
    input_dict = lag_data16_40(input_dict=input_dict, col_type='DART')

    return input_dict


def get_daily_input_data(predict_date_str_mm_dd_yyyy, working_directory, static_directory):
    print('')
    print('**********************************************************************************')
    print('')
    print('Fetching Input Data for Prediction Date: ' + predict_date_str_mm_dd_yyyy + '...')
    print('')
    print('**********************************************************************************')


    input_files_directory = working_directory + '\InputFiles\\'
    output_dict_dataframes = {'EST':None,'EPT':None,'CPT':None}

    predict_date = datetime.datetime.strptime(predict_date_str_mm_dd_yyyy, '%m_%d_%Y')
    current_date = predict_date - datetime.timedelta(days=1)


    start_date = predict_date - datetime.timedelta(days=1)
    end_date = predict_date
    end_date_plus_1 = predict_date + datetime.timedelta(days=1)
    start_date = start_date.strftime('%m/%d/%Y')
    end_date = end_date.strftime('%m/%d/%Y')
    end_date_plus_1 = end_date_plus_1.strftime('%m/%d/%Y')


    ## GET API DATA (NON DART)

    MISO_load_outage_dict = get_MISO_load_outage(start_date=start_date,
                                                 end_date=end_date)
    SPP_wind_dict = get_SPP_wind(start_date=start_date,
                                 end_date=end_date,
                                 static_directory=static_directory)
    SPP_outage_dict = get_SPP_outage(start_date=start_date,
                                     end_date=end_date,
                                     static_directory=static_directory)
    SPP_load_dict = get_SPP_load(start_date=start_date,
                                 end_date=end_date,
                                 static_directory=static_directory)
    PJM_outage_dict = get_PJM_outage(start_date=start_date,
                                     end_date=end_date)

    PJM_load_dict = get_PJM_load(start_date=start_date,
                                     end_date=end_date_plus_1)

    ISONE_outage_dict = get_ISONE_outage(start_date=start_date,
                                         end_date=end_date)
    ISONE_load_dict = get_ISONE_load(start_date=start_date,
                                     end_date=end_date_plus_1)
    NYISO_load_dict = get_NYISO_load(start_date=start_date,
                                     end_date=end_date_plus_1,
                                     static_directory=static_directory)

    #### Process and add YES data extract tall (temperatures only) files
    file_name = predict_date_str_mm_dd_yyyy+ '_DAILY_INPUT_YES_EXTRACT_TEMPS_TALL'
    time_zone = 'EST'

    # print('Processing YES Data Extract Tall (Temperatures) File: ' + file_name)
    # try:
    #     data_extract_input_df = pd.read_csv(input_files_directory+file_name+'.csv')
    # except:
    #     print('ERROR: ' + input_files_directory + file_name+'.csv' + ' file is not found. Please make sure it is in the "InputFiles" directory and is named correctly')
    #     exit()
    #
    # try:
    #     yes_tall_temps_dict = process_YES_data_extract_temps_tall(start_date=start_date,
    #                                                               end_date=end_date,
    #                                                               input_df=data_extract_input_df,
    #                                                               input_timezone=time_zone)
    #
    #     data_extract_start_date = yes_tall_temps_dict['EST'].index.get_level_values('Date')[0]
    #     data_extract_end_date =yes_tall_temps_dict['EST'].index.get_level_values('Date')[-1]
    #
    #     if (current_date!=data_extract_start_date) or (predict_date!=data_extract_end_date):
    #         print('ERROR: YES Data Extract (Temperature) File Error. Check Start and End Dates In YES File Download')
    #         exit()
    # except:
    #     print('ERROR: YES Data Extract (Temperature) File Error. Check Start and End Dates In YES File Download')
    #     exit()

    #### Process and add YES timeseries files
    file_name = predict_date_str_mm_dd_yyyy+ '_DAILY_INPUT_YES_TS'
    time_zone = 'EST'
    print('Processing YES Timeseries File: ' + file_name)
    try:
        ts_input_df = pd.read_excel(input_files_directory+file_name+'.xls')
    except:
        print('ERROR: ' +input_files_directory + file_name + '.xls' + ' file is not found. Please make sure it is in the "InputFiles" directory and is named correctly')
        exit()

    try:
        yes_timeseries_dict = process_YES_timeseries(start_date=start_date,
                                                     end_date=end_date,
                                                     input_df=ts_input_df,
                                                     input_timezone=time_zone)

        timeseries_start_date = yes_timeseries_dict['EST'].index.get_level_values('Date')[0]
        timeseries_end_date =yes_timeseries_dict['EST'].index.get_level_values('Date')[-1]


        if (current_date!=timeseries_start_date) or (predict_date!=timeseries_end_date):
            print('ERROR: YES Timeseries File Error. Check Start and End Dates In YES File Download')
            exit()

    except:
        print('ERROR: YES Timeseries File Error. Check Start and End Dates In YES File Download')
        exit()


    ### Get YES Pricetable DART data (All ISOs)
    file_name = predict_date_str_mm_dd_yyyy + '_DAILY_INPUT_YES_PRICE_TABLE_T-XDAY'
    print('Processing YES PriceTable File: ' + file_name)

    try:
        yes_pricetable_dict = process_YES_daily_price_tables(predict_date=predict_date,
                                                             input_timezone = 'CPT',
                                                             working_directory=working_directory,
                                                             dart_only=True)
    except:
        print('ERROR: YES PriceTable File Error. Check Start and End Dates In YES File Download')
        exit()


    ### LAG DART DATA
    yes_pricetable_dict = lag_data16_40(input_dict=yes_pricetable_dict,
                                        return_only_lagged=True)

    ### CONCAT ALL DATA INTO ONE DICTIONARY

    list_to_concat = [MISO_load_outage_dict,
                      SPP_wind_dict,
                      SPP_outage_dict,
                      SPP_load_dict,
                      PJM_outage_dict,
                      PJM_load_dict,
                      ISONE_outage_dict,
                      ISONE_load_dict,
                      NYISO_load_dict,
                      #yes_tall_temps_dict,
                      yes_timeseries_dict,
                      yes_pricetable_dict]

    for dict_to_concat in list_to_concat:
        for timezone in ['EST','EPT','CPT']:
            if dict_to_concat[timezone] is not None:
                df = output_dict_dataframes[timezone]
                try:
                    df = df.join(dict_to_concat[timezone],how='outer',on=['Date','HourEnding']).sort_values(['Date','HourEnding'],ascending=True)
                except:
                    df = dict_to_concat[timezone]
                output_dict_dataframes[timezone]= df


    # Save Final Dict
    save_obj(output_dict_dataframes, input_files_directory+predict_date_str_mm_dd_yyyy + '_RAW_DAILY_INPUT_DATA_DICT')

    return output_dict_dataframes


def get_lmps(start_date, end_date, previous_data_dict_name, concat_old_dict, working_directory, static_directory):

    if concat_old_dict ==True: previous_dict = load_obj(previous_data_dict_name)

    output_dict_dataframes = {'EST':None,'EPT':None,'CPT':None}
    input_files_directory = static_directory + '\ModelUpdateData\\'
    end_date_string = end_date
    end_date = datetime.datetime.strptime(end_date,'%Y_%m_%d')
    end_date = end_date.strftime('%m/%d/%Y')
    start_date = datetime.datetime.strptime(start_date, '%Y_%m_%d')
    start_date = start_date.strftime('%m/%d/%Y')

    #
    # #### Process and add YES data extract wide files (ERCOT LMPs)
    file_name = end_date_string + '_BACKTEST_INPUT_FILE_YES_DATA_EXTRACT_WIDE(ERCOT_DA+RT_LMPs)'
    time_zone = 'CPT'
    print('')
    print('Processing Data Extract Wide File: ' + file_name)
    try:
        data_extract_input_df = pd.read_csv(input_files_directory+file_name+'.csv')
    except:
        print(input_files_directory + file_name+'.csv' + ' file is not found. Please make sure it is in the "InputFiles" directory and is named correctly')
        exit()

    ERCOT_LMP_dict = process_YES_data_extract_wide(start_date=start_date,
                                                          end_date=end_date,
                                                          input_df=data_extract_input_df,
                                                          input_timezone=time_zone)

    #Pull individual ISO API data


    MISO_DALMP_dict, MISO_RTLMP_dict, MISO_DART_dict = get_MISO_LMPs(start_date=start_date,
                                                                     end_date=end_date,
                                                                     static_directory=static_directory)

    SPP_DALMP_dict, SPP_RTLMP_dict, SPP_DART_dict = get_SPP_LMPs(start_date=start_date,
                                                                 end_date=end_date,
                                                                 static_directory=static_directory)


    PJM_DALMP_dict, PJM_RTLMP_dict, PJM_DART_dict = get_PJM_LMPs(start_date=start_date,
                                                                 end_date=end_date,
                                                                 static_directory=static_directory)

    ISONE_DALMP_dict, ISONE_RTLMP_dict, ISONE_DART_dict = get_ISONE_LMPs(start_date=start_date,
                                                                     end_date=end_date)


    NYISO_DALMP_dict, NYISO_RTLMP_dict, NYISO_DART_dict = get_NYISO_LMPs(start_date=start_date,
                                                                     end_date=end_date,
                                                                         static_directory=static_directory)


    list_to_concat = [ERCOT_LMP_dict,MISO_DALMP_dict,SPP_DALMP_dict,PJM_DALMP_dict,ISONE_DALMP_dict,NYISO_DALMP_dict,
                      MISO_RTLMP_dict,SPP_RTLMP_dict,PJM_RTLMP_dict,ISONE_RTLMP_dict,NYISO_RTLMP_dict]


    for dict_to_concat in list_to_concat:
        for timezone in ['EST','EPT','CPT']:
            if dict_to_concat[timezone] is not None:
                try:
                    output_dict_dataframes[timezone] = output_dict_dataframes[timezone].join(dict_to_concat[timezone],how='outer',on=['Date','HourEnding']).sort_values(['Date','HourEnding'],ascending=True)
                except:
                    output_dict_dataframes[timezone] = dict_to_concat[timezone]

    if concat_old_dict == True:
        for timezone in ['EST','EPT','CPT']:
            old_df = previous_dict[timezone]
            new_df = output_dict_dataframes[timezone]
            concat_df = pd.concat([old_df, new_df],join='inner')
            concat_df = concat_df.loc[~concat_df.index.duplicated(keep='last')]
            concat_df = concat_df.sort_index()
            output_dict_dataframes[timezone] = concat_df

    dict_save_name = end_date_string + '_DALMP_DATA'
    save_obj(output_dict_dataframes, input_files_directory+dict_save_name+'_DICT_MASTER')

    # Save CSV Files
    for key, value in output_dict_dataframes.items():
        output_dict_dataframes[key].to_csv(input_files_directory+ dict_save_name+'_MASTER'+'_'+key+'.csv')

    return output_dict_dataframes


def create_max_min_limits(input_dict):

    input_df = input_dict['EST']
    input_df = input_df.astype('float')


    max_min_df = pd.DataFrame()
    max_min_df['MaxLimit'] = input_df.max()
    max_min_df['MinLimit'] = input_df.min()
    max_min_df['MaxLimit'] = max_min_df['MaxLimit']*1.1
    max_min_df['MinLimit'] = max_min_df['MinLimit']  - max_min_df['MinLimit']*0.1
    max_min_df.reset_index(inplace=True)
    max_min_df.columns = ['FeatureName','MaxLimit','MinLimit']
    max_min_df['Action'] = 'Delete'
    max_min_df['ISO'] = 'None'
    max_min_df['DART?'] = 'NO'
    max_min_df.loc[(max_min_df['FeatureName'].str.contains('DA_RT')),'Action']='Ignore'
    max_min_df.loc[(max_min_df['FeatureName'].str.contains('DART')), 'Action'] = 'Ignore'
    max_min_df.loc[(max_min_df['FeatureName'].str.contains('MISO_')), 'ISO'] = 'MISO'
    max_min_df.loc[(max_min_df['FeatureName'].str.contains('PJM_')), 'ISO'] = 'PJM'
    max_min_df.loc[(max_min_df['FeatureName'].str.contains('SPP_')), 'ISO'] = 'SPP'
    max_min_df.loc[(max_min_df['FeatureName'].str.contains('ERCOT_')), 'ISO'] = 'ERCOT'
    max_min_df.loc[(max_min_df['FeatureName'].str.contains('ISONE_')), 'ISO'] = 'ISONE'
    max_min_df.loc[(max_min_df['FeatureName'].str.contains('NYISO_')), 'ISO'] = 'NYISO'
    max_min_df.loc[(max_min_df['FeatureName'].str.contains('DART')), 'DART?'] = 'YES'
    max_min_df['DARTNameForYESCollection'] = max_min_df['FeatureName'].str.replace('_DART','').str.replace('MISO_','').str.replace('PJM_','').str.replace('SPP_','').str.replace('ERCOT_','').str.replace('ISONE_','').str.replace('NYISO_','')

    return max_min_df
