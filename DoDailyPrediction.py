import os
import pandas as pd
from API_Lib import get_daily_input_data
from XGBLib import create_VAR
from XGBLib import do_xgb_prediction
from XGBLib import post_process_trades
from XGBLib import create_trade_summary
from XGBLib import daily_PnL

from dateutil.parser import parse

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 5000)
pd.set_option('max_row', 100)


root_directory = 'X:\\Production\\'
# root_directory = 'C:\\XGBoostProduction\\'


trade_handler_directory = root_directory + '\\DailyTradeFiles\\'
upload_directory = root_directory + '\\UploadFiles\\'


trade_handler_df = pd.read_excel(trade_handler_directory+'$$$TradeHandler$$$.xls', skiprows=9).dropna(thresh=6)
trade_handler_df = trade_handler_df.fillna('')

for row in trade_handler_df.index:
    print(trade_handler_df.iloc[[row]])

    current_isos = trade_handler_df['ISOs'][row]
    current_isos = current_isos.replace(' ','').split(',')
    date = trade_handler_df['Date'][row]
    predict_date_str_mm_dd_yyyy = date.strftime('%m_%d_%Y')
    model_type = trade_handler_df['DARTorSPREAD'][row]
    do_data_pull = trade_handler_df['DataPull'][row]
    do_prediction = trade_handler_df['Prediction'][row]
    do_postprocessing = trade_handler_df['PostProcessing_YESFiles'][row]
    do_tradesummary = trade_handler_df['TradeSummary'][row]
    do_printcharts = trade_handler_df['PrintCharts'][row]
    do_daily_PnL = trade_handler_df['DailyPnL'][row]
    do_settlement_PnL = trade_handler_df['SettlementPnL'][row]
    do_VAR = trade_handler_df['VAR'][row]
    VAR_isos = trade_handler_df['VAR_ISOs'][row]
    VAR_isos = VAR_isos.replace(' ','').split(',')
    working_directory = trade_handler_df['WorkingDirectory'][row]
    static_directory = trade_handler_df['StaticDirectory'][row]
    daily_trade_file_name = trade_handler_df['TradeVariablesFileName'][row]
    spread_files_name = trade_handler_df['SpreadFilesName'][row]
    name_adder = trade_handler_df['PaperTrade?'][row]
    backtest_pnl_filename = trade_handler_df['BacktestPnLFilename'][row]


    yes_dfs_dict = {}
    upload_dfs_dict = {}
    failed_locations_dict = {}

    if do_data_pull:
        daily_input_data_dict = get_daily_input_data(predict_date_str_mm_dd_yyyy=predict_date_str_mm_dd_yyyy,
                                                     working_directory= working_directory,
                                                     static_directory=static_directory,
                                                     spread_files_name=spread_files_name)

    for iso in current_isos:

        if do_prediction:
            preds_tier1_df, preds_tier2_df, failed_locations_dict[iso] = do_xgb_prediction(predict_date_str_mm_dd_yyyy=predict_date_str_mm_dd_yyyy,
                                                                                           iso=iso,
                                                                                           daily_trade_file_name=daily_trade_file_name,
                                                                                           working_directory= working_directory,
                                                                                           static_directory=static_directory,
                                                                                           model_type=model_type)

        if do_postprocessing:
            trades_df, yes_dfs_dict[iso], upload_dfs_dict[iso] = post_process_trades(predict_date_str_mm_dd_yyyy=predict_date_str_mm_dd_yyyy,
                                                                                      iso=iso,
                                                                                      daily_trade_file_name=daily_trade_file_name,
                                                                                      name_adder=name_adder,
                                                                                      working_directory= working_directory,
                                                                                      static_directory=static_directory,
                                                                                      model_type=model_type)

    if do_postprocessing:
        # Write all upload files to same Excel file
        writer = pd.ExcelWriter(upload_directory + predict_date_str_mm_dd_yyyy + '_UPLOAD_FILES_ALL_' + name_adder + '.xlsx', engine='xlsxwriter',datetime_format='mm/dd/yy')
        for iso, upload_df in upload_dfs_dict.items():
            upload_df.to_excel(writer, sheet_name=iso, index=False)
        writer.save()
        writer.close()


    if do_tradesummary:
        create_trade_summary(predict_date_str_mm_dd_yyyy=predict_date_str_mm_dd_yyyy,
                             isos=current_isos,
                             do_printcharts=do_printcharts,
                             name_adder=name_adder,
                             working_directory= working_directory,
                             static_directory=static_directory)

    if do_daily_PnL:
        daily_PnL(predict_date_str_mm_dd_yyyy=predict_date_str_mm_dd_yyyy,
                  isos=current_isos,
                  name_adder=name_adder,
                  working_directory=root_directory,
                  static_directory=root_directory,
                  do_printcharts=do_printcharts,
                  backtest_pnl_filename=backtest_pnl_filename)


    if do_VAR:
        create_VAR(preds_dict = yes_dfs_dict,
                   VAR_ISOs=VAR_isos,
                   daily_trade_file_name=daily_trade_file_name,
                   working_directory=working_directory,
                   static_directory=static_directory,
                   model_type=model_type,
                   predict_date_str_mm_dd_yyyy=predict_date_str_mm_dd_yyyy,
                   name_adder=name_adder)

    print('')
    print('**********************************************************************************')
    print('')
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ submit trades get rich $$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('')
