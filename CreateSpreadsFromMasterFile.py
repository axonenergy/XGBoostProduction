import pandas as pd

### This input file should only have the raw darts to be pasted next to the values once spreads are created
darts_df = pd.read_csv('11_06_2019_GBM_DATA_ERCOT_V9.0_MASTER_104F_darts_only.csv')

### This input file should have a table of sources and sinks to run
spreads_to_run_df = pd.read_csv('ercot_top100_spreads.csv')

spread_df = pd.DataFrame()

for row in range(len(spreads_to_run_df)):
    source = spreads_to_run_df.iloc[row,0]
    sink = spreads_to_run_df.iloc[row, 1]
    spread_df[source+'$'+sink] = darts_df[source] - darts_df[sink]

print(spread_df)

spread_df.to_csv('spread_output.csv')