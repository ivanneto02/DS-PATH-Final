from config import *
import pandas as pd
from src.DataVisualization import source_graphs
from src.DataVisualization import step_one_graphs
from src.DataVisualization import step_two_graphs
from src.DataVisualization import step_three_graphs

def run_visualizations():
    print("> Starting the visualization process")
    # connect to database, return connection
    print("> Reading dataset")
    df = pd.read_csv(BARE_TABLE_SAVE_PATH + "/" + BARE_TABLE_IN_FILE, nrows=None)
    print("> Creating source graphs")
    source_graphs(df=df)

    df_step1 = pd.read_csv(BARE_TABLE_SAVE_PATH + "/" + BARE_TABLE_OUT_FILE_STEP_1, nrows=None)
    print("> Creating CUIs graph (STEP 1)")
    step_one_graphs(df=df_step1)

    df_step2 = pd.read_csv(BARE_TABLE_SAVE_PATH + "/" + BARE_TABLE_OUT_FILE_STEP_2, nrows=None)
    print("> Creating CUIs graph (STEP 2)")
    step_two_graphs(df=df_step2)

    df_step3 = pd.read_csv(BARE_TABLE_SAVE_PATH + "/" + BARE_TABLE_OUT_FILE_STEP_3, nrows=None)
    print("> Creating CUIs graph (STEP 3)")
    step_three_graphs(df=df_step3)

if __name__ == "__main__":
    run_visualizations()