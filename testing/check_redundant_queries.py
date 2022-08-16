from config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from src.DataVisualization import source_graphs
import os

def check_redundant_queries():
    print("> Starting the visualization process")
    print("> Creating connection")

    # connect to database, return connection
    print("> Reading dataset")
    df = pd.read_csv(DATA_PATH + "/" + FULL_TABLE_IN_FILE, nrows=None)

    print("> Creating images")
    source_graphs(df=df)

if __name__ == "__main__":
    check_redundant_queries()