import pandas as pd
from config import *

def display_data():
    df = pd.read_csv(BARE_TABLE_SAVE_PATH + "/" + BARE_TABLE_IN_FILE, nrows=None)
    
    for index, row in df[ df["name"].str.contains("Alzheimer", regex=False) ]["name"].iteritems():
        print(row)
    print("Length:", len(df[ df["name"].str.contains("'Alzheimer's", regex=False) ]["name"]))

if __name__ == "__main__":
    display_data()