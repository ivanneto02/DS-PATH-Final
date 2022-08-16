import pandas as pd
from config import *

def print_fewlines():
    df = pd.read_csv(DATA_PATH + "/" + OUT_FILE, nrows=10)

    # print(df[df["name"].str.lower == "Atrial Fibrillation"])

    # return

    for i in range(5):
        print(df["definition"].values[i])
        print("==============================")

    print(df.head(10))

if __name__ == "__main__":
    print_fewlines()