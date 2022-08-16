import pandas as pd
from config import *

def test_cui_dist():
    df = pd.read_csv(BARE_TABLE_SAVE_PATH + "/" + BARE_TABLE_IN_FILE, nrows=None)

    print(df.head(10))

    cui_df = df.dropna(subset=["CUI"])
    
    print(f"Percentage added: {(len(cui_df) / len(df)) * 100}% ({len(cui_df)} / {len(df)})")

if __name__ == "__main__":
    test_cui_dist()