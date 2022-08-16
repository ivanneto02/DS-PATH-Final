import sys
sys.path.insert(0, "")
from utils.proceed import proceed
from config import *
import pandas as pd
import time

'''
Currently not working very well. Will have to fix in the future.
- Ivan
'''

def name_fixes(name):
    return name

def fix_names():
    # do stuff to gather definitions
    print("> Fixing the source column...")
    df = pd.read_csv(DATA_PATH + "/" + FULL_TABLE_IN_FILE, nrows=None)

    # Apply fixes
    df["names"] = df.apply(lambda x : name_fixes(x["name"]), axis=1)

    # Save
    print("> Saving full table")
    df.to_csv(DATA_PATH + "/" + FULL_TABLE_OUT_FILE, index=False)
    print("> Saving bare table")
    df.drop(columns=["raw_html"], inplace=True)
    if not os.path.exists(BARE_TABLE_SAVE_PATH):
        os.makedirs(BARE_TABLE_SAVE_PATH)
    df.to_csv(BARE_TABLE_SAVE_PATH + "/" + BARE_TABLE_OUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    print("""You are running this program without the usage of `main.py`
    in the root directory. * This program will separate the name of the
    drugs from the definitions of the drugs, in the data that has been
    scraped from the internet.\n""")

    if proceed():
        fix_names()