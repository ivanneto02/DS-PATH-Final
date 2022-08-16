from utils.proceed import proceed
from config import *
import os
import time
import json
import pandas as pd

def compile_csv():
    # do stuff to compile csv
    print("> Compiling csv...")

    # Get all the letter folders
    print(DATA_PATH)
    source_folders = next(os.walk(DATA_PATH))[1]

    total_paths = []
    # Iterate through all of the sources
    for source in source_folders:
        curr = "None"
        if "disease" in source.lower():
            curr = "disease"
        elif "drug" in source.lower():
            curr = "drug"
        # Get all the letter folders
        letter_folders = next(os.walk(DATA_PATH + "/" + source))[1]
        # Iterate through all of the letters
        for letter in letter_folders:
            # get all the files
            files = next(os.walk(DATA_PATH + "/" + source + "/" + letter))[2]
            for fil in files:
                total_paths.append((curr, DATA_PATH + "/" + source + "/" + letter + "/" + fil))

    total_paths = total_paths[:NFILES]

    print("> Printing to make sure this works:")
    for i in range(5):
        print("    ", total_paths[i])
    print(f"> Number of paths: {len(total_paths)}")

    beginning = time.time()
    df = pd.DataFrame()

    print("> Compiling many JSON items (this may take a while)...")
    concat_list = []
    for i in range(len(total_paths)):
        # read the current DataFrame
        try:
            data = json.load(open(total_paths[i][1], "r"))
            curr_df = pd.DataFrame.from_dict(data, orient="index").T
            curr_df["concept_type"] = total_paths[i][0]
            concat_list.append(curr_df)
        except Exception as e:
            print(total_paths[i])
            print(e)

    print(f"> Concatenating into .csv...")
    df = pd.concat(concat_list, axis=0)
    print("> Done concatenating.")

    final = time.time()
    print(f"> Total time: {final - beginning} seconds")

    print(df.head(3))

    beginning = time.time()
    print(f"> Saving full table...")
    df["name"] = df["name"].str.strip()
    df.reset_index()
    df.to_csv(DATA_PATH + "/" + FULL_TABLE_OUT_FILE, index=False)
    print(f"> Saving bare table...")
    df.drop(columns=["raw_html"], inplace=True)
    if not os.path.exists(BARE_TABLE_SAVE_PATH):
        os.makedirs(BARE_TABLE_SAVE_PATH)
    df.to_csv(BARE_TABLE_SAVE_PATH + "/" + BARE_TABLE_OUT_FILE, index=False)
    final = time.time()
    print(f"> Done saving.")
    print(f"> Time taken: {final - beginning}s")

if __name__ == "__main__":
    print("""You are running this program without the usage of `main.py`
    in the root directory. * This program will create a csv file
    that has been processed from the raw data that has been 
    scraped from the internet.\n""")

    if proceed():
        compile_csv()

