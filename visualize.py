
from config import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib as mpl

from sklearn.model_selection import train_test_split

def main():

    plt.style.use('seaborn')

    pd.options.display.max_rows = 100
    # pd.options.display.max_columns = 20

    # Read in data
    print("> Reading DataFrame...")
    df = pd.read_csv(DATA_PATH + "/" + DATA_FILE, nrows=NROWS)

    print(df.iloc[:, SEP_COLUMN:].head(10))

    print(pd.get_dummies(df["concept_type"]).astype('float32').columns)

    return

    # print(len(df))

    # df["concept_type"].value_counts().plot(kind="pie", autopct='%1.1f%%', fontsize=17, label="")
    # plt.title("Distribution of drug vs. disease concepts")
    # plt.show()

    # df["concept_type"].value_counts().plot(kind="bar", fontsize=17, rot=0)
    # plt.title("Distribution of drug vs. disease concepts")
    # plt.show()

    mini = min(len(df[df["concept_type"] == "disease"].iloc[:, SEP_COLUMN:SEP_COLUMN+10]), len(df[df["concept_type"] == "drug"].iloc[:, SEP_COLUMN:SEP_COLUMN+10]))
    features = 30

    print(df[df["concept_type"] == "disease"].iloc[:, SEP_COLUMN:SEP_COLUMN+features].head(mini).sum(axis=0))
    print(df[df["concept_type"] == "drug"].iloc[:, SEP_COLUMN:SEP_COLUMN+features].head(mini).sum(axis=0))

    return

    # Separate into train, test, and validation
    train, test = train_test_split(df, test_size=TEST_SPLIT) # 80% train, 20% test 
    train, val = train_test_split(train, test_size=VAL_SPLIT/TRAIN_SPLIT) # Needs to be 10% of entire data, but out of 80% of the data
    # The final goal is 70% train, 20% test, 10% validation

    # Get train, test, and validation data
    x_train, y_train = train.iloc[:, SEP_COLUMN:], train["concept_type"]
    x_test, y_test = test.iloc[:, SEP_COLUMN:], test["concept_type"]
    x_val, y_val = val.iloc[:, SEP_COLUMN:], val["concept_type"]

    print("Splits:")
    print(f" - train: {len(train)/len(df)}%")
    print(f" - test: {len(test)/len(df)}%")
    print(f" - val: {len(val)/len(df)}%")

    return

    print("> Shuffling...")
    df = shuffle(df)

    print("> Fairly splitting...")
    df_dis = df[df["concept_type"] == "disease"]
    df_dru = df[df["concept_type"] == "drug"]
    n = min(len(df_dru), len(df_dis))
    df_dis = df_dis[:n//2]
    df_dru = df_dru[:n//2]
    df = shuffle(pd.concat([df_dis, df_dru]))

    print("> Plotting...")
    df["concept_type"].value_counts().plot(kind="bar")
    plt.show()

if __name__ == "__main__":
    main()