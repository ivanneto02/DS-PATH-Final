import pandas as pd
from config import *
import matplotlib.pyplot as plt

def visualize_concepts():
    df = pd.read_csv(DATA_PATH + "/" + FULL_TABLE_IN_FILE, nrows=None)
    print(df.head(20))

if __name__ == "__main__":
    visualize_concepts()