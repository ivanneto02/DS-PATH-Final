
from sklearn.model_selection import train_test_split
from config import *
import pandas as pd
from sklearn.utils import shuffle
import TFIDFClassifier
import os
from sklearn.metrics import classification_report
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 15}
plt.rc('font', **font)

def main():

    begin_time = time.time()

    pd.options.display.max_rows = 100
    pd.options.display.max_columns = 500
    np.set_printoptions(precision=4)

    # Read in data
    print("> Reading DataFrame...")
    df = pd.read_csv(DATA_PATH + "/" + DATA_FILE, nrows=NROWS)

    print("> Shuffling dataset...")
    df = shuffle(df)

    # print("> Fairly splitting...")
    df_dis = df[df["concept_type"] == "disease"]
    df_dru = df[df["concept_type"] == "drug"]
    # n = min(len(df_dru), len(df_dis))
    # df_dis = df_dis[:n]
    # df_dru = df_dru[:n]
    # df = shuffle(pd.concat([df_dis, df_dru]))

    print("> Performing train/test splits...")
    # Separate into train, test, and validation
    train, test = train_test_split(df, test_size=TEST_SPLIT) # 80% train, 20% test 
    train, val = train_test_split(train, test_size=VAL_SPLIT/TRAIN_SPLIT) # Needs to be 10% of entire data, but out of 80% of the data
    # The final goal is 70% train, 20% test, 10% validation
    # Get train, test, and validation data
    x_train, y_train = train.iloc[:, SEP_COLUMN:], pd.get_dummies(train["concept_type"]).astype('float32').values
    x_test, y_test = test.iloc[:, SEP_COLUMN:], pd.get_dummies(test["concept_type"]).astype('float32').values 
    x_val, y_val = val.iloc[:, SEP_COLUMN:], pd.get_dummies(val["concept_type"]).astype('float32').values 

    print("   Splits:")
    print(f"    - train: {len(train)/len(df)}%")
    print(f"    - test: {len(test)/len(df)}%")
    print(f"    - val: {len(val)/len(df)}%")

    print("> Compiling model and training...")
    tfidf_model = TFIDFClassifier.Classifier(num_classes=CLASSES)
    tfidf_model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    tfidf_model.fit(
        x = x_train,
        y = y_train,
        validation_data = (x_val, y_val),
        batch_size = BATCH_SIZE,
        epochs=EPOCHS,
    )

    print("> Saving model...")
    # Make directory if it does not exist
    if not os.path.exists("./saves/models/"):
        os.makedirs("./saves/models/")
    tfidf_model.save_weights('./saves/models/' + TFIDF_MODEL_NAME, save_format='tf')

    print("> Making predictions...")
    y_pred = np.argmax(tfidf_model.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    end_time = time.time()

    plt.figure(figsize=(20, 20))
    print("> Creating confusion matrix...")
    if not os.path.exists("./saves/images/"):
        os.makedirs("./saves/images/")
    labels = pd.get_dummies(train["concept_type"]).astype('float32').columns
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("Drugs vs. Diseases Classification\n")
    plt.savefig(f'./saves/images/tfidf_confusion_matrix_{datetime.now().strftime("%d_%m_%Y_%H-%M-%S")}', bbox_inches="tight")

    print("> Creating report...")
    report = classification_report(y_pred, y_test, digits=4, target_names=labels)
    if not os.path.exists("./saves/reports/"):
        os.makedirs("./saves/reports/")
    with open(f'./saves/reports/tfidf_report_{datetime.now().strftime("%d_%m_%Y_%H-%M-%S")}', "wt") as f_out:
        config = open("config.py", "rt")
        text = config.read()
        config.close()
        f_out.write("Configuration:\n")
        f_out.write(text)
        f_out.write(f"\n================")
        f_out.write(f"\nnsamples: 1.00 ({len(df)})\n")
        f_out.write(f"train: {len(train)/len(df)} ({len(train)})\n")
        f_out.write(f"test: {len(test)/len(df)} ({len(test)})\n")
        f_out.write(f"val: {len(val)/len(df)} ({len(val)})\n")
        f_out.write(f"drug_size: {len(df_dru)/len(df)} ({len(df_dru)})\n")
        f_out.write(f"disease_size: {len(df_dis)/len(df)} ({len(df_dis)})\n")
        f_out.write("Classification Report:\n")
        f_out.write(report)
        f_out.write(f"\nTime taken: {end_time - begin_time}s")

if __name__ == "__main__":
    main()