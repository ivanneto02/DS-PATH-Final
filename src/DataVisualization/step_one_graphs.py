import matplotlib.pyplot as plt
from datetime import datetime
import os
from config import *
import pandas as pd

from .color_dict import color_dic

def step_one_graphs(df=None):

    print("> Separating CUIs")
    df["present"] = df["CUI"].apply(lambda x: "MAPPED" if not pd.isnull(x) else "None")

    if not os.path.exists(IMAGES_SAVE_PATH):
        os.makedirs(IMAGES_SAVE_PATH)
    
    # All concepts
    print("> Saving step 1 CUIs pie chart")
    plt.figure()
    colors = [ color_dic[p] for p in df["present"].value_counts().index ]
    df["present"].value_counts().plot(kind="pie", autopct=lambda p: f"{p:.2f}% ({p*len(df)/100:.0f})", label="", colors=colors)
    plt.title("Mapped CUI Distribution for all concepts")
    plt.savefig(IMAGES_SAVE_PATH + "/" + f"cuis_step_1_pie_{datetime.now().strftime('%H-%M-%S_%Y-%m-%d')}", bbox_inches='tight', dpi=500)
    plt.close()

    print("> Saving step 1 mapped given sources")
    present_and_sources = pd.crosstab(df.present, df.source_name, normalize=True)
    sources_dist = present_and_sources.sum(axis=0)
    present_given_sources = present_and_sources.divide(sources_dist, axis=1)
    present_given_sources.T.plot(kind="bar", stacked=True, color=color_dic)
    plt.title("Mapping yields given sources")
    plt.xlabel("Source")
    plt.legend(loc="best")
    plt.savefig(IMAGES_SAVE_PATH + "/" + f"present_given_sources_step_1_{datetime.now().strftime('%H-%M-%S_%Y-%m-%d')}", bbox_inches='tight', dpi=500)
    plt.close()

    print("> Saving step 1 mapped given concept type")
    present_and_concept = pd.crosstab(df.present, df.concept_type, normalize=True)
    concept_dist = present_and_concept.sum(axis=0)
    present_given_concept = present_and_concept.divide(concept_dist, axis=1)
    present_given_concept.T.plot(kind="bar", stacked=True, color=color_dic)
    plt.title("Mapping yields given concept type")
    plt.xlabel("Concept Type")
    plt.legend(loc="best")
    plt.savefig(IMAGES_SAVE_PATH + "/" + f"present_given_concept_step_1_{datetime.now().strftime('%H-%M-%S_%Y-%m-%d')}", bbox_inches='tight', dpi=500)
    plt.close()

    print("> Saving step 1 mapped given sources for diseases concept types")
    df_dis = df[df["concept_type"] == "disease"]
    present_and_sources = pd.crosstab(df_dis.present, df_dis.source_name, normalize=True)
    sources_dist = present_and_sources.sum(axis=0)
    present_given_sources = present_and_sources.divide(sources_dist, axis=1)
    present_given_sources.T.plot(kind="bar", stacked=True, color=color_dic)
    plt.title("Mapping yields given sources (if concept_type = disease)")
    plt.xlabel("Source")
    plt.legend(loc="best")
    plt.savefig(IMAGES_SAVE_PATH + "/" + f"present_given_sources_for_diseases_step_1_{datetime.now().strftime('%H-%M-%S_%Y-%m-%d')}", bbox_inches='tight', dpi=500)
    plt.close()

    print("> Saving step 1 mapped given sources for drug concept types")
    df_dru = df[df["concept_type"] == "drug"]
    present_and_sources = pd.crosstab(df_dru.present, df_dru.source_name, normalize=True)
    sources_dist = present_and_sources.sum(axis=0)
    present_given_sources = present_and_sources.divide(sources_dist, axis=1)
    present_given_sources.T.plot(kind="bar", stacked=True, color=color_dic)
    plt.title("Mapping yields given sources (if concept_type = drug)")
    plt.xlabel("Source")
    plt.legend(loc="best")
    plt.savefig(IMAGES_SAVE_PATH + "/" + f"present_given_sources_for_drugs_step_1_{datetime.now().strftime('%H-%M-%S_%Y-%m-%d')}", bbox_inches='tight', dpi=500)
    plt.close()