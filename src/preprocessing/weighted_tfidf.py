from distutils.command.clean import clean
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from config import *
import pandas as pd
import re
import nltk
from sklearn.utils import shuffle
import time

nltk.download("wordnet")
nltk.download("omw-1.4")

def remove_html(x):
    soup = BeautifulSoup(x, "lxml")
    for data in soup(["style", "script"]):
        data.decompose()
    return " ".join(soup.stripped_strings)

def clean_strings(x):
    x = re.sub(r"\d", "", x)
    return x

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize(x):
    return " ".join([ lemmatizer.lemmatize(word) for word in w_tokenizer.tokenize(x) ])

def create_weighted_tfidf():
    return # TODO
    beg = time.time()
    # Read in data
    print("> Reading DataFrame...")
    df = pd.read_csv(DATA_PATH + "/" + FULL_TABLE_IN_FILE, nrows=NFILES)

    print("> Shuffling...")
    df = shuffle(df)

    df_dru = df[df["concept_type"] == "drug"]
    df_dis = df[df["concept_type"] == "disease"]

    dru_features = FEATURES*(len(df_dru)/len(df))
    dis_features = FEATURES*(len(df_dis)/len(df))

    df["doc"] = df["raw_html"].apply(remove_html).apply(clean_strings).apply(lemmatize).reset_index(drop=True)

    b = time.time()
    print("> Vectorizing drug concepts...")
    dru_vectorizer = TfidfVectorizer(max_features=dru_features, stop_words="english", ngram_range=(1, 3))
    dru_vectorizer = dru_vectorizer.fit(dru_docs)
    dru_vectors = dru_vectorizer.transform(df)
    dru_features = dru_vectorizer.get_feature_names()
    e = time.time()
    print(f"> Time taken: {e - b}s")
    b = time.time()
    print("> Vectorizing disease concepts...")
    dis_vectorizer = TfidfVectorizer(max_features=dis_features, stop_words="english", ngram_range=(1, 3))
    dis_vectorizer = dis_vectorizer.fit(dis_docs)
    dis_vectors = dis_vectorizer.transform(dis_docs)
    dis_features = dis_vectorizer.get_feature_names()
    e = time.time()
    print(f"> Time taken: {e - b}s")

    print("> Making Results DataFrame...")
    dru_results = pd.DataFrame(dru_vectors.toarray(), columns=dru_features).reset_index(drop=True)
    dis_results = pd.DataFrame(dis_vectors.toarray(), columns=dis_features).reset_index(drop=True)
    df_dru = df_dru.reset_index(drop=True)
    df_dis = df_dis.reset_index(drop=True)
    df = df.reset_index(drop=True)
    results = pd.concat([dru_results, dis_results], axis=1)
    df = pd.concat([df, results], axis=1)

    b = time.time()
    print("> Making Results DataFrame...")
    results = pd.DataFrame(vectors.toarray(), columns=feature_names)
    df = df.reset_index(drop=True, inplace=False)
    results = results.reset_index(drop=True, inplace=False)
    df = pd.concat([df, results], axis=1)
    e = time.time()
    print(f"> Time taken: {e - b}s")

    b = time.time()
    print("> Saving...")
    df.to_csv(DATA_PATH + "/" + FULL_TABLE_OUT_FILE, index=False)
    print("> Done!")
    end = time.time()

    print(f"> Time taken: {end - b}s")
    print(f"> Total time taken: {end - beg}s")

if __name__ == "__main__":
    create_weighted_tfidf()