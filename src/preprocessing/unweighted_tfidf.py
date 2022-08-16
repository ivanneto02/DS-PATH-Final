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

def create_unweighted_tfidf():
    beg = time.time()
    # Read in data
    print("> Reading DataFrame...")
    df = pd.read_csv(DATA_PATH + "/" + FULL_TABLE_IN_FILE, nrows=NFILES)

    b = time.time()
    # Remove HTML, clean strings, and lemmatize
    print("> Stripping HTML, cleaning up strings, lemmatize...")
    documents = df["raw_html"].apply(remove_html).apply(clean_strings).apply(lemmatize)
    print(len(documents))
    e = time.time()
    print(f"> Time taken: {e - b}s")

    print("> Creating train/test split...")
    x_train = documents#, x_test = train_test_split(documents, test_size=0.0)

    b = time.time()
    print("> Vectorizing the corpus...")
    vectorizer = TfidfVectorizer(max_features=300, stop_words="english", ngram_range=(1, 3))
    vectorizer = vectorizer.fit(x_train)
    vectors = vectorizer.transform(x_train)
    feature_names = vectorizer.get_feature_names()
    e = time.time()
    print(f"> Time taken: {e - b}s")

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
    create_unweighted_tfidf()