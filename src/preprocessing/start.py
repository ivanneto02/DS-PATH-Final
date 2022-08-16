from venv import create
from .compile_csv import compile_csv
from .fix_sources import fix_sources
from .gather_definitions import gather_definitions
from .unweighted_tfidf import create_unweighted_tfidf
from .weighted_tfidf import create_weighted_tfidf

from src.ConceptTableBuilder import map_to_umls

from utils import proceed

def selection_prompt():
    print("Running the preprocessing package.")
    print("Package steps:")
    print("    - compile_csv")
    print("    - gather_definitions")
    print("    - create_weighted_tfidf")
    print("    - create_unweighted_tfidf")
    print("    - create")
    print("Options: ")
    print("    1. Run entire package")
    print("    2. Run package subset")

    inp = input("\nChoice: ")
    while (inp != "1") and (inp != "2"):
        print("Wrong input! Try again.")
        inp = input("Choice: ")
    
    inp = int(inp)
    if inp == 1:
        start()
    elif inp == 2:
        start_subset()
    else:
        print("Something went wrong.")
    print("Package done running.")

'''Runs entire package'''
def start(): 
    # compile csv
    compile_csv()

    # gather definitions
    gather_definitions() 

    # create unweighted TFIDF scores
    create_weighted_tfidf() 

    # create weighted TFIDF scores
    create_unweighted_tfidf() 

    # map cuis
    map_to_umls()

    # create relational tables

'''Runs user selected subset of package'''
def start_subset():
    subset_in_order = [compile_csv, fix_sources, map_to_umls] # empty = no subset
    for function in subset_in_order:
        function()

if __name__ == "__main__":
    print("You are running the main program that will preprocess the raw data.")

    if proceed():
        start()